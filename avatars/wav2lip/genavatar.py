from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
from avatars.wav2lip import face_detection


def is_video_file(file_path):
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_exts


def osmakedirs(path_list):
    for p in path_list:
        os.makedirs(p, exist_ok=True)


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, pads, nosmooth, face_det_batch_size, device):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                             flip_input=False, device=device)

    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1: x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def create_wav2lip_human(
    file,
    avatar_id,
    img_size=256,
    pads=(0, 10, 0, 0),
    face_det_batch_size=16,
    nosmooth=False,
    gpu_id=0,
):
    """Build a Wav2Lip avatar from a video / image / directory of images.

    Writes ./data/avatars/<avatar_id>/{full_imgs, face_imgs, coords.pkl} which
    is exactly what avatars/wav2lip_avatar.py load_avatar expects. Callable as
    an import (no argparse globals); the CLI below wraps it for `python -m`.
    """
    # NOTE: face_detection.FaceAlignment does `if 'cuda' in device` (api.py:56),
    # which requires a STRING, not a torch.device object — passing torch.device
    # raises "argument of type 'torch.device' is not iterable". Keep it a string
    # (this is what wav2lip's original inference.py does). .to(device) accepts
    # strings fine downstream.
    device = f"cuda:{int(gpu_id)}" if torch.cuda.is_available() else "cpu"
    print('Using {} for inference.'.format(device))

    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    osmakedirs([avatar_path, full_imgs_path, face_imgs_path])

    with open(f"{avatar_path}/avator_info.json", "w") as f:
        json.dump({
            "avatar_id": avatar_id,
            "video_path": file,
            "pads": list(pads),
            "img_size": int(img_size),
            "face_det_batch_size": int(face_det_batch_size),
            "nosmooth": bool(nosmooth),
        }, f)

    # Source frames -> full_imgs/. Only zero-padded 00000000.png names are
    # written, so there is no leftover-source-frame bug (unlike musetalk's
    # image/dir path which copies originals alongside padded copies).
    if os.path.isfile(file):
        if is_video_file(file):
            video2imgs(file, full_imgs_path, ext='png')
        else:
            img = cv2.imread(file)
            if img is None:
                raise ValueError(f"Cannot read image: {file}")
            cv2.imwrite(f"{full_imgs_path}/00000000.png", img)
    elif os.path.isdir(file):
        files = sorted(glob(os.path.join(file, '*.[jpJP][pnPN]*[gG]')))
        if not files:
            raise ValueError(f"No images in directory: {file}")
        for i, fp in enumerate(files):
            img = cv2.imread(fp)
            if img is None:
                continue
            cv2.imwrite(f"{full_imgs_path}/{i:08d}.png", img)
    else:
        raise ValueError(f"Source not found: {file}")

    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
    if not input_img_list:
        raise ValueError("No frames produced from the source")

    frames = read_imgs(input_img_list)
    face_det_results = face_detect(frames, pads, nosmooth, face_det_batch_size, device)

    coord_list = []
    for idx, (face_crop, coords) in enumerate(face_det_results):
        resized_crop_frame = cv2.resize(face_crop, (int(img_size), int(img_size)))
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append(coords)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)

    print(f"Wav2Lip avatar '{avatar_id}' created: {len(coord_list)} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    # 256 matches the wav2lip.pth checkpoint (wav2lip_v2 trains on 256x256).
    # See avatars/wav2lip_avatar.py WAV2LIP_FACE_RES.
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    create_wav2lip_human(
        args.video_path,
        args.avatar_id,
        img_size=args.img_size,
        pads=tuple(args.pads),
        face_det_batch_size=args.face_det_batch_size,
        nosmooth=args.nosmooth,
        gpu_id=args.gpu_id,
    )