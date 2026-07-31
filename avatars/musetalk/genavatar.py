import argparse
import glob
import json
import math
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from diffusers import AutoencoderKL
# from face_alignment import NetworkSize
# from mmpose.apis import inference_topdown, init_model
# from mmpose.structures import merge_data_samples
from tqdm import tqdm

from avatars.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from avatars.musetalk.utils.blending import get_image_prepare_material
from avatars.musetalk.utils.utils import load_all_model

try:
    from utils.face_parsing import FaceParsing
except ModuleNotFoundError:
    from avatars.musetalk.utils.face_parsing import FaceParsing


vae = None
fp = None
_creator_config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000, max_frames=0, output_fps=25):
    """Extract frames from `vid_path` into `save_path` as 00000000.png, ...

    Samples evenly-spaced frames so the avatar's reference-frame loop plays at
    the SOURCE video's real-time speed. The avatar outputs at a fixed
    `output_fps` (25 — config requires `--fps 25`), so one reference loop lasts
    N/output_fps seconds. To match the source duration we want
    N ~= duration * output_fps = (total/src_fps) * output_fps frames, i.e.
    stride ~= src_fps / output_fps. A 60fps source -> stride 3 (~real-time at
    25fps output); a 25fps source -> stride 1 (already real-time).

    `max_frames` is a HARD ceiling: if the source is so long that real-time
    sampling would exceed it, sample sparser (faster-than-realtime motion)
    rather than build a multi-thousand-frame avatar. This is what bounds
    creation time / disk / per-session build (a 60fps x 68s source = 4084 frames
    -> creation ~20-30min, killed mid-save -> broken 0-byte latents.pt).

    MuseTalk lip-sync is driven by the audio feature, NOT the reference frames
    — frames only supply identity / pose / background — so even a sub-realtime
    loop is visually fine; only the head/pose/background loop speed changes.
    `cut_frame` is a hard upper bound on the source frame index. Returns the
    number of frames written.
    """
    cap = cv2.VideoCapture(vid_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_fps = float(src_fps) if (src_fps and src_fps > 0) else 0.0

    # Real-time target: frames needed so N/output_fps == source duration.
    target = 0
    if src_fps > 0 and total > 0:
        duration = total / src_fps
        target = max(1, int(round(duration * output_fps)))

    # cap_target = hard max number of frames to write this run.
    if max_frames and max_frames > 0:
        cap_target = min(target, max_frames) if target > 0 else max_frames
    else:
        cap_target = target  # 0 -> unlimited (legacy all-frames)

    # Even spacing to land on ~cap_target frames. round-half-up (not ceil):
    # ceil over-strides and loses ~20% of frames, leaving the loop ~1.25x fast
    # even when the cap allows real-time. round-half-up packs ~cap_target frames
    # so the loop duration matches the source when cap >= target; when cap <
    # target (cap-bound) it behaves like the old ceil (within +/-1). If total is
    # 0 (some HEVC containers misreport frame count) stride stays 1 and the
    # written-cap hard-stop below still bounds the output.
    stride = 1
    if cap_target > 0 and total > cap_target:
        stride = max(1, int(total / cap_target + 0.5))

    src_dur = (total / src_fps) if (src_fps > 0 and total > 0) else 0.0
    print(
        f"video2imgs: src total={total} fps={src_fps or 'n/a'} dur={src_dur:.1f}s "
        f"-> realtime_target={target or 'n/a'} cap={cap_target or 'unlimited'} "
        f"stride={stride}"
    )

    count = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count > cut_frame:
            break
        if count % stride == 0:
            # Stop once we have enough frames. Covers the HEVC case where
            # total misreported as 0 -> stride stayed 1 -> without this a
            # 4084-frame stream would still write every frame.
            if cap_target > 0 and written >= cap_target:
                break
            cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            cv2.imwrite(f"{save_path}/{written:08d}.png", frame)
            written += 1
        count += 1
    cap.release()
    print(f"video2imgs: wrote {written} frames")
    return written

'''
def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]  # 手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            f_landmark = (
                np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]),
                np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w, h = f[2] - f[0], f[3] - f[1]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]
    return coords_list, frames


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
            #             torch.backends.cuda.matmul.allow_tf32 = False
            #             torch.backends.cudnn.benchmark = True
            #             torch.backends.cudnn.deterministic = False
            #             torch.backends.cudnn.allow_tf32 = True
            print('cuda start')

        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)

        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))
        return results


def get_mask_tensor():
    """
    Creates a mask tensor for image processing.
    :return: A mask tensor.
    """
    mask_tensor = torch.zeros((256, 256))
    mask_tensor[:256 // 2, :] = 1
    mask_tensor[mask_tensor < 0.5] = 0
    mask_tensor[mask_tensor >= 0.5] = 1
    return mask_tensor


def preprocess_img(img_name, half_mask=False):
    window = []
    if isinstance(img_name, str):
        window_fnames = [img_name]
        for fname in window_fnames:
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256),
                             interpolation=cv2.INTER_LANCZOS4)
            window.append(img)
    else:
        img = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
        window.append(img)
    x = np.asarray(window) / 255.
    x = np.transpose(x, (3, 0, 1, 2))
    x = torch.squeeze(torch.FloatTensor(x))
    if half_mask:
        x = x * (get_mask_tensor() > 0.5)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    x = normalize(x)
    x = x.unsqueeze(0)  # [1, 3, 256, 256] torch tensor
    x = x.to(device)
    return x


def encode_latents(image):
    with torch.no_grad():
        init_latent_dist = vae.encode(image.to(vae.dtype)).latent_dist
    init_latents = vae.config.scaling_factor * init_latent_dist.sample()
    return init_latents


def get_latents_for_unet(img):
    ref_image = preprocess_img(img, half_mask=True)  # [1, 3, 256, 256] RGB, torch tensor
    masked_latents = encode_latents(ref_image)  # [1, 4, 32, 32], torch tensor
    ref_image = preprocess_img(img, half_mask=False)  # [1, 3, 256, 256] RGB, torch tensor
    ref_latents = encode_latents(ref_image)  # [1, 4, 32, 32], torch tensor
    latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
    return latent_model_input


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s


def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.2):
    body = Image.fromarray(image[:, :, ::-1])

    x, y, x1, y1 = face_box
    # print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
'''

##todo 简单根据文件后缀判断  要更精确的可以自己修改 使用 magic
def is_video_file(file_path):
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']  # 这里列出了一些常见的视频文件扩展名，可以根据需要添加更多
    file_ext = os.path.splitext(file_path)[1].lower()  # 获取文件扩展名并转换为小写
    return file_ext in video_exts


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


current_dir = './' #os.path.dirname(os.path.abspath(__file__))


def initialize_musetalk_creator(
    version="v15",
    gpu_id=0,
    bbox_shift=0,
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
):
    global vae, fp, _creator_config, device

    config = {
        "version": version,
        "gpu_id": int(gpu_id),
        "bbox_shift": int(bbox_shift),
        "extra_margin": int(extra_margin),
        "parsing_mode": parsing_mode,
        "left_cheek_width": int(left_cheek_width),
        "right_cheek_width": int(right_cheek_width),
    }
    if vae is not None and fp is not None and _creator_config == config:
        return vae, fp

    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")

    loaded_vae, _, _ = load_all_model(device=device)
    loaded_vae.vae = loaded_vae.vae.half().to(device)

    if config["version"] == "v15":
        loaded_fp = FaceParsing(
            left_cheek_width=config["left_cheek_width"],
            right_cheek_width=config["right_cheek_width"],
        )
    else:
        loaded_fp = FaceParsing()

    vae = loaded_vae
    fp = loaded_fp
    _creator_config = config
    return vae, fp


def create_musetalk_human(
    file,
    avatar_id,
    img_size=256,
    version="v15",
    bbox_shift=0,
    extra_margin=10,
    parsing_mode="jaw",
    gpu_id=0,
    left_cheek_width=90,
    right_cheek_width=90,
):
    initialize_musetalk_creator(
        version=version,
        gpu_id=gpu_id,
        bbox_shift=bbox_shift,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
    )
    # 保存文件设置 可以不动
    save_path = os.path.join(current_dir, f'./data/avatars/{avatar_id}')
    save_full_path = os.path.join(current_dir, f'./data/avatars/{avatar_id}/full_imgs')
    create_dir(save_path)
    create_dir(save_full_path)
    mask_out_path = os.path.join(current_dir, f'./data/avatars/{avatar_id}/mask')
    create_dir(mask_out_path)

    # 模型
    mask_coords_path = os.path.join(current_dir, f'{save_path}/mask_coords.pkl')
    coords_path = os.path.join(current_dir, f'{save_path}/coords.pkl')
    latents_out_path = os.path.join(current_dir, f'{save_path}/latents.pt')

    with open(os.path.join(current_dir, f'{save_path}/avator_info.json'), "w") as f:
        json.dump({
            "avatar_id": avatar_id,
            "video_path": file,
            "bbox_shift": bbox_shift,
            "version": version,
            "parsing_mode": parsing_mode,
        }, f)

    if os.path.isfile(file):
        if is_video_file(file):
            # Cap the extracted frame count so a long/high-fps source does not
            # become a multi-thousand-frame avatar (creation ~20-30min, killed
            # mid-save -> broken avatar). See video2imgs docstring. Default 600
            # (~24s pose loop at 25fps) — more pose variety than 300 with a
            # still-reasonable ~4-5min creation. Override via the env var.
            max_frames = int(os.getenv("LIVETALKING_AVATAR_MAX_FRAMES", "600"))
            output_fps = int(os.getenv("LIVETALKING_AVATAR_OUTPUT_FPS", "25"))
            written = video2imgs(file, save_full_path, ext='png',
                                 max_frames=max_frames, output_fps=output_fps)
        else:
            shutil.copyfile(file, f"{save_full_path}/{os.path.basename(file)}")
    else:
        files = os.listdir(file)
        files.sort()
        files = [file for file in files if file.split(".")[-1] == "png"]
        for filename in files:
            shutil.copyfile(f"{file}/{filename}", f"{save_full_path}/{filename}")
    input_img_list = sorted(glob.glob(os.path.join(save_full_path, '*.[jpJP][pnPN]*[gG]')))
    print("extracting landmarks...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    input_latent_list = []
    idx = -1
    # maker if the bbox is not sufficient
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    for bbox, frame in zip(coord_list, frame_list):
        idx = idx + 1
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        if version == "v15":
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)

    frame_list_cycle = frame_list #+ frame_list[::-1]
    coord_list_cycle = coord_list #+ coord_list[::-1]
    input_latent_list_cycle = input_latent_list #+ input_latent_list[::-1]
    mask_coords_list_cycle = []
    mask_list_cycle = []
    for i, frame in enumerate(tqdm(frame_list_cycle)):
        cv2.imwrite(f"{save_full_path}/{str(i).zfill(8)}.png", frame)

        x1, y1, x2, y2 = coord_list_cycle[i]
        if version == "v15":
            mode = parsing_mode
        else:
            mode = "raw"
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)

        mask_coords_list_cycle += [crop_box]
        mask_list_cycle.append(mask)

    # Remove the originally-copied source frames. For image / directory inputs
    # the source file(s) were copied into full_imgs/ with their original names
    # (see above); the loop just rewrote zero-padded copies, so BOTH now sit in
    # the dir. load_avatar globs full_imgs/* which would then be ~2x the
    # latent/coord/mask count -> a broken avatar that crashes at inference.
    # Keep only the zero-padded 00000000.png .. NNNNNNNN.png written by the loop.
    for f in glob.glob(os.path.join(save_full_path, '*')):
        b = os.path.basename(f)
        if not (len(b) == 12 and b.endswith('.png') and b[:8].isdigit()):
            try:
                os.remove(f)
            except OSError:
                pass

    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list_cycle, f)
    torch.save(input_latent_list_cycle, os.path.join(latents_out_path))


# initialize the mmpose model
# device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
# fa = FaceAlignment(1, flip_input=False, device=device)
# config_file = os.path.join(current_dir, 'utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
# checkpoint_file = os.path.abspath(os.path.join(current_dir, '../models/dwpose/dw-ll_ucoco_384.pth'))
# model = init_model(config_file, checkpoint_file, device=device)
# vae = AutoencoderKL.from_pretrained(os.path.abspath(os.path.join(current_dir, '../models/sd-vae-ft-mse')))
# vae.to(device)
# fp = FaceParsing(os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/resnet18-5c106cde.pth')),
#                  os.path.abspath(os.path.join(current_dir, '../models/face-parse-bisent/79999_iter.pth')))
if __name__ == '__main__':
    # 视频文件地址
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",
                        type=str,
                        default=r'D:\ok\00000000.png',
                        )
    parser.add_argument("--avatar_id",
                        type=str,
                        default='musetalk_avatar1',
                        )
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    args = parser.parse_args()

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model weights
    vae, unet, pe = load_all_model(
        device=device
    )
    vae.vae = vae.vae.half().to(device)
    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()

    _creator_config = {
        "version": args.version,
        "gpu_id": args.gpu_id,
        "bbox_shift": args.bbox_shift,
        "extra_margin": args.extra_margin,
        "parsing_mode": args.parsing_mode,
        "left_cheek_width": args.left_cheek_width,
        "right_cheek_width": args.right_cheek_width,
    }

    create_musetalk_human(
        args.file,
        args.avatar_id,
        version=args.version,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        gpu_id=args.gpu_id,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )
