import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm

def video2imgs(vid_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
        cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames to {save_path}")
    return count

def detect_faces(img_list):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    coords_list = []
    prev_bbox = None
    
    for img_path in tqdm(img_list, desc="Detecting faces"):
        frame = cv2.imread(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) > 0:
            # Pick largest face
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            x1, y1, x2, y2 = x, y, x+w, y+h
            prev_bbox = (x1, y1, x2, y2)
        elif prev_bbox is not None:
            x1, y1, x2, y2 = prev_bbox
        else:
            # Fallback: center crop
            h, w = frame.shape[:2]
            cx, cy = w//2, h//2
            size = min(w, h)//3
            x1, y1, x2, y2 = cx-size, cy-size, cx+size, cy+size
            prev_bbox = (x1, y1, x2, y2)
        
        coords_list.append((y1, y2, x1, x2))
    
    return coords_list

def smooth_coords(coords_list, window=5):
    if len(coords_list) == 0:
        return coords_list
    arr = np.array(coords_list, dtype=np.float32)
    smoothed = arr.copy()
    half = window // 2
    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        smoothed[i] = np.mean(arr[start:end], axis=0)
    return [tuple(map(int, row)) for row in smoothed]

def create_avatar(video_path, avatar_id, img_size=96):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    
    os.makedirs(avatar_path, exist_ok=True)
    os.makedirs(full_imgs_path, exist_ok=True)
    os.makedirs(face_imgs_path, exist_ok=True)
    
    # Step 1: extract frames
    video2imgs(video_path, full_imgs_path)
    
    # Step 2: list frames
    input_img_list = sorted([os.path.join(full_imgs_path, f) for f in os.listdir(full_imgs_path) if f.endswith('.png')])
    
    # Step 3: detect faces
    coords_list = detect_faces(input_img_list)
    coords_list = smooth_coords(coords_list, window=5)
    
    # Step 4: crop faces
    for i, (img_path, coords) in enumerate(tqdm(zip(input_img_list, coords_list), desc="Cropping faces")):
        frame = cv2.imread(img_path)
        y1, y2, x1, x2 = coords
        # Ensure within bounds
        h, w = frame.shape[:2]
        y1 = max(0, y1)
        y2 = min(h, y2)
        x1 = max(0, x1)
        x2 = min(w, x2)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            # fallback to a safe region
            face_crop = frame[h//4:3*h//4, w//4:3*w//4]
        
        resized = cv2.resize(face_crop, (img_size, img_size))
        cv2.imwrite(f"{face_imgs_path}/{i:08d}.png", resized)
    
    # Step 5: save coords
    with open(coords_path, 'wb') as f:
        pickle.dump(coords_list, f)
    
    print(f"Avatar '{avatar_id}' created at {avatar_path}")
    print(f"  Frames: {len(input_img_list)}")
    print(f"  Face size: {img_size}x{img_size}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--avatar_id', required=True, type=str)
    parser.add_argument('--img_size', type=int, default=96)
    args = parser.parse_args()
    
    create_avatar(args.video_path, args.avatar_id, args.img_size)
