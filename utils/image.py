
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# def read_imgs(img_list):
#     frames = []
#     logger.info('reading images...')
#     for img_path in tqdm(img_list):
#         frame = cv2.imread(img_path)
#         frames.append(frame)
#     return frames

def read_imgs(img_list):
    def load_image(index, img_path):
        return index, cv2.imread(img_path)

    frames = [None] * len(img_list)  # Initialize a list with the same length as img_list
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, idx, img_path): idx for idx, img_path in enumerate(img_list)}
        for future in tqdm(as_completed(futures), total=len(img_list)):
            idx, img = future.result()
            frames[idx] = img
    return frames

def mirror_index(size, index):
    # Accept a fractional index (float accumulator) so the caller can advance
    # the body frame by a non-integer step per output frame (e.g. 0.6 to play a
    # 15fps-extracted avatar at real-time on a 25fps output). Truncate to int.
    index = int(index)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


def forward_loop_index(size, index, start=0, end=None):
    """Map an increasing counter onto a forward-only [start, end) loop."""
    if size <= 0:
        return 0
    start = max(0, min(int(start), size - 1))
    end = size if end is None else max(start + 1, min(int(end), size))
    return start + (int(index) % (end - start))


def analyze_loop_frames(frames, fps=25, min_seconds=8.0, max_seconds=16.0):
    """Find a stable forward-only loop and report abrupt source transitions.

    The selected range is intentionally shorter than a gesture-heavy source.
    Its first and last frames are visually similar, reducing the wrap jump
    without synthesizing blended frames (which can create double-image ghosts).
    """
    count = len(frames)
    if count == 0:
        return {"mode": "forward", "start": 0, "end": 0, "frame_count": 0}

    descriptors = []
    for frame in frames:
        if frame is None:
            descriptors.append(None)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        descriptors.append(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32))

    valid = next((item for item in descriptors if item is not None), np.zeros((64, 64), np.float32))
    descriptors = [valid if item is None else item for item in descriptors]
    motion = np.array([
        float(np.mean(np.abs(descriptors[i] - descriptors[i - 1])))
        for i in range(1, count)
    ], dtype=np.float32)

    fps = max(1.0, float(fps))
    min_len = min(count, max(2, int(round(min_seconds * fps))))
    max_len = min(count, max(min_len, int(round(max_seconds * fps))))
    best = (float("inf"), 0, count)
    length_step = max(1, int(round(fps)))
    candidate_lengths = list(range(min_len, max_len + 1, length_step))
    if max_len not in candidate_lengths:
        candidate_lengths.append(max_len)

    for length in candidate_lengths:
        start_step = max(1, int(round(fps / 5.0)))
        for start in range(0, count - length + 1, start_step):
            end = start + length
            seam = float(np.mean(np.abs(descriptors[end - 1] - descriptors[start])))
            local = motion[start:end - 1]
            mean_motion = float(np.mean(local)) if local.size else 0.0
            p95_motion = float(np.percentile(local, 95)) if local.size else 0.0
            score = seam * 2.5 + p95_motion * 0.8 + mean_motion * 0.2
            if score < best[0]:
                best = (score, start, end)

    _, start, end = best
    seam = float(np.mean(np.abs(descriptors[end - 1] - descriptors[start])))
    if motion.size:
        median = float(np.median(motion))
        mad = float(np.median(np.abs(motion - median)))
        threshold = max(float(np.percentile(motion, 99)), median + 6.0 * max(mad, 0.05))
        abrupt = np.flatnonzero(motion >= threshold)
        abrupt_count = int(abrupt.size)
        abrupt = sorted(abrupt, key=lambda i: float(motion[i]), reverse=True)[:12]
        abrupt_transitions = [
            {"from": int(i), "to": int(i + 1), "mean_abs_diff": round(float(motion[i]), 4)}
            for i in abrupt
        ]
    else:
        threshold = 0.0
        abrupt_count = 0
        abrupt_transitions = []

    return {
        "mode": "forward",
        "start": int(start),
        "end": int(end),
        "frame_count": int(count),
        "loop_frames": int(end - start),
        "loop_seconds": round((end - start) / fps, 3),
        "seam_mean_abs_diff": round(seam, 4),
        "abrupt_threshold": round(threshold, 4),
        "abrupt_transition_count": abrupt_count,
        "abrupt_transitions": abrupt_transitions,
    }
