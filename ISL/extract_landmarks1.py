"""
Stage 1: ISL video -> MediaPipe landmarks (225/frame) -> normalized -> resampled to 64 frames.

Per frame: 33 pose + 21 left hand + 21 right hand = 75 landmarks x (x, y, z) = 225 values.
(Face mesh is dropped on purpose to keep the vector at 225.)

Install:   pip install mediapipe==0.10.14 opencv-python numpy tqdm
Run:       python extract_landmarks.py --input_dir videos --output_dir landmarks
Output:    one <video_name>.npy per video, shape (64, 225), float32
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

N_FRAMES = 64
L_SHOULDER, R_SHOULDER = 11, 12  # pose landmark indices
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

mp_holistic = mp.solutions.holistic


def _to_array(landmark_list, n):
    """Landmark list -> (n, 3). Missing detection -> zeros (and a False flag)."""
    if landmark_list is None:
        return np.zeros((n, 3), dtype=np.float32), False
    arr = np.array([[l.x, l.y, l.z] for l in landmark_list.landmark], dtype=np.float32)
    return arr, True


def extract_raw(video_path, holistic):
    """Return raw landmarks (T, 75, 3) and a presence mask (T, 75)."""
    cap = cv2.VideoCapture(str(video_path))
    frames, masks = [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pose, p_ok = _to_array(res.pose_landmarks, 33)
        lh, l_ok = _to_array(res.left_hand_landmarks, 21)
        rh, r_ok = _to_array(res.right_hand_landmarks, 21)
        frames.append(np.concatenate([pose, lh, rh], axis=0))
        masks.append(np.concatenate([
            np.full(33, p_ok), np.full(21, l_ok), np.full(21, r_ok)
        ]))
    cap.release()
    if not frames:
        return None, None
    return np.stack(frames), np.stack(masks)


def normalize(lm, mask):
    """
    Remove signer position and size.
    - Translation: subtract the shoulder midpoint (chest reference) per frame.
    - Scale: divide by the video's median shoulder width (stable across frames).
    Missing landmarks stay exactly zero so the model can tell they're absent.
    """
    ls, rs = lm[:, L_SHOULDER], lm[:, R_SHOULDER]            # (T, 3)
    valid = mask[:, L_SHOULDER] & mask[:, R_SHOULDER]
    if valid.sum() == 0:
        return None
    center = (ls + rs) / 2.0                                  # (T, 3)
    # forward-fill center for frames where pose was lost
    last = center[valid][0]
    for t in range(len(center)):
        if valid[t]:
            last = center[t]
        else:
            center[t] = last
    width = np.linalg.norm((ls - rs)[valid][:, :2], axis=1)
    scale = np.median(width)
    if scale < 1e-4:
        return None
    out = (lm - center[:, None, :]) / scale
    out[~mask] = 0.0
    return out


def resample(seq, n=N_FRAMES):
    """Linear interpolation along time to exactly n frames. seq: (T, D)."""
    T = seq.shape[0]
    if T == n:
        return seq
    src = np.linspace(0, T - 1, n)
    idx0 = np.floor(src).astype(int)
    idx1 = np.minimum(idx0 + 1, T - 1)
    w = (src - idx0)[:, None]
    return (1 - w) * seq[idx0] + w * seq[idx1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if args.max_videos:
        videos = videos[: args.max_videos]

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for v in tqdm(videos):
            # keep class folder in the name so labels stay recoverable
            name = "__".join(v.relative_to(args.input_dir).with_suffix("").parts)
            dst = out_dir / f"{name}.npy"
            if dst.exists():
                continue
            lm, mask = extract_raw(v, holistic)
            if lm is None:
                print("skip (unreadable):", v)
                continue
            lm = normalize(lm, mask)
            if lm is None:
                print("skip (no shoulders detected):", v)
                continue
            seq = resample(lm.reshape(len(lm), -1), N_FRAMES)  # (64, 225)
            np.save(dst, seq.astype(np.float32))
            holistic.reset() if hasattr(holistic, "reset") else None


if __name__ == "__main__":
    main()