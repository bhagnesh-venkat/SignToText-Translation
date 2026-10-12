"""
Extract pose + hand landmarks from sign language videos using MediaPipe Holistic.
Runs fine on a CPU laptop.

Setup (Python 3.10 or 3.11 recommended):
    pip install mediapipe==0.10.14 opencv-python numpy

Usage:
    python extract_landmarks.py --videos videos/ --out landmarks/ --preview

Output:
    landmarks/<video_name>.npy   -> array of shape (num_frames, 225)
    landmarks/<video_name>_preview.mp4 (only with --preview)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def landmarks_to_array(results):
    """One frame -> vector of 225 numbers:
    pose 33x3 + left hand 21x3 + right hand 21x3 (x, y, z each).
    Parts that are not detected (e.g. hand out of frame) become zeros."""

    def to_arr(lms, n_points):
        if lms is None:
            return np.zeros(n_points * 3, dtype=np.float32)
        return np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32).flatten()

    pose = to_arr(results.pose_landmarks, 33)
    left = to_arr(results.left_hand_landmarks, 21)
    right = to_arr(results.right_hand_landmarks, 21)
    return np.concatenate([pose, left, right])


def process_video(video_path, preview_path=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    writer = None

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(landmarks_to_array(results))

            if preview_path is not None:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    return np.stack(frames) if frames else np.zeros((0, 225), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True,
                        help="Root folder containing videos (one sub-folder per sign)")
    parser.add_argument("--out", required=True,
                        help="Folder to save .npy files (same sub-folder layout)")
    parser.add_argument("--per-class", type=int, default=None,
                        help="Max videos to use from each sub-folder, e.g. 20")
    parser.add_argument("--preview", action="store_true",
                        help="Also save a video with landmarks drawn")
    args = parser.parse_args()

    root = Path(args.videos)
    out_dir = Path(args.out)

    # Keep only real videos (skip hidden files and the __MACOSX junk from Mac unzipping)
    videos = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
        and not p.name.startswith(".")
        and "__MACOSX" not in p.parts
    )

    if args.per_class is not None:
        by_folder = {}
        for p in videos:
            by_folder.setdefault(p.parent, []).append(p)
        videos = [v for group in by_folder.values() for v in group[: args.per_class]]

    print(f"Found {len(videos)} videos")

    for i, vp in enumerate(videos, 1):
        rel = vp.relative_to(root)
        target = out_dir / rel.with_suffix(".npy")     # same folder layout as the input

        if target.exists():                            # lets you stop and resume later
            print(f"[{i}/{len(videos)}] already done: {rel}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        preview = target.with_name(target.stem + "_preview.mp4") if args.preview else None
        arr = process_video(vp, preview)
        np.save(target, arr)
        print(f"[{i}/{len(videos)}] {rel}: {arr.shape}")


if __name__ == "__main__":
    main()