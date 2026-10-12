"""
Clean landmark files so a model can learn from them.

Step 1 - Normalize: positions become relative to the shoulders
         (centered on the shoulder midpoint, scaled by shoulder width).
Step 2 - Fixed length: pick N frames spread evenly across the video.

Usage:
    python preprocess.py --inp landmarks --out processed --frames 64

Input : landmarks/<name>.npy      shape (any_frames, 225)
Output: processed/<name>.npy      shape (64, 225)
"""
import argparse
from pathlib import Path

import numpy as np

N_POINTS = 75          # 33 body + 21 left hand + 21 right hand
LEFT_SHOULDER = 11     # index inside the body points
RIGHT_SHOULDER = 12


def normalize(seq):
    """seq: (T, 225) -> (T, 225), relative to shoulders."""
    seq = seq.reshape(len(seq), N_POINTS, 3).copy()

    # Points that were not detected are exactly 0; remember them
    missing = np.all(seq == 0, axis=2)                    # (T, 75)

    left, right = seq[:, LEFT_SHOULDER, :], seq[:, RIGHT_SHOULDER, :]
    center = (left + right) / 2                           # (T, 3), per frame

    # One stable scale for the whole video: median shoulder width
    width = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
    valid = width > 1e-6
    scale = np.median(width[valid]) if valid.any() else 1.0

    seq = (seq - center[:, None, :]) / scale
    seq[missing] = 0                                      # keep "not detected" as 0
    return seq.reshape(len(seq), N_POINTS * 3)


def resample(seq, n_frames):
    """Pick n_frames evenly spaced frames (repeats frames if the video is short)."""
    idx = np.linspace(0, len(seq) - 1, n_frames).round().astype(int)
    return seq[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", required=True, help="Folder with landmark .npy files")
    parser.add_argument("--out", required=True, help="Folder to save processed files")
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.inp)
    files = sorted(inp.rglob("*.npy"))
    print(f"Found {len(files)} files")

    rows = []
    for f in files:
        seq = np.load(f)
        rel = f.relative_to(inp)
        if len(seq) == 0:
            print(f"Skipping {rel}: no frames")
            continue
        out = resample(normalize(seq), args.frames)
        target = out_dir / rel                      # same folder layout as the input
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, out.astype(np.float32))
        rows.append((rel.as_posix(), f.parent.name))   # label = name of the sign's folder
        print(f"{rel}: {seq.shape} -> {out.shape}")

    # labels.csv pairs every processed file with its sign name
    with open(out_dir / "labels.csv", "w", encoding="utf-8") as fh:
        fh.write("file,label\n")
        for file, label in rows:
            fh.write(f'"{file}","{label}"\n')
    print(f"Saved {len(rows)} rows to {out_dir / 'labels.csv'}")


if __name__ == "__main__":
    main()