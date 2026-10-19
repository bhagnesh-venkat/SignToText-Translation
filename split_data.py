"""
Split the landmark .npy files into train / val / test.

The split is done per sentence, so every sentence appears in all three groups
(the RAG database needs to know every sentence).

File names look like:  <English sentence>__<original video name>.npy
The part before '__' is the label.

Run:  python split_data.py --landmarks_dir landmarks1
Output: train.csv, val.csv, test.csv  (columns: file,label)
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    by_label = defaultdict(list)
    for p in sorted(Path(args.landmarks_dir).glob("*.npy")):
        label = p.stem.split("__")[0].strip()
        by_label[label].append(p.name)

    splits = {"train": [], "val": [], "test": []}
    for label, files in by_label.items():
        random.shuffle(files)
        n = len(files)
        if n >= 3:
            n_test = max(1, round(n * 0.15))
            n_val = max(1, round(n * 0.15))
        else:
            # too few videos: keep them all for training
            n_test = n_val = 0
        test = files[:n_test]
        val = files[n_test:n_test + n_val]
        train = files[n_test + n_val:]
        splits["train"] += [(f, label) for f in train]
        splits["val"] += [(f, label) for f in val]
        splits["test"] += [(f, label) for f in test]

    for name, rows in splits.items():
        with open(f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "label"])
            w.writerows(rows)
        print(f"{name}: {len(rows)} files")

    print("sentences:", len(by_label))
    few = [l for l, f in by_label.items() if len(f) < 3]
    if few:
        print("sentences with fewer than 3 videos (all in train):", len(few))


if __name__ == "__main__":
    main()