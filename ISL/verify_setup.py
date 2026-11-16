"""
Check that the Sign2Text (RAG) project is in the right place and consistent.

Put this file in the same folder as retrieve.py and run:
    cd ISL
    python verify_setup.py --landmarks_dir landmarks1

Every line starts with OK, WARN or FAIL. Fix FAIL lines first.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

results = {"OK": 0, "WARN": 0, "FAIL": 0}


def report(level, msg):
    results[level] += 1
    print(f"[{level:4}] {msg}")


def read_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [(r["file"], r["label"]) for r in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", default="landmarks1")
    ap.add_argument("--expected_videos", type=int, default=687)
    args = ap.parse_args()
    lm_dir = Path(args.landmarks_dir)

    # ---------------------------------------------------- 1. files exist
    print("\n1) Required files")
    required = ["train_encoder.py", "retrieve.py", "build_database.py", "split_data.py",
                "encoder.pt", "database.pt", "database.csv",
                "train.csv", "val.csv", "test.csv"]
    for name in required:
        if Path(name).exists():
            report("OK", name)
        else:
            report("FAIL", f"{name} not found. Are you inside the ISL folder? ({Path.cwd()})")
    if not lm_dir.is_dir():
        report("FAIL", f"folder {lm_dir}/ not found")
        return finish()

    # ----------------------------------------------- 2. landmark files
    print("\n2) Landmark files")
    files = sorted(lm_dir.glob("*.npy"))
    if len(files) == args.expected_videos:
        report("OK", f"{len(files)} .npy files in {lm_dir}/")
    else:
        report("FAIL", f"{len(files)} .npy files in {lm_dir}/ (expected {args.expected_videos})")

    bad_shape, bad_value, zero_fracs = [], [], []
    for f in files:
        x = np.load(f)
        if x.shape != (64, 225):
            bad_shape.append(f.name)
            continue
        if not np.isfinite(x).all():
            bad_value.append(f.name)
        zero_fracs.append(float((x == 0).mean()))
    report("OK" if not bad_shape else "FAIL", f"files with wrong shape (need (64, 225)): {len(bad_shape)}")
    report("OK" if not bad_value else "FAIL", f"files with NaN/inf values: {len(bad_value)}")
    if zero_fracs:
        report("OK", f"missing values (zeros): average {100 * np.mean(zero_fracs):.0f}%, "
                     f"worst {100 * np.max(zero_fracs):.0f}%   (slide 8 says 22% / 56%)")

    # ------------------------------------------------------ 3. splits
    print("\n3) Train / val / test split")
    splits = {n: read_rows(f"{n}.csv") for n in ("train", "val", "test") if Path(f"{n}.csv").exists()}
    if len(splits) == 3:
        sizes = {n: len(r) for n, r in splits.items()}
        total = sum(sizes.values())
        report("OK", f"sizes {sizes}   (slide 7 says 489 / 99 / 99)")
        report("OK" if total == args.expected_videos else "FAIL", f"total rows = {total}")
        names = {n: {f for f, _ in r} for n, r in splits.items()}
        overlap = (names["train"] & names["test"]) | (names["train"] & names["val"]) | (names["val"] & names["test"])
        report("OK" if not overlap else "FAIL", f"files that appear in more than one split: {len(overlap)}")
        missing = [f for r in splits.values() for f, _ in r if not (lm_dir / f).exists()]
        report("OK" if not missing else "FAIL", f"CSV rows pointing to a missing file: {len(missing)}")
        labels = {n: {l for _, l in r} for n, r in splits.items()}
        report("OK" if labels["train"] == labels["test"] else "WARN",
               f"sentences: train {len(labels['train'])}, val {len(labels['val'])}, test {len(labels['test'])}")

    # ------------------------------------- 4. encoder + database files
    print("\n4) Encoder and database")
    try:
        import torch
        from retrieve import load_everything, retrieve
        model, db, label2id, device = load_everything()
    except Exception as e:  # noqa: BLE001
        report("FAIL", f"could not load encoder.pt / database.pt: {e}")
        return finish()

    report("OK", f"encoder loaded on {device}, {len(label2id)} sentences in label2id")
    emb = db["embeddings"]
    report("OK", f"database keys: {sorted(db.keys())}")
    report("OK" if emb.shape[0] == len(label2id) else "FAIL",
           f"database rows = {emb.shape[0]}, encoder sentences = {len(label2id)}")
    for key in ("sentences", "glosses"):
        if key not in db:
            report("FAIL", f"database has no '{key}'")
        elif len(db[key]) != emb.shape[0]:
            report("FAIL", f"len(db['{key}']) = {len(db[key])} but embeddings = {emb.shape[0]}")
    norms = emb.norm(dim=1)
    report("OK" if torch.allclose(norms, torch.ones_like(norms), atol=1e-3) else "WARN",
           "database vectors have length 1 (cosine = dot product)")

    # row i of the database must be sentence i of label2id, because retrieve.py uses label2id as the row index
    id2label = {i: l for l, i in label2id.items()}
    norm = lambda t: "".join(str(t).lower().split())
    mism = [i for i in range(len(id2label))
        if norm(db["sentences"][i]) != norm(id2label[i])]
    report("OK" if not mism else "FAIL",
           f"database row order matches encoder label order (mismatches: {len(mism)})")

    # ------------------- 5. end to end: is the DB built from THIS encoder?
    print("\n5) End-to-end test on test.csv (this proves encoder.pt and database.pt belong together)")
    rows = splits.get("test", [])
    hit1 = hit5 = 0
    for f, label in rows:
        res = retrieve(model, db, lm_dir / f, device, k=5)
        true_idx = label2id[label]
        hit1 += res[0][1] == true_idx
        hit5 += any(i == true_idx for _, i in res)
    n = max(len(rows), 1)
    top1, top5 = 100 * hit1 / n, 100 * hit5 / n
    print(f"       top-1 = {top1:.1f}%   top-5 = {top5:.1f}%   on {len(rows)} test videos")
    print("       (slide 10 reports 44.4% / 75.4% as an average of 3 runs; one run can differ by +-5)")
    if top1 < 10:
        report("FAIL", "accuracy near random: encoder.pt and database.pt probably do not match. Run build_database.py again.")
    elif top1 < 30:
        report("WARN", "lower than your slide. encoder.pt may be from a different run than the one you reported.")
    else:
        report("OK", "encoder and database agree, retrieval works")

    finish()


def finish():
    print(f"\nSummary: {results['OK']} OK, {results['WARN']} WARN, {results['FAIL']} FAIL")
    if results["FAIL"] == 0:
        print("Everything needed for the demo is in place.")


if __name__ == "__main__":
    main()