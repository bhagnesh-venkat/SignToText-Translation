"""
RAG retrieval: landmarks of one video -> embedding -> closest rows of the database.

Demo on test videos (the model has never seen these):
  python retrieve.py --landmarks_dir landmarks1 --n 8

One specific file:
  python retrieve.py --landmarks_dir landmarks1 --file "landmarks1/can i help you__help (3).npy"

The function retrieve() is what the LLM decoder step will call later.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from train_encoder import (add_hand_features, add_motion, build_model_from_ckpt,
                           read_csv, pick_device)


def load_everything(checkpoint="encoder.pt", database="database.pt"):
    device = pick_device()
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model_from_ckpt(ckpt, device)
    db = torch.load(database, map_location="cpu")
    return model, db, ckpt["label2id"], device


@torch.no_grad()
def retrieve(model, db, npy_path, device, k=5):
    """Return the k closest database rows: list of (similarity, row_index)."""
    x = np.load(npy_path).astype(np.float32)             # (64, 225)
    x = add_motion(add_hand_features(x))                 # (64, 702)
    emb = model.embed(torch.from_numpy(x.astype(np.float32))[None].to(device)).cpu()
    emb = F.normalize(emb, dim=1)
    sims = (emb @ db["embeddings"].T)[0]                 # cosine similarity to every row
    top = sims.topk(k)
    return [(s.item(), i.item()) for s, i in zip(top.values, top.indices)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", required=True)
    ap.add_argument("--file", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show", type=int, default=3, help="how many matches to print")
    args = ap.parse_args()

    model, db, label2id, device = load_everything()

    if args.file:
        jobs = [(args.file, None)]
    else:
        rows = read_csv("test.csv")
        random.seed(1)
        rows = random.sample(rows, args.n)
        jobs = [(f"{args.landmarks_dir}/{f}", label) for f, label in rows]

    hits1 = hits_k = 0
    for path, true_label in jobs:
        result = retrieve(model, db, path, device, k=args.k)
        print("=" * 70)
        print("video:", path.split("/")[-1])
        if true_label is not None:
            true_idx = label2id[true_label]
            print("true sentence:", db["sentences"][true_idx])
            hits1 += result[0][1] == true_idx
            hits_k += any(i == true_idx for _, i in result)
        for rank, (sim, i) in enumerate(result[: args.show], 1):
            mark = ""
            if true_label is not None:
                mark = "   <-- correct" if i == label2id[true_label] else ""
            print(f"  {rank}. {sim:.3f}  {db['sentences'][i]}   [{db['glosses'][i]}]{mark}")

    if len(jobs) > 1:
        print("=" * 70)
        print(f"in this demo: correct at rank 1 = {hits1}/{len(jobs)}, "
              f"correct in top {args.k} = {hits_k}/{len(jobs)}")


if __name__ == "__main__":
    main()