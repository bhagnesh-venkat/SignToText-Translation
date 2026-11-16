"""
Build the training data for the LLM decoder.

Problem: if the decoder is trained on candidates retrieved for videos the encoder
already trained on, retrieval looks too easy and the decoder never sees real mistakes.
Fix: K-fold. For each fold, train a fresh encoder on the other folds, then retrieve
candidates for the held-out videos (which that encoder has never seen).

Outputs (folder decoder_data/):
  train.jsonl  candidates for the train videos, from the K held-out folds
  val.jsonl    candidates for val videos,  from your final encoder.pt
  test.jsonl   candidates for test videos, from your final encoder.pt
Each line: {"video", "true", "true_in_candidates", "candidates": [{"sentence","gloss","sim"}, ...]}

This script NEVER writes encoder.pt or database.pt.

Run inside the ISL folder (use the same settings as your final encoder):
    python make_decoder_data.py --landmarks_dir landmarks1 --dim 128 --epochs 40
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_encoder import (LandmarkDataset, LandmarkEncoder, PKSampler, build_model_from_ckpt,
                           get_embeddings, pick_device, read_csv, supcon_loss)


def make_loader(rows, args, label2id):
    return DataLoader(LandmarkDataset(rows, args.landmarks_dir, label2id, train=False), batch_size=64)


def train_fold_encoder(rows, label2id, args, device):
    """Contrastive training (same recipe as train_encoder.py --loss supcon), last epoch is kept."""
    labels = [label2id[l] for _, l in rows]
    dl = DataLoader(LandmarkDataset(rows, args.landmarks_dir, label2id, train=True),
                    batch_sampler=PKSampler(labels))
    model = LandmarkEncoder(len(label2id), d=args.dim, proj="linear").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl))
    for _ in range(args.epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = supcon_loss(model.embed(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def candidates_for(model, db_rows, query_rows, label2id, db, args, device):
    """Database = mean train embedding per sentence (from db_rows). Returns top-k candidates per query."""
    tr_emb, tr_lab = get_embeddings(model, make_loader(db_rows, args, label2id), device)
    present = sorted(set(tr_lab.tolist()))                      # sentences that have a database vector
    protos = F.normalize(torch.stack([tr_emb[tr_lab == c].mean(0) for c in present]), dim=1)
    q_emb, _ = get_embeddings(model, make_loader(query_rows, args, label2id), device)
    top = (q_emb @ protos.T).topk(min(args.k, len(present)), dim=1)
    out = []
    for r, (f, label) in enumerate(query_rows):
        true_idx = label2id[label]
        idxs = [present[j] for j in top.indices[r].tolist()]
        out.append({
            "video": f,
            "true": db["sentences"][true_idx],
            "true_in_candidates": true_idx in idxs,
            "candidates": [{"sentence": db["sentences"][i], "gloss": db["glosses"][i], "sim": round(s, 4)}
                           for s, i in zip(top.values[r].tolist(), idxs)],
        })
    return out


def top1(items):
    return 100 * np.mean([it["candidates"][0]["sentence"] == it["true"] for it in items])


def topk(items):
    return 100 * np.mean([it["true_in_candidates"] for it in items])


def save(items, path):
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", default="landmarks1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="decoder_data")
    args = ap.parse_args()

    device = pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    db = torch.load("database.pt", map_location="cpu")             # clean sentence + gloss text
    ckpt = torch.load("encoder.pt", map_location=device)
    label2id = ckpt["label2id"]

    train_rows, val_rows, test_rows = (read_csv(f"{n}.csv") for n in ("train", "val", "test"))

    # ---- assign every train video to a fold (spread each sentence's videos over folds)
    rng = random.Random(args.seed)
    by_label = {}
    for i, (_, l) in enumerate(train_rows):
        by_label.setdefault(l, []).append(i)
    fold_of = {}
    for idxs in by_label.values():
        rng.shuffle(idxs)
        offset = rng.randrange(args.folds)
        for pos, i in enumerate(idxs):
            fold_of[i] = (pos + offset) % args.folds

    # ---- out-of-fold candidates for the train videos
    oof = []
    for k in range(args.folds):
        t0 = time.time()
        torch.manual_seed(args.seed + k)
        np.random.seed(args.seed + k)
        fit = [r for i, r in enumerate(train_rows) if fold_of[i] != k]
        held = [r for i, r in enumerate(train_rows) if fold_of[i] == k]
        model = train_fold_encoder(fit, label2id, args, device)
        items = candidates_for(model, fit, held, label2id, db, args, device)
        oof += items
        print(f"fold {k + 1}/{args.folds}: trained on {len(fit)}, held out {len(held)}  "
              f"top-1 {top1(items):.1f}%  top-{args.k} {topk(items):.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    save(oof, out_dir / "train.jsonl")

    # ---- val / test candidates from the final encoder (database built from all train videos)
    final = build_model_from_ckpt(ckpt, device)
    for name, rows in (("val", val_rows), ("test", test_rows)):
        items = candidates_for(final, train_rows, rows, label2id, db, args, device)
        save(items, out_dir / f"{name}.jsonl")
        print(f"{name}: {len(items)} videos  top-1 {top1(items):.1f}%  top-{args.k} {topk(items):.1f}%")

    print(f"\ntrain (out-of-fold): {len(oof)} videos  top-1 {top1(oof):.1f}%  top-{args.k} {topk(oof):.1f}%")
    print(f"saved to {out_dir}/  (train.jsonl, val.jsonl, test.jsonl)")


if __name__ == "__main__":
    main()