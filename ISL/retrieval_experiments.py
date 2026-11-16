"""
Quick retrieval experiments (no training, no changes to encoder.pt or database.pt).

Compares, on validation and test videos:
  - prototype retrieval (your current method: one averaged vector per sentence)
  - kNN retrieval       (a sentence is scored by its single closest training video)
  - ensemble            (average the similarity scores of several encoders)
and reports top-1 / top-5 / top-10.

Usage (inside the ISL folder):
    python retrieval_experiments.py --checkpoints encoder_final.pt
    python retrieval_experiments.py --checkpoints encoder_final.pt encoder_s1.pt encoder_s2.pt

Choose a method by looking at the VALIDATION rows. Look at test only for the one you picked.
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_encoder import LandmarkDataset, build_model_from_ckpt, get_embeddings, pick_device, read_csv


def embed_rows(model, rows, folder, label2id, device):
    dl = DataLoader(LandmarkDataset(rows, folder, label2id, train=False), batch_size=64)
    return get_embeddings(model, dl, device)          # normalized embeddings, labels (on cpu)


def similarity_scores(emb_tr, lab_tr, emb_q, n):
    """Return (prototype scores, kNN scores), each of shape (queries, n sentences)."""
    protos = F.normalize(torch.stack([emb_tr[lab_tr == c].mean(0) for c in range(n)]), dim=1)
    proto = emb_q @ protos.T
    full = emb_q @ emb_tr.T                            # (queries, train videos)
    knn = torch.stack([full[:, lab_tr == c].max(dim=1).values for c in range(n)], dim=1)
    return proto, knn


def top_k(sim, y, ks=(1, 5, 10)):
    order = sim.argsort(dim=1, descending=True)
    rank = (order == y[:, None]).float().argmax(dim=1) + 1
    return [100 * (rank <= k).float().mean().item() for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["encoder.pt"])
    ap.add_argument("--landmarks_dir", default="landmarks1")
    args = ap.parse_args()

    device = pick_device()
    train_rows, val_rows, test_rows = (read_csv(f"{n}.csv") for n in ("train", "val", "test"))
    results = {"val": {"proto": [], "knn": []}, "test": {"proto": [], "knn": []}}
    labels = {}

    print(f"{'checkpoint':<22}{'split':<6}{'method':<8}{'top-1':>7}{'top-5':>7}{'top-10':>8}")
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device)
        label2id = ckpt["label2id"]
        n = len(label2id)
        model = build_model_from_ckpt(ckpt, device)
        emb_tr, lab_tr = embed_rows(model, train_rows, args.landmarks_dir, label2id, device)
        for split, rows in (("val", val_rows), ("test", test_rows)):
            emb_q, lab_q = embed_rows(model, rows, args.landmarks_dir, label2id, device)
            proto, knn = similarity_scores(emb_tr, lab_tr, emb_q, n)
            labels[split] = lab_q
            results[split]["proto"].append(proto)
            results[split]["knn"].append(knn)
            for name, sim in (("proto", proto), ("knn", knn)):
                a, b, c = top_k(sim, lab_q)
                print(f"{path:<22}{split:<6}{name:<8}{a:>6.1f}%{b:>6.1f}%{c:>7.1f}%")

    if len(args.checkpoints) > 1:
        print("\nENSEMBLE (average of the similarity scores of all checkpoints)")
        for split in ("val", "test"):
            for name in ("proto", "knn"):
                sim = torch.stack(results[split][name]).mean(0)
                a, b, c = top_k(sim, labels[split])
                print(f"{'ensemble':<22}{split:<6}{name:<8}{a:>6.1f}%{b:>6.1f}%{c:>7.1f}%")

    print(f"\nRandom guessing over {len(label2id)} sentences: top-1 {100 / len(label2id):.1f}%, "
          f"top-5 {500 / len(label2id):.1f}%, top-10 {1000 / len(label2id):.1f}%")
    print("Reminder: 99 videos per split, so differences under about 5 points are noise.")


if __name__ == "__main__":
    main()