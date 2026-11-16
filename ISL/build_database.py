"""
Build the gloss-text database for RAG.

For every sentence we store:
  - an embedding (average of the encoder's embeddings of its TRAIN videos)
  - the gloss   (from 'ISL Corpus sign glosses.csv')
  - the English sentence

Run:
  python build_database.py --landmarks_dir landmarks1 \
      --gloss_csv "isl_data/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv"

Needs: encoder.pt (made by train_encoder.py) and train.csv.
Makes: database.pt (used by the retrieval step) and database.csv (readable).
"""
import argparse
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_encoder import LandmarkDataset, build_model_from_ckpt, read_csv, pick_device


def norm_text(t):
    """Lower-case and remove ALL spaces, so 'collegeschool' matches 'college school'."""
    return "".join(t.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", required=True)
    ap.add_argument("--gloss_csv", required=True)
    ap.add_argument("--checkpoint", default="encoder.pt")
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    label2id = ckpt["label2id"]
    id2label = {i: l for l, i in label2id.items()}
    dim = ckpt["model"]["pos"].shape[-1]      # model size is stored in the weights
    model = build_model_from_ckpt(ckpt, device)
    print(f"encoder loaded (dim {dim}), {len(label2id)} sentences")

    # glosses, matched to sentences by lower-case text
    glosses = {}
    with open(args.gloss_csv, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            glosses[norm_text(r["Sentence"])] = (r["Sentence"].strip(), r["SIGN GLOSSES"].strip())

    # embeddings of the TRAIN videos only (no augmentation)
    rows = read_csv("train.csv")
    dl = DataLoader(LandmarkDataset(rows, args.landmarks_dir, label2id, train=False),
                    batch_size=64, shuffle=False)
    embs, labs = [], []
    with torch.no_grad():
        for x, y in dl:
            embs.append(model.embed(x.to(device)).cpu())
            labs.append(y)
    embs = F.normalize(torch.cat(embs), dim=1)
    labs = torch.cat(labs)

    n = len(label2id)
    protos = F.normalize(torch.stack([embs[labs == c].mean(0) for c in range(n)]), dim=1)

    sentences, gloss_list, missing = [], [], []
    for i in range(n):
        folder_name = id2label[i]
        hit = glosses.get(norm_text(folder_name))
        if hit is None:
            missing.append(folder_name)
            sentences.append(folder_name)
            gloss_list.append("")
        else:
            sentences.append(hit[0])      # clean sentence text from the CSV
            gloss_list.append(hit[1])

    torch.save({"embeddings": protos, "sentences": sentences, "glosses": gloss_list},
               "database.pt")
    with open("database.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence", "gloss"])
        w.writerows(zip(sentences, gloss_list))

    print(f"database saved: {n} rows (database.pt, database.csv)")
    print(f"rows with a gloss: {n - len(missing)}")
    if missing:
        print("sentences with NO gloss found:")
        for s in missing:
            print("  -", s)


if __name__ == "__main__":
    main()