"""
Train a small landmark encoder (1D-conv + Transformer) to recognise isolated signs.

Setup (once, inside your slt-env):
    pip install torch matplotlib

Usage:
    python train_classifier.py --data processed --epochs 30

Input : processed/labels.csv + the (64, 225) .npy files from preprocess.py
Output: encoder.pt            (trained model + class names)
        training_curves.png   (loss and accuracy plot, handy for your slides)
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw plots to a file, no window needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------- data
def load_data(data_dir):
    """Read labels.csv and load every processed (64, 225) file."""
    data_dir = Path(data_dir)
    X, names = [], []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(np.load(data_dir / row["file"]))
            names.append(row["label"])
    classes = sorted(set(names))
    y = np.array([classes.index(n) for n in names])
    return np.stack(X).astype(np.float32), y, classes


def split_data(y, val_fraction=0.2, seed=0):
    """Per word: about 80% of videos for training, 20% for validation."""
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for c in sorted(set(y.tolist())):
        idx = [i for i, label in enumerate(y) if label == c]
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_fraction)) if len(idx) >= 2 else 0
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    return train_idx, val_idx


# --------------------------------------------------------------- model
class LandmarkEncoder(nn.Module):
    """(batch, 64, 225) landmarks -> (batch, 32, d_model) sequence of features.
    This is the 'landmark encoder' box of the architecture."""

    def __init__(self, in_dim=225, d_model=128, n_frames=64,
                 n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(          # looks at neighbouring frames, halves the length
            nn.Conv1d(in_dim, d_model, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, n_frames // 2, d_model))  # learned time positions
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):                   # x: (B, 64, 225)
        h = self.conv(x.transpose(1, 2))    # (B, d_model, 32)
        h = h.transpose(1, 2) + self.pos    # (B, 32, d_model)
        return self.transformer(h)          # (B, 32, d_model)


class SignClassifier(nn.Module):
    """Encoder + a small head that predicts which word was signed."""

    def __init__(self, n_classes, d_model=128):
        super().__init__()
        self.encoder = LandmarkEncoder(d_model=d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        features = self.encoder(x)          # (B, 32, d_model)
        return self.head(features.mean(dim=1))   # average over time, then classify


# ------------------------------------------------------------ training
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.02,
                        help="small random noise added while training (helps with little data)")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    X, y, classes = load_data(args.data)
    train_idx, val_idx = split_data(y)
    if not val_idx:
        raise SystemExit("Not enough videos per word to make a validation set.")

    print(f"{len(X)} videos, {len(classes)} words: {classes}")
    print(f"train: {len(train_idx)}  validation: {len(val_idx)}  "
          f"(random guessing would give about {100 / len(classes):.0f}%)")

    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    train_loader = DataLoader(TensorDataset(Xt[train_idx], yt[train_idx]),
                              batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xt[val_idx], yt[val_idx]),
                            batch_size=args.batch)

    model = SignClassifier(n_classes=len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    history = {"loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb in train_loader:
            if args.noise > 0:               # jitter only real (non-zero) values
                xb = xb + args.noise * torch.randn_like(xb) * (xb != 0)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)    # how wrong are the predictions?
            loss.backward()                  # work out how to change the weights
            optimizer.step()                 # change them a little
            running += loss.item()
            n_batches += 1

        train_acc = accuracy(model, DataLoader(TensorDataset(Xt[train_idx], yt[train_idx]),
                                               batch_size=args.batch))
        val_acc = accuracy(model, val_loader)
        history["loss"].append(running / n_batches)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:3d}  loss {running / n_batches:.3f}  "
              f"train acc {train_acc:.2f}  val acc {val_acc:.2f}")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "classes": classes}, "encoder.pt")

    print(f"\nBest validation accuracy: {best_val:.2f}  (saved to encoder.pt)")

    # plot for the slides
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(history["loss"])
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")


if __name__ == "__main__":
    main()