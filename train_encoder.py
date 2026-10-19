"""
Landmark encoder: (64, 225) landmarks -> 256-number embedding.

Training: the encoder learns to recognise the 101 sentences (classification).
Testing : we also check the RAG idea. We store the average embedding of every
          sentence (from train videos), then for each test video we find the
          closest stored sentence using cosine similarity.

Run:  python train_encoder.py --landmarks_dir landmarks1
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

N_FRAMES, N_FEATS, EMB = 64, 225, 256
IN_FEATS = 2 * (N_FEATS + 2 * 21 * 3)   # (225 body + 126 hand-shape) positions + same amount of motion


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_csv(path):
    with open(path) as fh:
        return [(r["file"], r["label"]) for r in csv.DictReader(fh)]


def augment(x):
    """Make a slightly different copy of a training video (x: (64, 225))."""
    # 1) random speed change: take 48-64 frames and stretch them back to 64
    length = np.random.randint(48, N_FRAMES + 1)
    start = np.random.randint(0, N_FRAMES - length + 1)
    seg = x[start:start + length]
    src = np.linspace(0, length - 1, N_FRAMES)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, length - 1)
    w = (src - i0)[:, None]
    present = (seg != 0)[np.where(w[:, 0] < 0.5, i0, i1)]   # keep "missing" = 0
    x = (1 - w) * seg[i0] + w * seg[i1]

    # 2) small tilt (camera angle), up to 10 degrees
    a = np.radians(np.random.uniform(-10, 10))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = x.reshape(N_FRAMES, 75, 3).copy()
    p[..., :2] = p[..., :2] @ rot.T
    x = p.reshape(N_FRAMES, N_FEATS)

    # 3) size change and tiny noise
    x = x * np.random.uniform(0.85, 1.15)
    x = x + np.random.normal(0, 0.02, x.shape)
    return x * present


def add_hand_features(x):
    """Add hand-shape numbers: each hand measured from its own wrist, scaled by hand size.
    x: (64, 225) -> (64, 351). A missing hand stays all zeros."""
    p = x.reshape(N_FRAMES, 75, 3)
    extra = []
    for a, b in ((33, 54), (54, 75)):          # left hand, right hand
        hand = p[:, a:b]                        # (64, 21, 3)
        present = (hand != 0).any(axis=(1, 2))  # (64,)
        wrist = hand[:, 0:1]
        size = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)   # wrist -> middle finger base
        size = np.clip(size, 1e-3, None)[:, None, None]
        local = (hand - wrist) / size
        local[~present] = 0.0
        extra.append(local.reshape(N_FRAMES, -1))
    return np.concatenate([x] + extra, axis=1)


def add_motion(x):
    """Add how much each number changed since the previous frame (x: (64, 351) -> (64, 702)).
    If a point was missing in either frame, its motion is 0."""
    present = x != 0
    prev_present = np.concatenate([present[:1], present[:-1]])
    prev = np.concatenate([x[:1], x[:-1]])
    motion = (x - prev) * (present & prev_present)
    return np.concatenate([x, motion], axis=1)


class LandmarkDataset(Dataset):
    def __init__(self, rows, folder, label2id, train=False):
        self.rows, self.folder, self.label2id, self.train = rows, Path(folder), label2id, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        f, label = self.rows[i]
        x = np.load(self.folder / f).astype(np.float32)  # (64, 225)
        if self.train:
            x = augment(x)
        x = add_motion(add_hand_features(x))
        return torch.from_numpy(x.astype(np.float32)), self.label2id[label]


class LandmarkEncoder(nn.Module):
    def __init__(self, n_classes, d=EMB, proj="linear"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(IN_FEATS, d, 5, padding=2), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=2), nn.BatchNorm1d(d), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, N_FRAMES, d))
        layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=2 * d, dropout=0.3,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        # projection layer: moves the encoder summary into the space where the database lives
        # 'linear' = one layer (worked best on our small data), 'mlp' = two layers
        self.proj = nn.Linear(d, d) if proj == "linear" else nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.classifier = nn.Linear(d, n_classes)

    def encode(self, x):                     # x: (B, 64, 702) -> (B, d)   "Landmark encoder" box
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, 64, d)
        h = self.transformer(h + self.pos)
        return h.mean(dim=1)

    def embed(self, x):                      # encoder + "Projection layer" box
        return self.proj(self.encode(x))

    def forward(self, x):
        return self.classifier(self.embed(x))


def build_model_from_ckpt(ckpt, device):
    """Rebuild the encoder from a saved checkpoint (model size and projection type are read from it)."""
    sd = ckpt["model"]
    d = sd["pos"].shape[-1]
    proj = "linear" if "proj.weight" in sd else "mlp"
    model = LandmarkEncoder(len(ckpt["label2id"]), d=d, proj=proj).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def get_embeddings(model, loader, device):
    model.eval()
    embs, labels = [], []
    for x, y in loader:
        embs.append(model.embed(x.to(device)).cpu())
        labels.append(y)
    return F.normalize(torch.cat(embs), dim=1), torch.cat(labels)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


class PKSampler:
    """Each batch = P sentences x K videos of each, so every video has same-sentence partners."""
    def __init__(self, labels, p=16, k=4, n_batches=15):
        self.by = {}
        for i, l in enumerate(labels):
            self.by.setdefault(l, []).append(i)
        self.classes, self.p, self.k, self.n = list(self.by), p, k, n_batches

    def __iter__(self):
        for _ in range(self.n):
            batch = []
            for c in np.random.choice(self.classes, self.p, replace=False):
                idx = self.by[c]
                batch += list(np.random.choice(idx, self.k, replace=len(idx) < self.k))
            yield [int(i) for i in batch]

    def __len__(self):
        return self.n


def supcon_loss(z, y, temp=0.1):
    """Videos of the same sentence are pulled together, other sentences pushed apart."""
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temp
    me = torch.eye(len(y), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(me, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~me
    cnt = pos.sum(1)
    ok = cnt > 0
    return (-(log_prob * pos).sum(1)[ok] / cnt[ok]).mean()


@torch.no_grad()
def retrieval_acc(model, db_loader, query_loader, n_classes, device):
    """Top-1 accuracy of cosine retrieval against one stored vector per sentence."""
    db_emb, db_lab = get_embeddings(model, db_loader, device)
    protos = F.normalize(torch.stack([db_emb[db_lab == c].mean(0) for c in range(n_classes)]), dim=1)
    q_emb, q_lab = get_embeddings(model, query_loader, device)
    return ((q_emb @ protos.T).argmax(1) == q_lab).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=256, help="model size (try 128 for smaller)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj", choices=["linear", "mlp"], default="linear",
                    help="projection layer: linear (1 layer) or mlp (2 layers)")
    ap.add_argument("--loss", choices=["ce", "supcon"], default="ce",
                    help="ce = pick the sentence (old), supcon = contrastive (new)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print("device:", device)

    train_rows, val_rows, test_rows = (read_csv(f"{n}.csv") for n in ("train", "val", "test"))
    labels = sorted({l for _, l in train_rows})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    mk = lambda rows, tr: DataLoader(
        LandmarkDataset(rows, args.landmarks_dir, label2id, train=tr),
        batch_size=args.batch, shuffle=tr)
    train_dl, val_dl, test_dl = mk(train_rows, True), mk(val_rows, False), mk(test_rows, False)
    train_eval_dl = mk(train_rows, False)  # no augmentation, used to build the database
    if args.loss == "supcon":
        train_dl = DataLoader(
            LandmarkDataset(train_rows, args.landmarks_dir, label2id, train=True),
            batch_sampler=PKSampler([label2id[l] for _, l in train_rows]))

    model = LandmarkEncoder(len(labels), d=args.dim, proj=args.proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if args.loss == "supcon":
                loss = supcon_loss(model.embed(x), y)
            else:
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()
        if args.loss == "supcon":
            val_acc = retrieval_acc(model, train_eval_dl, val_dl, len(labels), device)
        else:
            val_acc = accuracy(model, val_dl, device)
        print(f"epoch {ep:2d}  loss {total_loss / len(train_dl):.3f}  val acc {val_acc:.3f}")
        if val_acc >= best:
            best = val_acc
            torch.save({"model": model.state_dict(), "label2id": label2id}, "encoder.pt")

    # ---- final test with the best model, using cosine retrieval ----
    ckpt = torch.load("encoder.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    tr_emb, tr_lab = get_embeddings(model, train_eval_dl, device)
    protos = torch.stack([tr_emb[tr_lab == c].mean(0) for c in range(len(labels))])
    protos = F.normalize(protos, dim=1)               # the "database": one vector per sentence

    te_emb, te_lab = get_embeddings(model, test_dl, device)
    sims = te_emb @ protos.T                          # cosine similarity
    top5 = sims.topk(5, dim=1).indices
    top1_acc = (top5[:, 0] == te_lab).float().mean().item()
    top5_acc = (top5 == te_lab[:, None]).any(1).float().mean().item()
    print(f"\nBEST val acc: {best:.3f}")
    print(f"TEST top-1 retrieval accuracy: {top1_acc:.3f}")
    print(f"TEST top-5 retrieval accuracy: {top5_acc:.3f}")


if __name__ == "__main__":
    main()