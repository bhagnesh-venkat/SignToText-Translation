"""
Live demo: raw ISL video -> MediaPipe landmarks -> preprocessing -> encoder -> RAG retrieval.

Put this file in the ISL folder (next to retrieve.py) and run, inside slt-env:

    python translate.py "path/to/video.mp4"
    python translate.py "path/to/video.mp4" --true "that is so kind of you"

--true is optional: give the real sentence to mark the correct row in the table.

Uses your existing code:
  extract_landmarks1.py  -> extract_raw, normalize, resample   (preprocessing)
  train_encoder.py       -> add_hand_features, add_motion       (model input features)
  retrieve.py            -> load_everything                     (encoder.pt + database.pt)
If you renamed extract_landmarks1.py, change the import on the line marked (*) below.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from extract_landmarks1 import extract_raw, normalize, resample, mp_holistic   # (*)
from train_encoder import add_hand_features, add_motion
from retrieve import load_everything

N_FRAMES = 64


def norm_text(t):
    return "".join(str(t).lower().split())


def video_to_landmarks(video_path):
    """Stage 1+2: raw video -> (T,75,3) landmarks -> normalized -> (64, 225)."""
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        lm, mask = extract_raw(video_path, holistic)
    if lm is None:
        raise SystemExit(f"Could not read any frames from {video_path}")
    n_frames_raw = len(lm)
    lm = normalize(lm, mask)
    if lm is None:
        raise SystemExit("No shoulders detected: the signer must be visible from the waist up.")
    seq = resample(lm.reshape(len(lm), -1), N_FRAMES).astype(np.float32)
    return seq, n_frames_raw


@torch.no_grad()
def embed(model, seq, device):
    """Stage 3: (64,225) -> (64,702) model input -> 256-number embedding."""
    x = add_motion(add_hand_features(seq)).astype(np.float32)
    z = model.embed(torch.from_numpy(x)[None].to(device)).cpu()
    return x, F.normalize(z, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--true", default=None, help="the real sentence, to mark the correct row")
    args = ap.parse_args()

    model, db, label2id, device = load_everything()
    print(f"\nVideo: {args.video}")
    print("-" * 70)

    t0 = time.time()
    seq, n_raw = video_to_landmarks(args.video)
    zero_pct = 100 * float((seq == 0).mean())
    print(f"1. MediaPipe landmarks   : {n_raw} frames x 225 numbers per frame")
    print(f"2. Normalize + resample  : ({n_raw}, 225) -> {seq.shape}   "
          f"[{zero_pct:.0f}% of values missing = hand/body not detected]")

    x, z = embed(model, seq, device)
    print(f"3. Hand-shape + motion   : model input {x.shape}")
    print(f"   Encoder + projection  : embedding {tuple(z.shape[1:])} (length 1)")

    sims = (z @ db["embeddings"].T)[0]
    top = sims.topk(args.k)
    print(f"4. RAG retrieval         : cosine similarity against {len(sims)} database rows")
    print(f"   (total time {time.time() - t0:.1f} s)")
    print("-" * 70)
    print(f"{'Rank':<5}{'Sim':<8}{'Sentence':<42}Gloss")
    for rank, (s, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        mark = ""
        if args.true and norm_text(db["sentences"][i]) == norm_text(args.true):
            mark = "   <-- correct"
        print(f"{rank:<5}{s:<8.3f}{str(db['sentences'][i]):<42}{db['glosses'][i]}{mark}")
    print("-" * 70)
    print(f"Current output (top-1): {db['sentences'][top.indices[0].item()]}")
    print("Next stage: the LLM decoder will choose/write the final sentence from these candidates + glosses.")


if __name__ == "__main__":
    main()