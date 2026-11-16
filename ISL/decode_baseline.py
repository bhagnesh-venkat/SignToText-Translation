"""
Zero-shot decoder baseline (no training).

For every test video: retrieve the top-k candidate sentences (with glosses and
similarity scores), show them to FLAN-T5, and let it pick one. Then compare the
accuracy of its pick with plain retrieval top-1.

Setup (once, inside slt-env; needs internet for the first model download):
    pip install transformers sentencepiece

Run inside the ISL folder:
    python decode_baseline.py --landmarks_dir landmarks1
    python decode_baseline.py --landmarks_dir landmarks1 --model google/flan-t5-large   # optional, slower
"""
import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retrieve import load_everything, retrieve
from train_encoder import read_csv


def build_prompt(cands):
    lines = [f"{n}. {s} (sign gloss: {g}; match score {sim:.2f})"
             for n, (s, g, sim) in enumerate(cands, 1)]
    return ("A sign language video was compared with a database of sentences. "
            "These are the closest candidates, best match first:\n"
            + "\n".join(lines)
            + "\nWhich candidate sentence was most likely signed? Answer with the number.")


@torch.no_grad()
def choose(tok, lm, prompt, n):
    """Return the index (0..n-1) of the candidate number FLAN-T5 finds most likely."""
    enc = tok(prompt, return_tensors="pt")
    start = torch.tensor([[lm.config.decoder_start_token_id]])
    logits = lm(**enc, decoder_input_ids=start).logits[0, -1]
    num_ids = [tok(str(i)).input_ids[0] for i in range(1, n + 1)]
    return int(torch.stack([logits[i] for i in num_ids]).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", default="landmarks1")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--examples", type=int, default=5, help="how many changed answers to print")
    args = ap.parse_args()

    enc_model, db, label2id, device = load_everything()
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()   # runs on CPU, fine for 99 videos

    rows = read_csv("test.csv")
    n = ret1 = dec = topk = fixed = broken = 0
    shown = 0
    for f, label in rows:
        true_idx = label2id[label]
        res = retrieve(enc_model, db, f"{args.landmarks_dir}/{f}", device, k=args.k)
        cands = [(db["sentences"][i], db["glosses"][i], sim) for sim, i in res]
        pick = choose(tok, lm, build_prompt(cands), len(cands))
        pick_idx = res[pick][1]

        n += 1
        r_ok = res[0][1] == true_idx
        d_ok = pick_idx == true_idx
        ret1 += r_ok
        dec += d_ok
        topk += any(i == true_idx for _, i in res)
        if d_ok and not r_ok:
            fixed += 1
        if r_ok and not d_ok:
            broken += 1
        if pick != 0 and shown < args.examples:
            shown += 1
            print(f"\nvideo: {f}\n  true:      {db['sentences'][true_idx]}"
                  f"\n  retrieval: {cands[0][0]}\n  decoder:   {cands[pick][0]}"
                  f"   -> {'correct' if d_ok else 'wrong'}")

    print("\n" + "=" * 60)
    print(f"model: {args.model}   test videos: {n}")
    print(f"retrieval top-1 accuracy : {100 * ret1 / n:.1f}%")
    print(f"decoder pick accuracy    : {100 * dec / n:.1f}%")
    print(f"upper bound (true sentence in top-{args.k}): {100 * topk / n:.1f}%")
    print(f"decoder fixed {fixed} retrieval mistakes and broke {broken} correct answers")


if __name__ == "__main__":
    main()