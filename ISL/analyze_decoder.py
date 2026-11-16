"""
Per-video comparison on the TEST set: retrieval (rank 1) vs the trained LoRA decoder.

Put this file in the ISL folder (next to train_decoder_lora.py, decoder_data/ and decoder_lora/).
Run:   python analyze_decoder.py
If the Mac GPU gives an error:   python analyze_decoder.py --device cpu

Outputs: printed summary, full top-5 tables for FIXED and BROKEN videos, decoder_analysis.csv
"""
import argparse
import csv
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# reuse the exact functions used for training/evaluation
from train_decoder_lora import choose, load_jsonl, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="decoder_data")
ap.add_argument("--adapter", default="decoder_lora")
ap.add_argument("--model", default="google/flan-t5-base")
ap.add_argument("--device", default="auto")
args = ap.parse_args()

device = pick_device(args.device)
print("device:", device)

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForSeq2SeqLM.from_pretrained(args.model)
model = PeftModel.from_pretrained(base, args.adapter).to(device)
model.eval()

examples = load_jsonl(f"{args.data}/test.jsonl")
rows, details = [], {}

for ex in examples:
    cands = ex["candidates"]
    sents = [c["sentence"] for c in cands]
    sims = [c["sim"] for c in cands]
    true = ex["true"]

    pick = choose(model, tok, ex, device)          # index of decoder's answer
    ret_pred, dec_pred = sents[0], sents[pick]
    ret_ok, dec_ok = ret_pred == true, dec_pred == true
    true_rank = sents.index(true) + 1 if true in sents else None

    if ret_ok and dec_ok:
        outcome = "kept_correct"
    elif dec_ok:
        outcome = "FIXED"
    elif ret_ok:
        outcome = "BROKEN"
    else:
        outcome = "both_wrong_in_top5" if true_rank else "both_wrong_not_in_top5"

    rows.append({
        "video": ex["video"], "true": true,
        "retrieval_pred": ret_pred, "decoder_pred": dec_pred,
        "outcome": outcome, "true_rank": true_rank,
        "sim_gap_1_2": round(sims[0] - sims[1], 3),
        "words_retrieval": len(ret_pred.split()), "words_decoder": len(dec_pred.split()),
        "decoder_changed": pick != 0,
    })
    details[ex["video"]] = (ex, pick)

with open("decoder_analysis.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\n", dict(Counter(r["outcome"] for r in rows)))
acc = lambda key: 100 * sum(r[key] == r["true"] for r in rows) / len(rows)
print(f"retrieval top-1 {acc('retrieval_pred'):.1f}%   decoder {acc('decoder_pred'):.1f}%   "
      f"(report: 43.4% and 41.4%)")

for label in ("FIXED", "BROKEN"):
    print("\n" + "=" * 70 + f"\n{label}\n" + "=" * 70)
    for r in rows:
        if r["outcome"] != label:
            continue
        ex, pick = details[r["video"]]
        print(f"\nvideo: {r['video']}\ntrue : {r['true']}   (gap 1-2 = {r['sim_gap_1_2']})")
        for n, c in enumerate(ex["candidates"]):
            tag = ""
            if c["sentence"] == r["true"]:
                tag += " [TRUE]"
            if n == pick:
                tag += " <- decoder"
            print(f"  {n + 1}. {c['sim']:.3f}  {c['sentence']}{tag}")

changed = [r for r in rows if r["decoder_changed"]]
shorter = sum(r["words_decoder"] < r["words_retrieval"] for r in changed)
print(f"\nDecoder changed {len(changed)} of {len(rows)} answers; "
      f"{shorter} moved to a shorter sentence.")