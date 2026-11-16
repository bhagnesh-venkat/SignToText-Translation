"""
Train the LLM decoder (FLAN-T5 + LoRA).

Input : the retrieved candidates (sentence, gloss, similarity) for a video
Output: the final English sentence

Data  : decoder_data/train.jsonl  (out-of-fold candidates, made by make_decoder_data.py)
        decoder_data/val.jsonl    (used ONLY to pick the best epoch)
        decoder_data/test.jsonl   (evaluated once, at the end)

Setup (once, inside slt-env):
    pip install peft

Quick smoke test (about 1-2 minutes, checks nothing crashes):
    python train_decoder_lora.py --model google/flan-t5-small --epochs 1

Full run:
    python train_decoder_lora.py
If the Mac GPU (mps) gives an error, add:  --device cpu

How accuracy is measured: for every video, the decoder scores each of its candidate
sentences (log-probability of writing that sentence given the prompt) and the best one
is the final answer. A video counts as correct if that sentence is the true one.
"""
import argparse
import json
import random
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_prompt(item):
    lines = [f"{n}. {c['sentence']} | sign gloss: {c['gloss']} | match score {c['sim']:.2f}"
             for n, c in enumerate(item["candidates"], 1)]
    return ("Candidate sentences retrieved for a sign language video, best match first:\n"
            + "\n".join(lines) + "\nFinal sentence:")


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def choose(model, tok, item, device):
    """Return index of the candidate sentence the decoder finds most likely."""
    sents = [c["sentence"] for c in item["candidates"]]
    n = len(sents)
    enc = tok([build_prompt(item)] * n, return_tensors="pt", padding=True,
              truncation=True, max_length=256).to(device)
    lab = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=48).input_ids.to(device)
    masked = lab.masked_fill(lab == tok.pad_token_id, -100)
    logits = model(**enc, labels=masked).logits
    token_lp = torch.log_softmax(logits, dim=-1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
    return int((token_lp * (masked != -100)).sum(1).argmax())


def evaluate(model, tok, items, device, show=0):
    model.eval()
    n = len(items)
    dec = ret = ceiling = fixed = broken = 0
    shown = 0
    for it in items:
        pick = choose(model, tok, it, device)
        true = it["true"]
        d_ok = it["candidates"][pick]["sentence"] == true
        r_ok = it["candidates"][0]["sentence"] == true
        dec += d_ok
        ret += r_ok
        ceiling += it["true_in_candidates"]
        fixed += d_ok and not r_ok
        broken += r_ok and not d_ok
        if pick != 0 and shown < show:
            shown += 1
            print(f"  true: {true}\n  retrieval: {it['candidates'][0]['sentence']}"
                  f"\n  decoder:   {it['candidates'][pick]['sentence']}   -> {'correct' if d_ok else 'wrong'}\n")
    return {"decoder": 100 * dec / n, "retrieval_top1": 100 * ret / n, "ceiling": 100 * ceiling / n,
            "fixed": int(fixed), "broken": int(broken), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="decoder_data")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--all", action="store_true",
                    help="also train on examples whose true sentence is NOT among the candidates")
    ap.add_argument("--out", default="decoder_lora")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print("device:", device)

    train = load_jsonl(f"{args.data}/train.jsonl")
    val = load_jsonl(f"{args.data}/val.jsonl")
    test = load_jsonl(f"{args.data}/test.jsonl")
    if not args.all:
        train = [it for it in train if it["true_in_candidates"]]
    print(f"train examples: {len(train)}   val: {len(val)}   test: {len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=args.rank, lora_alpha=2 * args.rank,
                     lora_dropout=0.1, target_modules=["q", "v"])
    model = get_peft_model(base, cfg).to(device)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(train) + args.batch - 1) // args.batch
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / total))

    print("\nbefore training (val):", evaluate(model, tok, val, device))

    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        random.shuffle(train)
        running = 0.0
        for i in range(0, len(train), args.batch):
            batch = train[i:i + args.batch]
            enc = tok([build_prompt(it) for it in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            lab = tok([it["true"] for it in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=48).input_ids.to(device)
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        res = evaluate(model, tok, val, device)
        print(f"epoch {ep:2d}  loss {running / steps_per_epoch:.3f}  val decoder {res['decoder']:.1f}%  "
              f"(retrieval {res['retrieval_top1']:.1f}%)  fixed {res['fixed']} broke {res['broken']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if res["decoder"] > best:
            best = res["decoder"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}

    model.load_state_dict(best_state, strict=False)       # go back to the best epoch (chosen on val)
    model.save_pretrained(args.out)
    print(f"\nbest val accuracy {best:.1f}%   (adapter saved to {args.out}/)")

    print("\nFINAL TEST (evaluated once):")
    res = evaluate(model, tok, test, device, show=5)
    print("=" * 60)
    print(f"test videos                 : {res['n']}")
    print(f"retrieval top-1             : {res['retrieval_top1']:.1f}%")
    print(f"trained decoder (LoRA)      : {res['decoder']:.1f}%")
    print(f"ceiling (true in top-5)     : {res['ceiling']:.1f}%")
    print(f"decoder fixed {res['fixed']} retrieval mistakes and broke {res['broken']} correct answers")
    print("(zero-shot baseline was 23.2%)")


if __name__ == "__main__":
    main()