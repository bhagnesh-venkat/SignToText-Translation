"""
Whole pipeline on ONE new video:
  raw video -> MediaPipe landmarks -> normalize + 64 frames -> hand-shape + motion
  -> landmark encoder -> cosine retrieval (top-k) -> FLAN-T5 + LoRA decoder picks the final sentence

Put this file in the ISL folder (next to translate.py and train_decoder_lora.py).

    python translate_full.py "path/to/new_video.mp4"
    python translate_full.py "path/to/new_video.mp4" --true "the real sentence"   # optional, marks correct rows
    python translate_full.py "path/to/new_video.mp4" --no_decoder                 # retrieval only (like translate.py)

Needs: encoder.pt, database.pt (a matching pair) and the folder decoder_lora/ (from train_decoder_lora.py).
Nothing is trained or overwritten.
"""
import argparse
import time

import torch

from retrieve import load_everything
from translate import embed, norm_text, video_to_landmarks


def load_decoder(path, device):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    cfg = PeftConfig.from_pretrained(path)
    base_name = cfg.base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(base_name)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base, path).to(device).eval()
    return tok, model, base_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--true", default=None)
    ap.add_argument("--decoder_dir", default="decoder_lora")
    ap.add_argument("--no_decoder", action="store_true")
    ap.add_argument("--device", default="auto", help="device for the decoder (auto, mps, cpu)")
    args = ap.parse_args()

    enc_model, db, label2id, device = load_everything()
    print(f"\nVideo: {args.video}")
    print("-" * 72)

    # ---- stages 1-4: video -> candidates
    t0 = time.time()
    seq, n_raw = video_to_landmarks(args.video)
    x, z = embed(enc_model, seq, device)
    sims = (z @ db["embeddings"].T)[0]
    top = sims.topk(args.k)
    cands = [{"sentence": str(db["sentences"][i]), "gloss": str(db["glosses"][i]), "sim": s}
             for s, i in zip(top.values.tolist(), top.indices.tolist())]
    print(f"1. MediaPipe landmarks : {n_raw} frames x 225 numbers")
    print(f"2. Normalize + resample: {tuple(seq.shape)}   [{100 * float((seq == 0).mean()):.0f}% of values missing]")
    print(f"3. Model input         : {tuple(x.shape)}  -> encoder -> fingerprint {tuple(z.shape[1:])}")
    print(f"4. Retrieval           : cosine similarity against {len(sims)} database rows ({time.time() - t0:.1f} s)")

    print("-" * 72)
    print(f"{'Rank':<5}{'Sim':<8}{'Sentence':<40}Gloss")
    for r, c in enumerate(cands, 1):
        mark = "   <-- correct" if args.true and norm_text(c["sentence"]) == norm_text(args.true) else ""
        print(f"{r:<5}{c['sim']:<8.3f}{c['sentence']:<40}{c['gloss']}{mark}")
    print("-" * 72)
    print(f"Retrieval answer (top-1): {cands[0]['sentence']}")

    # ---- stage 5: decoder
    if args.no_decoder:
        return
    try:
        from train_decoder_lora import build_prompt, choose, pick_device
        dev = pick_device(args.device)
        tok, dec, base_name = load_decoder(args.decoder_dir, dev)
        pick = choose(dec, tok, {"candidates": cands}, dev)
        print(f"Decoder answer ({base_name} + LoRA): {cands[pick]['sentence']}   (candidate {pick + 1})")
        if args.true:
            print("Decoder is", "correct" if norm_text(cands[pick]["sentence"]) == norm_text(args.true) else "wrong")
    except Exception as e:  # noqa: BLE001
        print(f"Decoder step failed ({e}). Retrieval result above is still valid; try --device cpu or --no_decoder.")


if __name__ == "__main__":
    main()