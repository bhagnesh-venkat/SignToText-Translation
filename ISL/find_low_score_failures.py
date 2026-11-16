"""
Find test videos where retrieval fails AND the best-match similarity is low.

For every test video it records the top-1 similarity, whether top-1 is correct, the rank of the
true sentence and the gap between the first two scores. It then prints:
  1. the failures with the LOWEST top-1 similarity (good examples of a low-confidence failure),
     each with a ready-to-run translate.py command,
  2. accuracy by top-1 similarity group, to show whether a low score really means a low chance
     of being correct,
  3. the average top-1 score of correct and of wrong videos.

Run inside the ISL folder with slt-env active:
    python find_low_score_failures.py 2>/dev/null

Output: printed tables and score_analysis.csv (one row per test video).
Nothing is trained or overwritten; it only uses encoder.pt, database.pt, test.csv and landmarks1/.
"""
import argparse
import csv
from pathlib import Path

VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def score_rows(test_rows, retrieve_fn, label2id):
    """retrieve_fn(file) -> list of (similarity, row_index) sorted by similarity, for ALL sentences."""
    out = []
    for f, label in test_rows:
        res = retrieve_fn(f)
        true_idx = label2id[label]
        order = [i for _, i in res]
        out.append({
            'file': f, 'label': label,
            'top1_sim': res[0][0], 'top1_idx': res[0][1],
            'gap': res[0][0] - res[1][0],
            'true_rank': order.index(true_idx) + 1,
            'true_sim': dict((i, s) for s, i in res)[true_idx],
        })
    return out


def accuracy_by_score(rows, groups=4):
    """Split the videos into equal-size groups by top-1 similarity (lowest first)."""
    ordered = sorted(rows, key=lambda r: r['top1_sim'])
    n = len(ordered)
    result = []
    for g in range(groups):
        part = ordered[g * n // groups:(g + 1) * n // groups]
        correct = sum(r['true_rank'] == 1 for r in part)
        result.append({'lo': part[0]['top1_sim'], 'hi': part[-1]['top1_sim'], 'n': len(part), 'correct': correct})
    return result


def video_index(root):
    idx = {}
    if Path(root).exists():
        for p in Path(root).rglob('*'):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT and 'Videos_Sentence_Level' in p.parts:
                idx[(p.parent.name, p.stem)] = p
    return idx


def raw_path(file, index):
    name = file[:-4] if file.endswith('.npy') else file
    if '__' not in name:
        return None
    folder, stem = name.split('__', 1)
    return index.get((folder, stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks_dir', default='landmarks1')
    ap.add_argument('--videos', default='isl_data')
    ap.add_argument('--show', type=int, default=8, help='how many low-score failures to list')
    ap.add_argument('--csv', default='score_analysis.csv')
    args = ap.parse_args()

    import numpy as np
    from retrieve import load_everything, retrieve
    from train_encoder import read_csv

    model, db, label2id, device = load_everything()
    sentences = db['sentences']
    n_rows = len(sentences)
    test_rows = read_csv('test.csv')
    rows = score_rows(test_rows, lambda f: retrieve(model, db, f'{args.landmarks_dir}/{f}', device, k=n_rows), label2id)

    wrong = [r for r in rows if r['true_rank'] != 1]
    right = [r for r in rows if r['true_rank'] == 1]
    print(f'\ntest videos: {len(rows)} | correct at rank 1: {len(right)} | wrong: {len(wrong)}')
    print(f"average top-1 similarity: correct videos {np.mean([r['top1_sim'] for r in right]):.3f} | "
          f"wrong videos {np.mean([r['top1_sim'] for r in wrong]):.3f}")

    print('\nAccuracy by top-1 similarity (4 equal groups, lowest scores first):')
    print(f"{'similarity range':<20}{'videos':>8}{'correct':>9}{'accuracy':>10}")
    for g in accuracy_by_score(rows):
        print(f"{g['lo']:.3f} - {g['hi']:.3f}{'':<6}{g['n']:>8}{g['correct']:>9}{100 * g['correct'] / g['n']:>9.0f}%")

    index = video_index(args.videos)
    print(f'\nThe {args.show} wrong videos with the LOWEST top-1 similarity:')
    for r in sorted(wrong, key=lambda r: r['top1_sim'])[:args.show]:
        print('=' * 78)
        print(f"video: {r['file']}")
        print(f"  true sentence : {sentences[label2id[r['label']]]}   (true rank {r['true_rank']}, similarity {r['true_sim']:.3f})")
        print(f"  retrieved     : {sentences[r['top1_idx']]}   (similarity {r['top1_sim']:.3f}, gap to rank 2: {r['gap']:.3f})")
        p = raw_path(r['file'], index)
        if p is not None:
            print(f'  run it live:\n    python translate.py "{p}" --true "{sentences[label2id[r["label"]]]}" 2>/dev/null')

    with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['video', 'true_sentence', 'top1_sentence', 'top1_similarity', 'gap_to_rank2', 'true_rank', 'true_similarity'])
        for r in sorted(rows, key=lambda r: r['top1_sim']):
            w.writerow([r['file'], sentences[label2id[r['label']]], sentences[r['top1_idx']],
                        f"{r['top1_sim']:.4f}", f"{r['gap']:.4f}", r['true_rank'], f"{r['true_sim']:.4f}"])
    print(f'\nsaved {args.csv}')


if __name__ == '__main__':
    main()