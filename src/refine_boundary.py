"""
C-team-style boundary refinement using saliency_scores.

Reads a submission jsonl that contains `pred_saliency_scores` (per-second
saliency from the model), and rewrites the Top-1 predicted window with a
locally optimal boundary that maximizes:

    refinement_score = contrast - alpha * move_penalty - beta * length_penalty

Top-2..K predictions are kept unchanged (C-team found Top-1-only refinement
to be the safest setting).

Usage:
    python refine_boundary.py \
        -i ../results/submission_test.jsonl \
        -o ../results/submission_test_refined.jsonl
"""

import json
import argparse
import numpy as np


def refine_one(top1, saliency, search_radius=5, alpha=0.1, beta=0.5):
    """Find a locally optimal [s', e'] within +-search_radius of original."""
    s, e, score = float(top1[0]), float(top1[1]), float(top1[2])
    T = len(saliency)
    saliency = np.asarray(saliency, dtype=np.float32)

    s_int = int(round(s))
    e_int = int(round(e))
    orig_len = max(e - s, 1e-6)

    best_score = -1e18
    best_cand = (s_int, e_int)

    for ds in range(-search_radius, search_radius + 1):
        for de in range(-search_radius, search_radius + 1):
            cs = s_int + ds
            ce = e_int + de
            # constraints: in [0, T], start < end
            if cs < 0 or ce > T or cs >= ce:
                continue

            inside = saliency[cs:ce]
            n_outside = T - (ce - cs)
            if n_outside <= 0:
                outside_mean = 0.0
            else:
                outside_sum = saliency.sum() - inside.sum()
                outside_mean = float(outside_sum / n_outside)
            inside_mean = float(inside.mean()) if len(inside) > 0 else 0.0
            contrast = inside_mean - outside_mean

            move_penalty = abs(cs - s) + abs(ce - e)
            length_penalty = abs((ce - cs) - orig_len) / orig_len

            ref_score = contrast - alpha * move_penalty - beta * length_penalty

            if ref_score > best_score:
                best_score = ref_score
                best_cand = (cs, ce)

    # keep the original confidence score, only move boundary
    return [float(best_cand[0]), float(best_cand[1]), score]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='move_penalty weight (larger = stick to original)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='length_penalty weight (larger = preserve length)')
    parser.add_argument('--radius', type=int, default=5,
                        help='search +- N seconds around the original boundary')
    args = parser.parse_args()

    refined = []
    n_changed = 0
    n_skipped = 0
    n_total = 0

    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            n_total += 1

            if 'pred_saliency_scores' not in data:
                n_skipped += 1
                refined.append(data)
                continue

            preds = data['pred_relevant_windows']
            saliency = data['pred_saliency_scores']
            if not preds or not saliency:
                refined.append(data)
                continue

            top1 = preds[0]
            new_top1 = refine_one(top1, saliency,
                                  search_radius=args.radius,
                                  alpha=args.alpha,
                                  beta=args.beta)

            if (new_top1[0] != top1[0]) or (new_top1[1] != top1[1]):
                n_changed += 1

            preds_new = [new_top1] + preds[1:]
            data_new = dict(data)
            data_new['pred_relevant_windows'] = preds_new
            # drop saliency from output to keep file small / match eval format
            data_new.pop('pred_saliency_scores', None)
            refined.append(data_new)

    with open(args.output, 'w') as f:
        for data in refined:
            f.write(json.dumps(data) + '\n')

    print(f"Total samples:  {n_total}")
    print(f"Refined:        {n_changed} ({100.0 * n_changed / max(n_total, 1):.1f}%)")
    print(f"Skipped (no saliency): {n_skipped}")
    print(f"Settings: alpha={args.alpha}, beta={args.beta}, radius={args.radius}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
