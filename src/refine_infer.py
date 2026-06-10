"""
学習済み1D-CNNで submission の Top-1 境界を補正し、test R1 を評価。

予測長 > max_len の超長区間は補正スキップ(Top-1のまま)。
補正前後の R1@0.7/@0.5 と長さ帯別を出力し、ルールベース(+1.41)・補正天井(+16.70)
と比較できるようにする。

Usage (サーバー, cwd=src):
    ../.venv/bin/python refine_infer.py \
        --submission ../results_v2_btfix/submission_test.jsonl \
        --gt ../data/castella_test_release.jsonl \
        --feat_dir ../features/castella/clap \
        --model ../results_refine/best.pt \
        --out ../results_refine/submission_test_cnn.jsonl
"""

import argparse
import json
import numpy as np
import torch

from refine_cnn import RefineCNN, decode


def cos_diff(feat_dir, vid):
    a = np.load(f"{feat_dir}/{vid}.npz")["features"].astype(np.float32)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    return 1.0 - np.sum(an[1:] * an[:-1], axis=1)


def gaussian(L, mu, s):
    x = np.arange(L, dtype=np.float32)
    return np.exp(-((x - mu) ** 2) / (2.0 * s * s))


def iou_1d(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1]); i = max(0.0, e - s)
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def best_iou(p, gs):
    return max((iou_1d(p, g) for g in gs), default=0.0)


def build_input(dd, s, e, W, sigma):
    T = len(dd) + 1
    center = int(round(0.5 * (s + e)))
    ws = center - W
    L = 2 * W + 1
    cha = np.zeros(L, dtype=np.float32)
    for k in range(L):
        p = ws + k
        if 1 <= p <= T - 1:
            cha[k] = dd[p - 1]
    chp = gaussian(L, s - ws, sigma) + gaussian(L, e - ws, sigma)
    X = np.stack([cha, chp])[None]  # (1, 2, L)
    return torch.from_numpy(X), ws


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--W", type=int, default=30)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--max_len", type=float, default=36.0)
    args = p.parse_args()

    model = RefineCNN()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    recs = [json.loads(l) for l in open(args.gt) if l.strip()]
    GT = {d["qid"]: d["relevant_windows"] for d in recs}
    VID = {d["qid"]: (d.get("vid") or d["qid"].rsplit("_", 1)[0]) for d in recs}
    sub = [json.loads(l) for l in open(args.submission) if l.strip()]

    cache = {}
    def diff(vid):
        if vid not in cache:
            cache[vid] = cos_diff(args.feat_dir, vid)
        return cache[vid]

    new, bef, aft = [], [], []
    n_refined = 0
    for d in sub:
        q = d["qid"]
        preds = d["pred_relevant_windows"]
        top1 = preds[0]
        s, e = float(top1[0]), float(top1[1])
        score = float(top1[2]) if len(top1) > 2 else 1.0
        new_top1 = [s, e, score]
        if q in GT and (e - s) <= args.max_len:
            try:
                X, ws = build_input(diff(VID[q]), s, e, args.W, args.sigma)
                with torch.no_grad():
                    out = model(X)[0]
                ns, ne = decode(out, ws)
                new_top1 = [ns, ne, score]
                n_refined += 1
            except Exception:
                pass
        d2 = dict(d)
        d2["pred_relevant_windows"] = [new_top1] + preds[1:]
        d2.pop("pred_saliency_scores", None)
        new.append(d2)
        if q in GT:
            bef.append(([s, e], GT[q]))
            aft.append((new_top1[:2], GT[q]))

    with open(args.out, "w") as f:
        for d in new:
            f.write(json.dumps(d) + "\n")

    def r1(items, t):
        return 100.0 * np.mean([best_iou(x, g) >= t for x, g in items]) if items else 0.0

    def band(g):
        L = g[1] - g[0]
        return "<=3s" if L <= 3 else ("3-10s" if L <= 10 else ">10s")

    print(f"補正適用 {n_refined}/{len(sub)} (長区間>{args.max_len}sはスキップ)")
    print(f"R1@0.7  before={r1(bef,0.7):.2f} -> CNN={r1(aft,0.7):.2f}  ({r1(aft,0.7)-r1(bef,0.7):+.2f})")
    print(f"R1@0.5  before={r1(bef,0.5):.2f} -> CNN={r1(aft,0.5):.2f}  ({r1(aft,0.5)-r1(bef,0.5):+.2f})")
    bands = {"<=3s": ([], []), "3-10s": ([], []), ">10s": ([], [])}
    for (xb, g), (xa, _) in zip(bef, aft):
        bn = band(max(g, key=lambda gg: iou_1d(xb, gg)))
        bands[bn][0].append((xb, g)); bands[bn][1].append((xa, g))
    for bn in ["<=3s", "3-10s", ">10s"]:
        b, a = bands[bn]
        print(f"  [{bn}] n={len(b)}  R1@0.7 {r1(b,0.7):.2f} -> {r1(a,0.7):.2f}")


if __name__ == "__main__":
    main()
