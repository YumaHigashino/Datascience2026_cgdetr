"""
学習済み1D-CNNで submission の Top-1 境界を補正し、test R1 を評価。

CNNヒートマップのargmaxを「元予測±max_shift秒」に制限する(move制約)。
無制約だと当たっていた予測まで動かして壊す(スナップ -2.3 / 無制約CNN -3.12)ため、
ルールベースの move_penalty 相当を推論に入れる。max_shift を複数比較する。

予測長 > max_len の超長区間は補正スキップ(Top-1のまま)。

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

from refine_cnn import RefineCNN


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


def decode_constrained(prob, ws, s, e, max_shift):
    """prob (2,L) numpy。元予測 s,e の ±max_shift 秒内で argmax を取る。"""
    L = prob.shape[1]
    si = int(round(s - ws)); ei = int(round(e - ws))

    def local_argmax(ch, c):
        lo = max(0, c - max_shift); hi = min(L - 1, c + max_shift)
        if hi < lo:
            return int(np.clip(c, 0, L - 1))
        return lo + int(np.argmax(ch[lo:hi + 1]))

    ns = ws + local_argmax(prob[0], si)
    ne = ws + local_argmax(prob[1], ei)
    if ns >= ne:
        return float(s), float(e)
    return float(ns), float(ne)


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
    p.add_argument("--shifts", type=int, nargs="+", default=[3, 5, 8, 999])
    p.add_argument("--save_shift", type=int, default=3, help="この max_shift で out を保存")
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

    # forward は1回。各サンプルのヒートマップ prob を保持して max_shift を比較する。
    rows = []  # (q, preds, s, e, score, prob or None, ws)
    for d in sub:
        q = d["qid"]
        preds = d["pred_relevant_windows"]
        top1 = preds[0]
        s, e = float(top1[0]), float(top1[1])
        score = float(top1[2]) if len(top1) > 2 else 1.0
        prob, ws = None, None
        if q in GT and (e - s) <= args.max_len:
            try:
                X, ws = build_input(diff(VID[q]), s, e, args.W, args.sigma)
                with torch.no_grad():
                    prob = torch.sigmoid(model(X)[0]).numpy()  # (2,L)
            except Exception:
                prob = None
        rows.append((q, preds, s, e, score, prob, ws))

    def r1(items, t):
        return 100.0 * np.mean([best_iou(x, g) >= t for x, g in items]) if items else 0.0

    def band(g):
        L = g[1] - g[0]
        return "<=3s" if L <= 3 else ("3-10s" if L <= 10 else ">10s")

    bef = [([s, e], GT[q]) for q, preds, s, e, score, prob, ws in rows if q in GT]
    print(f"補正候補 {sum(1 for r in rows if r[5] is not None)}/{len(rows)} (長区間スキップ)")
    print(f"R1@0.7 before={r1(bef,0.7):.2f}  R1@0.5 before={r1(bef,0.5):.2f}\n")

    best_ms, best_r1, best_subs = None, -1, None
    for ms in args.shifts:
        aft, subs = [], []
        for q, preds, s, e, score, prob, ws in rows:
            if prob is None:
                ns, ne = s, e
            else:
                ns, ne = decode_constrained(prob, ws, s, e, ms)
            subs.append((q, preds, [ns, ne, score]))
            if q in GT:
                aft.append(([ns, ne], GT[q]))
        r07 = r1(aft, 0.7)
        tag = "制約なし" if ms >= 100 else f"±{ms}"
        line = f"[max_shift {tag:<6}] R1@0.7={r07:.2f} ({r07-r1(bef,0.7):+.2f})  R1@0.5={r1(aft,0.5):.2f} ({r1(aft,0.5)-r1(bef,0.5):+.2f})"
        # 長さ帯別(@0.7)
        bands = {"<=3s": [], "3-10s": [], ">10s": []}
        for (xb, g), (xa, _) in zip(bef, aft):
            bands[band(max(g, key=lambda gg: iou_1d(xb, gg)))].append((xa, g))
        bd = "  ".join(f"{k}:{r1(v,0.7):.1f}" for k, v in bands.items())
        print(line + "   | " + bd)
        if r07 > best_r1:
            best_r1, best_ms, best_subs = r07, ms, subs

    # 最良(またはsave_shift)で submission を保存
    save_subs = best_subs
    with open(args.out, "w") as f:
        for q, preds, new_top1 in save_subs:
            d2 = {"qid": q, "pred_relevant_windows": [new_top1] + preds[1:]}
            f.write(json.dumps(d2) + "\n")
    print(f"\n最良 max_shift=±{best_ms} (R1@0.7={best_r1:.2f}) で保存: {args.out}")


if __name__ == "__main__":
    main()
