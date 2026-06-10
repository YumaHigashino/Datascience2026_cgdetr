"""
音響変化点スナップの簡易ルールベース検証（CNNの下限ベースライン）。

Top-1端点を ±radius 内で音響変化点(cos距離)が最大の位置にスナップし、
R1@0.7/@0.5 が改善するかを測る。学習不要。

  改善 > +1.41 → スナップ自体が成果。CNNはさらに上積みを狙う
  改善 < +1.41 → 単純スナップは不十分=CNN(文脈学習)の必要性が確定
  悪化         → 境界以外のピークを誤吸着。radius縮小 or CNN必須

サーバー実行(特徴 features/castella/clap が必要)。

Usage:
    .venv/bin/python src/diagnose_audio_snap.py \
        --submission results_v2_btfix/submission_test.jsonl \
        --gt data/castella_test_release.jsonl \
        --feat_dir features/castella/clap
"""

import numpy as np
import json
import argparse


def iou_1d(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1]); i = max(0.0, e - s)
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def best_iou(p, gts):
    return max((iou_1d(p, g) for g in gts), default=0.0)


def cos_diff(feat_dir, vid):
    a = np.load(f"{feat_dir}/{vid}.npz")["features"]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    return 1.0 - np.sum(an[1:] * an[:-1], axis=1)   # (T-1,)


def snap(p, d, r):
    T = len(d) + 1
    lo, hi = max(1, p - r), min(T - 1, p + r)
    return max(range(lo, hi + 1), key=lambda q: d[q - 1]) if hi >= lo else p


def band(g):
    L = g[1] - g[0]
    return "<=3s" if L <= 3 else ("3-10s" if L <= 10 else ">10s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--feat_dir", required=True)
    parser.add_argument("--radii", type=int, nargs="+", default=[3, 5])
    args = parser.parse_args()

    recs = [json.loads(l) for l in open(args.gt) if l.strip()]
    GT = {d["qid"]: d["relevant_windows"] for d in recs}
    VID = {d["qid"]: (d.get("vid") or d["qid"].rsplit("_", 1)[0]) for d in recs}
    sub = [json.loads(l) for l in open(args.submission) if l.strip()]

    def r1(items, t):
        return 100.0 * np.mean([best_iou(x, g) >= t for x, g in items]) if items else 0.0

    for R in args.radii:
        bef, aft = [], []
        bb = {"<=3s": [], "3-10s": [], ">10s": []}
        ba = {"<=3s": [], "3-10s": [], ">10s": []}
        for d in sub:
            q = d["qid"]
            if q not in GT:
                continue
            gts = GT[q]
            p = d["pred_relevant_windows"][0][:2]
            try:
                dd = cos_diff(args.feat_dir, VID[q])
            except Exception:
                continue
            s0, e0 = int(round(p[0])), int(round(p[1]))
            s1, e1 = snap(s0, dd, R), snap(e0, dd, R)
            if s1 >= e1:
                s1, e1 = s0, e0
            bef.append((p, gts)); aft.append(([s1, e1], gts))
            bn = band(max(gts, key=lambda g: iou_1d(p, g)))
            bb[bn].append((p, gts)); ba[bn].append(([s1, e1], gts))

        print(f"=== radius ±{R}  (n={len(bef)}) ===")
        print(f"  R1@0.7 before={r1(bef,0.7):.2f} -> snap={r1(aft,0.7):.2f}  ({r1(aft,0.7)-r1(bef,0.7):+.2f})")
        print(f"  R1@0.5 before={r1(bef,0.5):.2f} -> snap={r1(aft,0.5):.2f}  ({r1(aft,0.5)-r1(bef,0.5):+.2f})")
        for bn in ["<=3s", "3-10s", ">10s"]:
            print(f"    [{bn}] n={len(bb[bn])}  R1@0.7 {r1(bb[bn],0.7):.2f} -> {r1(ba[bn],0.7):.2f}")


if __name__ == "__main__":
    main()
