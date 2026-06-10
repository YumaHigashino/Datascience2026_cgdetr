"""
補正(境界調整)の「天井」を測る。

Top-1の境界 [s,e] を ±radius 秒の範囲で神様が最適に動かせたら、
R1@0.7 はどこまで上がるか。これが学習/ルールベース問わず
「境界補正で取りうる上限」。既存ルールベース(+1.41)と比べ、
学習版MLPに投資する価値があるかを判断する。

GPU不要。submission(Top-1) + GT のみ。

Usage:
    python diagnose_refine_ceiling.py \
        -s ../../team_predictions/submission_test.jsonl \
        -g ../../Datascience2026-d/data/castella_test_release.jsonl
"""

import json
import argparse
import numpy as np


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iou_1d(pred, gt):
    ps, pe = float(pred[0]), float(pred[1])
    gs, ge = float(gt[0]), float(gt[1])
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def best_iou_over_gts(pred, gts):
    return max((iou_1d(pred, g) for g in gts), default=0.0)


def ceiling_iou(top1, gts, radius):
    """±radius内で整数秒の全(s',e')を試し、GTとの最大IoUを返す。"""
    s0 = int(round(float(top1[0])))
    e0 = int(round(float(top1[1])))
    best = best_iou_over_gts([s0, e0], gts)
    for ds in range(-radius, radius + 1):
        for de in range(-radius, radius + 1):
            s, e = s0 + ds, e0 + de
            if s < 0 or s >= e:
                continue
            v = best_iou_over_gts([s, e], gts)
            if v > best:
                best = v
    return best


def band_of(glen):
    return "<=3s" if glen <= 3 else ("3-10s" if glen <= 10 else ">10s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", "-s", required=True)
    parser.add_argument("--gt", "-g", required=True)
    parser.add_argument("--radii", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--thd", type=float, default=0.7)
    args = parser.parse_args()

    sub = load_jsonl(args.submission)
    gt = load_jsonl(args.gt)
    gt_by_qid = {d["qid"]: d["relevant_windows"] for d in gt}

    recs = []  # (top1, gts, 現IoU, 最寄りGT長)
    for d in sub:
        q = d["qid"]
        if q not in gt_by_qid:
            continue
        gts = gt_by_qid[q]
        preds = d["pred_relevant_windows"]
        if not gts or not preds:
            continue
        top1 = preds[0][:2]
        now = best_iou_over_gts(top1, gts)
        # 最寄りGT（現Top-1基準）の長さで帯分け
        best_g = max(gts, key=lambda g: iou_1d(top1, g))
        glen = float(best_g[1]) - float(best_g[0])
        recs.append((top1, gts, now, glen))

    n = len(recs)
    now_r1 = 100.0 * np.mean([r[2] >= args.thd for r in recs])
    print(f"n = {n},  IoU閾値 = {args.thd}")
    print(f"現状 R1@{args.thd} (Top-1そのまま) = {now_r1:.2f}\n")

    print(f"=== 補正の天井（±radius内で最適境界） ===")
    print(f"{'radius':<8}{'天井R1':<10}{'gap(天井-現状)':<16}")
    for r in args.radii:
        ceil = 100.0 * np.mean([ceiling_iou(t, g, r) >= args.thd for t, g, _, _ in recs])
        print(f"±{r:<6} {ceil:<10.2f}+{ceil - now_r1:<.2f}")

    # 長さ帯別（radius=5固定）
    R = 5
    print(f"\n=== 長さ帯別の天井（±{R}秒, @{args.thd}） ===")
    print(f"{'帯':<8}{'件数':<7}{'現状%':<9}{'天井%':<9}{'伸びしろ':<8}")
    bands = {"<=3s": [], "3-10s": [], ">10s": []}
    for top1, gts, now, glen in recs:
        bands[band_of(glen)].append((now, ceiling_iou(top1, gts, R)))
    for b in ["<=3s", "3-10s", ">10s"]:
        arr = bands[b]
        if not arr:
            continue
        now_p = 100.0 * np.mean([a[0] >= args.thd for a in arr])
        ceil_p = 100.0 * np.mean([a[1] >= args.thd for a in arr])
        print(f"{b:<8}{len(arr):<7}{now_p:<9.2f}{ceil_p:<9.2f}+{ceil_p - now_p:.2f}")


if __name__ == "__main__":
    main()
