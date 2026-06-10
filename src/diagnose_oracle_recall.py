"""
Oracle Recall@K 診断: リランキングの「天井」を測る。

公式R1(eval.py compute_mr_r1)は予測Top-1のみ採点。複数GTがある場合は
「予測に最も近いGT」を正解扱いする寛大な定義。これをTop-K候補に拡張し、
「Top-K候補のどれか × GTのどれか」で最大IoUを取って当たり判定する。

  Oracle@K = 候補上位K個の中に正解が眠っている率
  Gap@K    = Oracle@K - R1(Top-1)  ← リランキングで取りうる最大の伸びしろ

GPU不要。submission jsonl + GT jsonl だけで動く。

Usage:
    python diagnose_oracle_recall.py \
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
    """1次元 temporal IoU。pred, gt = [start, end]."""
    ps, pe = float(pred[0]), float(pred[1])
    gs, ge = float(gt[0]), float(gt[1])
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def best_iou_over(preds, gts):
    """preds(複数候補) × gts(複数正解) の全ペアで最大IoUと、その時のGTを返す。"""
    best = 0.0
    best_gt = gts[0] if gts else None
    for p in preds:
        for g in gts:
            v = iou_1d(p, g)
            if v > best:
                best = v
                best_gt = g
    return best, best_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", "-s", required=True)
    parser.add_argument("--gt", "-g", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--thds", type=float, nargs="+", default=[0.5, 0.7])
    args = parser.parse_args()

    sub = load_jsonl(args.submission)
    gt = load_jsonl(args.gt)

    gt_by_qid = {d["qid"]: d["relevant_windows"] for d in gt}
    pred_by_qid = {d["qid"]: d["pred_relevant_windows"] for d in sub}

    common = [q for q in pred_by_qid if q in gt_by_qid]
    print(f"submission: {len(pred_by_qid)} qids / GT: {len(gt_by_qid)} qids / 共通: {len(common)}")
    if not common:
        print("!! 共通qidが0。qidの形式が submission と GT で食い違っている可能性。")
        return

    max_k = max(args.ks)
    # 各qidについて、Top-K(K=各値) での最大IoU を記録
    # best_iou_at_k[qid][k] = Top-k候補での最大IoU
    rows = {}        # k -> list of best_iou per qid
    for k in args.ks:
        rows[k] = []
    # 長さ帯別(全候補=max_k での最良GTの長さで分類)
    len_band_hits = {b: {"n": 0, "hit_top1": 0, "hit_oracle": 0}
                     for b in ["<=3s", "3-10s", ">10s"]}

    for q in common:
        gts = gt_by_qid[q]
        preds_all = pred_by_qid[q]
        if not gts or not preds_all:
            continue
        # 各K
        for k in args.ks:
            preds_k = preds_all[:k]
            bi, _ = best_iou_over(preds_k, gts)
            rows[k].append(bi)

        # 長さ帯: max_k 候補での最良GT長で分類、Top-1当たりとoracle当たり(@0.7)を集計
        top1_iou, _ = best_iou_over(preds_all[:1], gts)
        oracle_iou, best_gt = best_iou_over(preds_all[:max_k], gts)
        glen = float(best_gt[1]) - float(best_gt[0])
        band = "<=3s" if glen <= 3 else ("3-10s" if glen <= 10 else ">10s")
        len_band_hits[band]["n"] += 1
        len_band_hits[band]["hit_top1"] += int(top1_iou >= 0.7)
        len_band_hits[band]["hit_oracle"] += int(oracle_iou >= 0.7)

    n = len(rows[args.ks[0]])
    print(f"\n=== Oracle Recall（候補に正解が眠っている率） n={n} ===")
    header = "          " + "".join([f"IoU>={t:<7}" for t in args.thds])
    print(header)
    base = {}  # thd -> R1(Top-1) を覚えてGap計算に使う
    for k in args.ks:
        arr = np.array(rows[k])
        cells = []
        for t in args.thds:
            r = 100.0 * np.mean(arr >= t)
            cells.append(f"{r:<11.2f}")
            if k == 1:
                base[t] = r
        label = f"R1(Top-1)" if k == 1 else f"Oracle@{k}"
        print(f"{label:<10}" + "".join(cells))

    print(f"\n=== Gap = Oracle@K - R1(Top-1)  ← リランキングの天井（伸びしろ） ===")
    for k in args.ks:
        if k == 1:
            continue
        arr = np.array(rows[k])
        cells = []
        for t in args.thds:
            gap = 100.0 * np.mean(arr >= t) - base[t]
            cells.append(f"+{gap:<10.2f}")
        print(f"Gap@{k:<6}" + "".join(cells))

    print(f"\n=== 長さ帯別（@0.7, Top-{max_k}候補での最良GT長で分類） ===")
    print("帯        件数   Top-1当たり%   Oracle当たり%   伸びしろ")
    for b in ["<=3s", "3-10s", ">10s"]:
        d = len_band_hits[b]
        if d["n"] == 0:
            continue
        t1 = 100.0 * d["hit_top1"] / d["n"]
        orc = 100.0 * d["hit_oracle"] / d["n"]
        print(f"{b:<9} {d['n']:<6} {t1:<14.2f} {orc:<15.2f} +{orc - t1:.2f}")


if __name__ == "__main__":
    main()
