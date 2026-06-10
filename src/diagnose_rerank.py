"""
リランカー設計のための診断。2つの問いに同時に答える。

(A) 正解候補は今「何位」にいるか
    → 道具の作りやすさ。2位に多い=簡単、5位以下に散る=難しい。

(B) どんな信号が正解候補を見分けられそうか
    正解候補 と (Top-1だが不正解の候補) を比べ、以下が分離できるか測る:
      - saliencyコントラスト (区間内平均 - 区間外平均)
      - 候補の長さ
    分離できる信号ほど、リランカーの特徴量として有望。

GPU不要。submission(pred_saliency_scores付き) + GT のみ。

Usage:
    python diagnose_rerank.py \
        -s ../../team_predictions/submission_test.jsonl \
        -g ../../Datascience2026-d/data/castella_test_release.jsonl --thd 0.7
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


def saliency_contrast(pred, saliency):
    """候補区間 [s,e] の内側平均 - 外側平均。saliencyは1秒=1要素とみなす。"""
    if saliency is None or len(saliency) == 0:
        return None
    sal = np.asarray(saliency, dtype=np.float32)
    T = len(sal)
    s = max(0, int(round(float(pred[0]))))
    e = min(T, int(round(float(pred[1]))))
    if e <= s:
        return None
    inside = sal[s:e]
    n_out = T - (e - s)
    out_mean = (sal.sum() - inside.sum()) / n_out if n_out > 0 else 0.0
    return float(inside.mean() - out_mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", "-s", required=True)
    parser.add_argument("--gt", "-g", required=True)
    parser.add_argument("--thd", type=float, default=0.7)
    parser.add_argument("--max_k", type=int, default=10)
    args = parser.parse_args()

    sub = load_jsonl(args.submission)
    gt = load_jsonl(args.gt)
    gt_by_qid = {d["qid"]: d["relevant_windows"] for d in gt}

    # --- (A) 正解候補の最小ランク分布 ---
    rank_bins = {"rank1(既に正解)": 0, "rank2": 0, "rank3-5": 0, "rank6-10": 0, "miss(候補に無し)": 0}
    n = 0

    # --- (B) 信号の分離能: 正解候補 vs Top1不正解候補 ---
    pos_contrast, neg_contrast = [], []   # saliencyコントラスト
    pos_len, neg_len = [], []             # 候補長さ
    top1_score_diff = []                  # 正解候補スコア - Top1スコア（負なら確信度が正解を過小評価）

    for d in sub:
        q = d["qid"]
        if q not in gt_by_qid:
            continue
        gts = gt_by_qid[q]
        preds = d["pred_relevant_windows"][:args.max_k]
        saliency = d.get("pred_saliency_scores")
        if not gts or not preds:
            continue
        n += 1

        ious = [best_iou_over_gts(p, gts) for p in preds]
        hit_ranks = [i for i, v in enumerate(ious) if v >= args.thd]  # 0-indexed

        if not hit_ranks:
            rank_bins["miss(候補に無し)"] += 1
        else:
            r = hit_ranks[0] + 1  # 1-indexed 最小ランク
            if r == 1:
                rank_bins["rank1(既に正解)"] += 1
            elif r == 2:
                rank_bins["rank2"] += 1
            elif r <= 5:
                rank_bins["rank3-5"] += 1
            else:
                rank_bins["rank6-10"] += 1

        # (B) Top-1が不正解(ious[0] < thd) かつ 候補内に正解がある時だけ、
        #     正解候補 vs 現Top-1 を比較する（これがリランカーが直すべきケース）
        if ious[0] < args.thd and hit_ranks:
            best_idx = max(range(len(preds)), key=lambda i: ious[i])
            pos, neg = preds[best_idx], preds[0]
            cp, cn = saliency_contrast(pos, saliency), saliency_contrast(neg, saliency)
            if cp is not None and cn is not None:
                pos_contrast.append(cp); neg_contrast.append(cn)
            pos_len.append(float(pos[1]) - float(pos[0]))
            neg_len.append(float(neg[1]) - float(neg[0]))
            if len(pos) >= 3 and len(neg) >= 3:
                top1_score_diff.append(float(pos[2]) - float(neg[2]))

    # ===== 出力 (A) =====
    print(f"n = {n} queries,  IoU閾値 = {args.thd}\n")
    print("=== (A) 正解候補の最小ランク分布 ===")
    for k, v in rank_bins.items():
        print(f"  {k:<18} {v:>5}  ({100.0*v/n:5.2f}%)")
    救える = rank_bins["rank2"] + rank_bins["rank3-5"] + rank_bins["rank6-10"]
    print(f"  --> リランキングで救える(rank>=2に正解): {救える} ({100.0*救える/n:.2f}%)")
    print(f"  --> 並べ替えでは無理(miss): {rank_bins['miss(候補に無し)']} "
          f"({100.0*rank_bins['miss(候補に無し)']/n:.2f}%)")

    # ===== 出力 (B) =====
    m = len(pos_len)
    print(f"\n=== (B) 信号の分離能（Top-1が外したが候補内に正解がある {m} 件） ===")
    if m == 0:
        print("  該当ケース無し")
        return

    def stat(a):
        a = np.asarray(a, dtype=np.float32)
        return a.mean(), a.std()

    if pos_contrast:
        pm, ps = stat(pos_contrast); nm, ns = stat(neg_contrast)
        win = 100.0 * np.mean(np.array(pos_contrast) > np.array(neg_contrast))
        print(f"  saliencyコントラスト: 正解候補 {pm:.4f}±{ps:.4f}  vs  現Top-1 {nm:.4f}±{ns:.4f}")
        print(f"    -> 正解候補の方が高い割合: {win:.1f}%  (50%なら無力, 高いほど有望)")

    pm, ps = stat(pos_len); nm, ns = stat(neg_len)
    short_win = 100.0 * np.mean(np.array(pos_len) < np.array(neg_len))
    print(f"  候補の長さ(秒):        正解候補 {pm:.2f}±{ps:.2f}  vs  現Top-1 {nm:.2f}±{ns:.2f}")
    print(f"    -> 正解候補の方が短い割合: {short_win:.1f}%  (現Top-1が長すぎ説の検証)")

    if top1_score_diff:
        sm, ss = stat(top1_score_diff)
        print(f"  確信度スコア差(正解候補 - 現Top-1): {sm:.4f}±{ss:.4f}  (負=モデルが正解を過小評価)")


if __name__ == "__main__":
    main()
