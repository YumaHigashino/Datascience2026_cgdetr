#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量子化診断 — 1秒グリッド丸め(postprocessing.round_multiple, clip_length=1)が
短モーメントの R1@0.7 をどれだけ奪っているかを、既存 submission(整数) と GT から推定する。

出力:
  (A) GT境界の小数部分布 ........ 仮説の生死判定(GTが連続秒なら丸めは有害 / 整数なら無害)
  (B) 長さ別 R1@0.7/0.5/平均IoU .. どの長さ帯で負けているか
  (C) 丸め除去の楽観的上限 ....... 各 pred 端を ±0.5s で最適化したときの到達可能 R1@0.7
                                  (= 丸めが唯一の誤差だった場合に回復しうる上限。実効果ではない)

GPU不要・依存は numpy のみ・既存の submission + GT だけで動く。

  python diagnose_quantization.py \
      --gt   data/castella_test_release.jsonl \
      --pred results_v2_btfix/test_hybrid_top1.jsonl
"""
import json
import argparse
import numpy as np


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iou_1d(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 1e-9 else 0.0


def best_iou(pred, gts):
    """eval 準拠: pred Top-1 と最大 IoU の GT window を採用。"""
    bi, bg = -1.0, gts[0]
    for g in gts:
        v = iou_1d(pred, g)
        if v > bi:
            bi, bg = v, g
    return bi, bg


def dequant_upper_iou(pred, gts, r=0.5, step=0.05):
    """pred の各端を ±r 動かせた場合に到達できる最大 IoU(全 GT で最大化)。
    round_multiple で整数化される前の連続値は [round-0.5, round+0.5] に存在したはず
    なので、この範囲での最良値が『丸め除去で回復しうる上限』になる。"""
    s0, e0 = pred[0], pred[1]
    ss = np.arange(s0 - r, s0 + r + 1e-9, step)
    ee = np.arange(e0 - r, e0 + r + 1e-9, step)
    best = 0.0
    for g in gts:
        for s in ss:
            for e in ee:
                if e - s <= 0:
                    continue
                v = iou_1d((s, e), g)
                if v > best:
                    best = v
    return best


def bucket_of(length):
    if length <= 3:
        return '<=3s'
    if length <= 10:
        return '3-10s'
    return '>10s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--dequant_radius', type=float, default=0.5,
                    help='丸め幅の半分(clip_length=1 なら 0.5)')
    ap.add_argument('--step', type=float, default=0.05)
    args = ap.parse_args()

    gt_rows = load_jsonl(args.gt)
    pred_rows = load_jsonl(args.pred)
    gt_by_qid = {str(d['qid']): d['relevant_windows'] for d in gt_rows}

    # ---- (A) GT境界の小数部分布 = 仮説の生死 ----
    fracs = []
    for d in gt_rows:
        for w in d['relevant_windows']:
            for x in (w[0], w[1]):
                fracs.append(abs(x - round(x)))
    fracs = np.array(fracs)
    frac_integer = float(np.mean(fracs < 1e-6)) if len(fracs) else 1.0
    print("=" * 68)
    print("(A) GT境界の小数部 — 仮説の生死判定")
    print("-" * 68)
    print(f"  GT境界端の総数            : {len(fracs)}")
    print(f"  整数ちょうど(|frac|<1e-6) : {frac_integer * 100:.1f}%")
    if len(fracs):
        print(f"  小数部 平均 {fracs.mean():.3f} / 中央値 {np.median(fracs):.3f}")
        hist, edges = np.histogram(fracs, bins=[0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5001])
        for i in range(len(hist)):
            print(f"    [{edges[i]:.2f}, {edges[i + 1]:.2f}) : {hist[i]:>6}  ({hist[i] / len(fracs) * 100:.1f}%)")
    if frac_integer > 0.95:
        print("  >> GTはほぼ整数秒。round_multiple は無害。仮説は棄却方向。")
    else:
        print("  >> GTは連続秒。round_multiple は短モーメントで損害。仮説は生存。")

    # ---- (B)(C) 長さ別 R1 と 丸め除去の上限 ----
    buckets = ['<=3s', '3-10s', '>10s', 'all']
    agg = {b: {'iou': [], 'dq': [], 'cerr': [], 'lr': [], 'iou0': []} for b in buckets}
    n_missing = 0
    for d in pred_rows:
        qid = str(d['qid'])
        if qid not in gt_by_qid:
            n_missing += 1
            continue
        gts = gt_by_qid[qid]
        if not gts or not d.get('pred_relevant_windows'):
            continue
        pred = d['pred_relevant_windows'][0][:2]
        bi, bg = best_iou(pred, gts)
        dq = dequant_upper_iou(pred, gts, r=args.dequant_radius, step=args.step)
        b = bucket_of(bg[1] - bg[0])
        pc = (pred[0] + pred[1]) / 2.0
        gc = (bg[0] + bg[1]) / 2.0
        plen = pred[1] - pred[0]
        glen = bg[1] - bg[0]
        for key in (b, 'all'):
            agg[key]['iou'].append(bi)
            agg[key]['dq'].append(dq)
            agg[key]['cerr'].append(pc - gc)
            agg[key]['lr'].append(plen / glen if glen > 1e-9 else 0.0)
            agg[key]['iou0'].append(1.0 if bi < 1e-9 else 0.0)

    print()
    print("=" * 68)
    print("(B)(C) 長さ別 R1@0.7 — 現状(整数) vs 丸め除去の上限")
    print("-" * 68)
    hdr = f"{'bucket':>7} | {'n':>5} | {'R1@0.7':>7} | {'上限':>7} | {'差':>6} | {'R1@0.5':>7} | {'meanIoU':>8}"
    print(hdr)
    print("-" * len(hdr))
    for b in buckets:
        iou = np.array(agg[b]['iou'])
        dq = np.array(agg[b]['dq'])
        if len(iou) == 0:
            continue
        r7 = np.mean(iou >= 0.7) * 100
        r7dq = np.mean(dq >= 0.7) * 100
        r5 = np.mean(iou >= 0.5) * 100
        print(f"{b:>7} | {len(iou):>5} | {r7:>6.2f}% | {r7dq:>6.2f}% | {r7dq - r7:>+5.2f} | {r5:>6.2f}% | {iou.mean():>8.3f}")
    if n_missing:
        print(f"  (注: GTに無い qid {n_missing} 件をスキップ)")
    print()
    print(f"注: 『上限』は各 pred 端を ±{args.dequant_radius}s で最適化した場合の到達可能 R1@0.7。")
    print("    真の連続値がこの最適点にある保証はないので楽観的上限。実効果は round 除去の再推論で測る。")

    # ---- (D) 局在診断: 中心を外しているか / 長く出しすぎか ----
    print()
    print("=" * 68)
    print("(D) 局在診断 — 中心を外しているか / 長く出しすぎか")
    print("-" * 68)
    hdr2 = f"{'bucket':>7} | {'n':>5} | {'|中心誤差|中央':>11} | {'中心誤差中央(符号)':>14} | {'長さ比中央':>9} | {'IoU=0%':>7}"
    print(hdr2)
    print("-" * 70)
    for b in buckets:
        ce = np.array(agg[b]['cerr'])
        lr = np.array(agg[b]['lr'])
        i0 = np.array(agg[b]['iou0'])
        if len(ce) == 0:
            continue
        print(f"{b:>7} | {len(ce):>5} | {np.median(np.abs(ce)):>11.2f} | {np.median(ce):>14.2f} | {np.median(lr):>9.2f} | {np.mean(i0) * 100:>6.1f}%")
    print()
    print("読み: |中心誤差|が大 → 局在自体に失敗(中心を外す)。長さ比>>1 → 長く出しすぎ。")
    print("      長さ比≈1かつ中心誤差小なのにR1@0.7低い → 1秒前後の微ズレ(後処理で救える)。")


if __name__ == '__main__':
    main()
