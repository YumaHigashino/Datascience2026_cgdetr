"""
音響変化点が境界の手がかりになるか診断。

CLAP frame feature (T,768) の隣接フレーム差分(cos距離/L2距離)が、
GT境界(区間の開始秒/終了秒)で大きいかを測る。

  境界>非境界の割合, 境界diffの音声内パーセンタイル平均
  → 65%超なら「音響変化点は境界を示す」=CNN主入力として有望
  → 50%付近なら弱い。saliency中心 or 設計見直し

サーバー実行(特徴は features/castella/clap)。GPU不要。

Usage:
    .venv/bin/python src/diagnose_audio_boundary.py \
        --gt data/castella_test_release.jsonl \
        --feat_dir features/castella/clap
"""

import numpy as np
import json
import random
import argparse


def diffs(feat_dir, vid):
    a = np.load(f"{feat_dir}/{vid}.npz")["features"]  # (T,768)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    cosd = 1.0 - np.sum(an[1:] * an[:-1], axis=1)      # (T-1,) 隣接コサイン距離
    l2d = np.linalg.norm(a[1:] - a[:-1], axis=1)        # (T-1,) 隣接L2距離
    return cosd, l2d


def run(name, GT, feat_dir, pick, seed=0):
    random.seed(seed)
    bvals, rvals, pcts, nb = [], [], [], 0
    for d in GT:
        qid = d["qid"]
        vid = d.get("vid") or qid.rsplit("_", 1)[0]
        try:
            cosd, l2d = diffs(feat_dir, vid)
        except Exception:
            continue
        dd = pick(cosd, l2d)
        T = len(dd) + 1
        if T < 5:
            continue
        bps = {p for w in d["relevant_windows"]
               for p in [int(round(w[0])), int(round(w[1]))] if 1 <= p <= T - 1}
        if not bps:
            continue
        nonb = [i for i in range(1, T) if i not in bps]
        if not nonb:
            continue
        for b in bps:
            nb += 1
            val = dd[b - 1]                       # 位置bの変化は dd[b-1]
            bvals.append(val)
            pcts.append(100.0 * np.mean(dd <= val))
            rvals.append(dd[random.choice(nonb) - 1])

    bvals, rvals, pcts = map(np.array, (bvals, rvals, pcts))
    print(f"[{name}] 境界点={nb}  境界diff平均={bvals.mean():.4f}  非境界diff平均={rvals.mean():.4f}")
    print(f"         境界>非境界={100*np.mean(bvals>rvals):.1f}%  "
          f"音声内パーセンタイル平均={pcts.mean():.1f}%  (50%=無相関, 高いほど有望)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--feat_dir", required=True)
    args = parser.parse_args()
    GT = [json.loads(l) for l in open(args.gt) if l.strip()]
    print(f"GT {len(GT)} 件\n")
    run("cos距離", GT, args.feat_dir, lambda c, l: c)
    run("L2距離", GT, args.feat_dir, lambda c, l: l)


if __name__ == "__main__":
    main()
