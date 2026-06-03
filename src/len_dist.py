#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学習/評価データのモーメント長分布を比較する。

短モーメント(<=3s)が学習データにどれだけ含まれるかを確認し、CASTELLA test の
長さ分布(<=3s が 36%)と突き合わせる。学習データに短モーメントが乏しければ
「モデルが短モーメントを学べていない」ことが原因として裏付けられる。

  python len_dist.py \
      data/clotho_moment_train_release.jsonl \
      data/castella_train_release.jsonl \
      data/castella_test_release.jsonl
"""
import json
import sys
import numpy as np

BINS = [(0, 3), (3, 10), (10, 30), (30, 1e9)]
LABELS = ['<=3s', '3-10s', '10-30s', '>30s']


def main():
    header = (f"{'file':>40} | {'n_win':>7} | " +
              " | ".join(f"{lab:>11}" for lab in LABELS) +
              f" | {'med':>5} | {'mean':>5}")
    print(header)
    print("-" * len(header))
    for path in sys.argv[1:]:
        lens = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    for w in d.get('relevant_windows', []):
                        lens.append(w[1] - w[0])
        except FileNotFoundError:
            print(f"{path.split('/')[-1]:>40} | (not found)")
            continue
        lens = np.array(lens, dtype=float)
        if len(lens) == 0:
            print(f"{path.split('/')[-1]:>40} | (no windows)")
            continue
        cells = []
        for lo, hi in BINS:
            m = (lens > lo) & (lens <= hi)
            cells.append(f"{m.sum():>5}({m.mean() * 100:>4.1f}%)")
        print(f"{path.split('/')[-1]:>40} | {len(lens):>7} | " +
              " | ".join(f"{c:>11}" for c in cells) +
              f" | {np.median(lens):>5.1f} | {lens.mean():>5.1f}")


if __name__ == '__main__':
    main()
