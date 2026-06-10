"""
1D-CNN境界補正モデルの学習。

学習・検証とも RefineDataset(train=True) でノイズ注入した擬似予測を入力する
(推論時の1段目予測のズレを再現)。val=ノイズ無しにすると擬似境界マスクのコピーで
当たってしまい能力を測れないため、val もノイズありで評価する。

best/last の両方を保存し、最終判断は test submission を refine_infer で補正した
R1@0.7 で行う(val=352 は小さく当てにならない、という既知の知見に従う)。

Usage (サーバー):
    .venv/bin/python src/refine_train.py \
        --train_gt data/castella_train_release.jsonl \
        --val_gt   data/castella_val_release.jsonl \
        --feat_dir features/castella/clap \
        --out results_refine
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from refine_dataset import RefineDataset
from refine_cnn import RefineCNN


def collate(batch):
    X = torch.stack([b[0] for b in batch])
    Y = torch.stack([b[1] for b in batch])
    return X, Y


def run_epoch(model, loader, crit, opt, dev, train):
    model.train(train)
    tot, n = 0.0, 0
    for X, Y in loader:
        X, Y = X.to(dev), Y.to(dev)
        with torch.set_grad_enabled(train):
            out = model(X)
            loss = crit(out, Y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        tot += loss.item() * len(X)
        n += len(X)
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_gt", required=True)
    p.add_argument("--val_gt", required=True)
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--out", default="results_refine")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bsz", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--W", type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    tr = RefineDataset(args.train_gt, args.feat_dir, W=args.W, train=True, seed=0)
    va = RefineDataset(args.val_gt, args.feat_dir, W=args.W, train=True, seed=1)
    print(f"train={len(tr)} (除外{tr.n_excluded})  val={len(va)} (除外{va.n_excluded})")

    trl = DataLoader(tr, batch_size=args.bsz, shuffle=True, collate_fn=collate)
    val = DataLoader(va, batch_size=args.bsz, shuffle=False, collate_fn=collate)

    model = RefineCNN().to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 1e9
    for ep in range(args.epochs):
        tr_loss = run_epoch(model, trl, crit, opt, dev, True)
        va_loss = run_epoch(model, val, crit, opt, dev, False)
        torch.save(model.state_dict(), f"{args.out}/last.pt")
        flag = ""
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), f"{args.out}/best.pt")
            flag = " *best"
        print(f"ep{ep:02d}  train={tr_loss:.4f}  val={va_loss:.4f}{flag}")

    print(f"done. best val loss={best:.4f}  -> {args.out}/best.pt, last.pt")


if __name__ == "__main__":
    main()
