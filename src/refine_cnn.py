"""
1D-CNN 境界補正モデル。

入力  (B, 2, L) : ch0=音響変化点, ch1=擬似境界マスク
出力  (B, 2, L) : ch0=start度, ch1=end度 のヒートマップ(logits)

セグメンテーション型(入力長=出力長)。各秒が境界らしいかを畳み込みで判定する。
D班のDynamic Refinement Head(座標をMLPで回帰)とは、アーキ(CNN)・入力(時系列)・
出力(ヒートマップ)の3点で別物。

推論: ヒートマップのargmax位置を窓座標→絶対秒に戻す。「どのピークが境界か」を
学習で選べるので、単純スナップ(最大ピーク吸着で-2.3悪化)の誤吸着を避けられる。
"""

import torch
import torch.nn as nn


class RefineCNN(nn.Module):
    def __init__(self, in_ch=2, hidden=64, n_blocks=4, k=5):
        super().__init__()
        pad = k // 2
        self.inp = nn.Sequential(
            nn.Conv1d(in_ch, hidden, k, padding=pad),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, k, padding=pad),
                nn.BatchNorm1d(hidden), nn.ReLU(),
            ) for _ in range(n_blocks)
        ])
        self.head = nn.Conv1d(hidden, 2, 1)

    def forward(self, x):           # x: (B, 2, L)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)            # 残差接続で深くしても安定
        return self.head(h)         # (B, 2, L) logits


def decode(logits, win_start):
    """ヒートマップ logits (2, L) → 絶対秒 (s, e)。argmaxで境界位置を取る。"""
    prob = torch.sigmoid(logits)
    s_idx = int(prob[0].argmax().item())
    e_idx = int(prob[1].argmax().item())
    s = win_start + s_idx
    e = win_start + e_idx
    if s >= e:                      # 破綻時は最小幅1秒を確保
        e = s + 1
    return float(s), float(e)


if __name__ == "__main__":
    # スモークテスト: ランダム入力で forward と損失が通るか
    B, L = 8, 61
    model = RefineCNN()
    n_param = sum(p.numel() for p in model.parameters())
    x = torch.randn(B, 2, L)
    y = torch.rand(B, 2, L)         # ガウシアンソフトラベル相当(0-1)
    out = model(x)
    loss = nn.BCEWithLogitsLoss()(out, y)
    loss.backward()
    print(f"パラメータ数: {n_param:,}")
    print(f"入力 {tuple(x.shape)} → 出力 {tuple(out.shape)}  (一致すべき)")
    print(f"BCE損失: {loss.item():.4f}  / backward OK")
    s, e = decode(out[0].detach(), win_start=40)
    print(f"decode例: s={s}, e={e}")
