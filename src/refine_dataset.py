"""
1D-CNN境界補正モデルの学習データ生成 (PyTorch Dataset)。

各正解区間[s,e]にノイズを注入して「擬似的な1段目予測」を作り、その中心±W秒を
クロップする。推論時に来る「ズレた予測」を学習時に再現する(train-test mismatch対策)。

  入力 X = (2ch, L) の時系列
    ch0: 音響変化点 = CLAP frame featureの隣接cos距離 (境界を70%で指す主信号)
    ch1: 擬似境界マスク = 擬似予測のstart/end位置のガウシアン (元予測位置をCNNに伝え、
         飛びすぎを抑制させる)
  ターゲット Y = (2ch, L) のヒートマップ
    ch0: GT start のガウシアン, ch1: GT end のガウシアン

毎 __getitem__ で違うノイズを引く(データ拡張)。train=False でノイズ無し(デバッグ用)。
推論時はノイズ注入でなく実際の1段目予測を中心にcrop する(refine_infer.py 側で処理)。
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


def gaussian(length, mu, sigma):
    x = np.arange(length, dtype=np.float32)
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


class RefineDataset(Dataset):
    def __init__(self, gt_path, feat_dir, W=30, sigma=1.5,
                 center_noise=5.0, center_clip=12.0,
                 len_scale_lo=0.5, len_scale_hi=2.0,
                 max_gt_len=36.0, train=True, seed=0):
        self.feat_dir = feat_dir
        self.W = W
        self.L = 2 * W + 1
        self.sigma = sigma
        self.center_noise = center_noise
        self.center_clip = center_clip
        self.log_lo = np.log(len_scale_lo)
        self.log_hi = np.log(len_scale_hi)
        self.max_gt_len = max_gt_len
        self.train = train
        self.rng = np.random.RandomState(seed)
        self._cache = {}

        recs = [json.loads(l) for l in open(gt_path, encoding="utf-8") if l.strip()]
        # 1正解区間 = 1学習サンプル（複数GTは個別に展開）。
        # 超長区間(>max_gt_len)は ±W 窓に端点が収まらず補正の対象外なので除外。
        self.items = []
        self.n_excluded = 0
        for d in recs:
            vid = d.get("vid") or d["qid"].rsplit("_", 1)[0]
            for w in d["relevant_windows"]:
                s, e = float(w[0]), float(w[1])
                if e - s > self.max_gt_len:
                    self.n_excluded += 1
                    continue
                self.items.append((vid, s, e))

    def _cos_diff(self, vid):
        if vid not in self._cache:
            a = np.load(f"{self.feat_dir}/{vid}.npz")["features"].astype(np.float32)
            an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            self._cache[vid] = (1.0 - np.sum(an[1:] * an[:-1], axis=1)).astype(np.float32)
        return self._cache[vid]  # (T-1,)  d[i] は時刻 i→i+1 の変化

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid, s, e = self.items[idx]
        d = self._cos_diff(vid)
        T = len(d) + 1
        c = 0.5 * (s + e)
        L = max(e - s, 1.0)

        # --- ノイズ注入で擬似予測を作る ---
        if self.train:
            cn = float(np.clip(self.rng.randn() * self.center_noise,
                               -self.center_clip, self.center_clip))
            sc = float(np.exp(self.rng.uniform(self.log_lo, self.log_hi)))
        else:
            cn, sc = 0.0, 1.0
        cp = c + cn
        Lp = L * sc
        sp, ep = cp - 0.5 * Lp, cp + 0.5 * Lp

        # --- 擬似予測中心 ±W をクロップ ---
        center = int(round(cp))
        win_start = center - self.W
        Lw = self.L

        # ch0: 音響変化点（位置 p の手がかりは d[p-1]、窓外は0）
        ch_audio = np.zeros(Lw, dtype=np.float32)
        for k in range(Lw):
            p = win_start + k
            if 1 <= p <= T - 1:
                ch_audio[k] = d[p - 1]

        # ch1: 擬似境界マスク（sp, ep をガウシアンで）
        ch_pred = gaussian(Lw, sp - win_start, self.sigma) + gaussian(Lw, ep - win_start, self.sigma)

        X = np.stack([ch_audio, ch_pred], axis=0)  # (2, Lw)

        # ターゲット: GT境界 start/end のヒートマップ（窓外に出たら全0=学習対象外気味）
        y_start = gaussian(Lw, s - win_start, self.sigma)
        y_end = gaussian(Lw, e - win_start, self.sigma)
        Y = np.stack([y_start, y_end], axis=0)  # (2, Lw)

        meta = {"vid": vid, "s": float(s), "e": float(e), "win_start": int(win_start)}
        return torch.from_numpy(X), torch.from_numpy(Y), meta


if __name__ == "__main__":
    # スモークテスト（サーバーで: .venv/bin/python src/refine_dataset.py）
    import sys
    gt = sys.argv[1] if len(sys.argv) > 1 else "data/castella_train_release.jsonl"
    fd = sys.argv[2] if len(sys.argv) > 2 else "features/castella/clap"
    ds = RefineDataset(gt, fd, train=True)
    print(f"サンプル数(展開後の正解区間数): {len(ds)}")
    X, Y, m = ds[0]
    print(f"X shape={tuple(X.shape)} (2ch, L)  Y shape={tuple(Y.shape)}")
    print(f"ch0(音響変化点) min/max: {X[0].min():.3f}/{X[0].max():.3f}")
    print(f"ch1(擬似境界) ピーク数: {(X[1] > 0.5).sum().item()}")
    print(f"Y start argmax={Y[0].argmax().item()}  end argmax={Y[1].argmax().item()}  "
          f"(GT s={m['s']}, e={m['e']}, win_start={m['win_start']})")
