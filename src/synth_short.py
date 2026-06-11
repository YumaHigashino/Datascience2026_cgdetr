"""
短モーメントの「合成」オーバーサンプリング（cut-paste at feature level）。

前回の複製(short_oversample)は同じ3点セットを参照する行を増やすだけで新情報ゼロ→
短バイアスで全体悪化(-1.63)。本スクリプトは新しい3点セットを物理生成する:

  ソース = <=thresh 秒の relevant_window を持つ train サンプルの、その短区間の特徴スライス
  背景   = 別サンプルの音声特徴(全長)。背景のGT区間と重複しない位置にスライスを上書き
  →  新vid特徴(npz) + 新jsonl行 + テキスト特徴(ソースのをリンク流用)

背景 × 短区間 × 挿入位置 の組合せで多様性を作る(複製と違い新情報がある)。
元の共有特徴(シンボリックリンク先 /data/kokuryumaru/...)には一切書き込まない。
合成専用dirに「元npzへのsymlink + 合成npz」を置き、a_feat_dir を差し替えて使う。

Usage (サーバー, cwd=src):
    ../.venv/bin/python synth_short.py \
        --train_gt ../data/castella_train_release.jsonl \
        --a_feat_dir ../features/castella/clap \
        --t_feat_dir ../features/castella/clap_text \
        --out_jsonl ../data/castella_train_synth.jsonl \
        --out_a_dir ../features/castella/clap_aug \
        --out_t_dir ../features/castella/clap_text_aug \
        --thresh 3.0 --mult 5 --margin 2 --seed 0

スモーク確認(少数だけ):  上記に --limit_sources 20 を足す
"""

import argparse
import json
import os
import numpy as np


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def feat_path(d, vid):
    return os.path.join(d, f"{vid}.npz")


def load_feat(d, vid):
    return np.load(feat_path(d, vid))["features"].astype(np.float32)


def overlaps(p, seglen, windows, margin):
    """[p, p+seglen] が windows のどれかと ±margin で重なるか。"""
    a0, a1 = p - margin, p + seglen + margin
    for w in windows:
        if a0 < w[1] and w[0] < a1:
            return True
    return False


def link_originals(recs, src_a_dir, src_t_dir, out_a_dir, out_t_dir):
    """元の全 train が参照する音声/テキスト npz への symlink を out dir に張る。
    合成vidと元vidを単一 a_feat_dir で引けるようにする(dataset.py は単一dir前提)。"""
    n_a, n_t = 0, 0
    for d in recs:
        vid = d["vid"]
        src = os.path.abspath(feat_path(src_a_dir, vid))
        dst = feat_path(out_a_dir, vid)
        if not os.path.exists(dst) and os.path.exists(src):
            os.symlink(src, dst); n_a += 1
        qid = d["qid"]
        ts = os.path.abspath(os.path.join(src_t_dir, f"qid{qid}.npz"))
        td = os.path.join(out_t_dir, f"qid{qid}.npz")
        if not os.path.exists(td) and os.path.exists(ts):
            os.symlink(ts, td); n_t += 1
    print(f"[link] 元npzへのsymlink: audio +{n_a}, text +{n_t}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_gt", required=True)
    p.add_argument("--a_feat_dir", required=True)
    p.add_argument("--t_feat_dir", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--out_a_dir", required=True)
    p.add_argument("--out_t_dir", required=True)
    p.add_argument("--thresh", type=float, default=3.0, help="短区間とみなす秒数")
    p.add_argument("--mult", type=int, default=5, help="1ソースあたり何個の背景に貼るか")
    p.add_argument("--margin", type=int, default=2, help="背景GT区間を避けるマージン秒")
    p.add_argument("--max_tries", type=int, default=30, help="挿入位置の試行回数上限")
    p.add_argument("--min_seg", type=int, default=1, help="この秒未満の超短区間は除外")
    p.add_argument("--limit_sources", type=int, default=0, help=">0 でソース数を制限(スモーク用)")
    p.add_argument("--qid_base", type=int, default=9000000, help="新qidの開始番号(既存と衝突しない大きい値)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_a_dir, exist_ok=True)
    os.makedirs(args.out_t_dir, exist_ok=True)

    recs = load_jsonl(args.train_gt)
    print(f"train samples: {len(recs)}")

    # --- ソース: <=thresh 秒の window を持つサンプル(その window) ---
    sources = []  # (rec, window)
    for d in recs:
        for w in d.get("relevant_windows", []):
            seglen = int(round(w[1])) - int(round(w[0]))
            if 0 < (w[1] - w[0]) <= args.thresh and seglen >= args.min_seg:
                sources.append((d, [int(round(w[0])), int(round(w[1]))]))
    print(f"short sources (<= {args.thresh}s, seg>={args.min_seg}s): {len(sources)}")
    if args.limit_sources > 0:
        sources = sources[: args.limit_sources]
        print(f"  -> limited to {len(sources)} (smoke)")

    # 背景候補(全 train。十分長いものを優先するため duration でソートはしない=多様性重視)
    bg_pool = recs

    new_recs = []
    qid = args.qid_base
    n_made, n_skip = 0, 0
    for (src, win) in sources:
        try:
            src_feat = load_feat(args.a_feat_dir, src["vid"])
        except Exception:
            n_skip += args.mult
            continue
        si, ei = win[0], min(win[1], len(src_feat))
        seg = src_feat[si:ei]
        seglen = len(seg)
        if seglen < args.min_seg:
            n_skip += args.mult
            continue

        made_for_src = 0
        tries_bg = 0
        while made_for_src < args.mult and tries_bg < args.mult * 4:
            tries_bg += 1
            bg = bg_pool[rng.randint(len(bg_pool))]
            if bg["vid"] == src["vid"]:
                continue
            try:
                bg_feat = load_feat(args.a_feat_dir, bg["vid"])
            except Exception:
                continue
            T = len(bg_feat)
            if T <= seglen + 2 * args.margin:
                continue
            bg_wins = bg.get("relevant_windows", [])
            # 重複しない挿入位置 p を探す
            p = None
            for _ in range(args.max_tries):
                cand = int(rng.randint(0, T - seglen))
                if not overlaps(cand, seglen, bg_wins, args.margin):
                    p = cand
                    break
            if p is None:
                continue

            new_feat = bg_feat.copy()
            new_feat[p:p + seglen] = seg
            new_vid = f"synth{qid}"
            np.savez(feat_path(args.out_a_dir, new_vid), features=new_feat)

            # テキスト特徴はソースの query と同一 → 元 qid の npz を新 qid 名で symlink
            ts = os.path.abspath(os.path.join(args.t_feat_dir, f"qid{src['qid']}.npz"))
            td = os.path.join(args.out_t_dir, f"qid{qid}.npz")
            if not os.path.exists(ts):
                # テキスト特徴が無いソースは破棄(音声npzも消す)
                os.remove(feat_path(args.out_a_dir, new_vid))
                continue
            if not os.path.exists(td):
                os.symlink(ts, td)

            new_recs.append({
                "qid": qid,
                "query": src["query"],
                "duration": int(T),
                "vid": new_vid,
                "relevant_windows": [[p, p + seglen]],
            })
            qid += 1
            n_made += 1
            made_for_src += 1
        if made_for_src == 0:
            n_skip += args.mult

    print(f"synthesized: {n_made}  (skipped ~{n_skip})")

    # 元npzへのsymlinkを張り、合成vid/qidと同一dirで引けるようにする
    link_originals(recs, args.a_feat_dir, args.t_feat_dir, args.out_a_dir, args.out_t_dir)

    # 出力jsonl = 元train + 合成
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for d in recs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        for d in new_recs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {args.out_jsonl}: {len(recs)} + {len(new_recs)} = {len(recs)+len(new_recs)} lines")

    # 長さ帯の確認(合成後に短区間がどれだけ増えたか)
    def band_count(rs):
        b = {"<=3s": 0, "3-10s": 0, ">10s": 0}
        for d in rs:
            for w in d["relevant_windows"]:
                L = w[1] - w[0]
                b["<=3s" if L <= 3 else ("3-10s" if L <= 10 else ">10s")] += 1
        return b
    print(f"  band before: {band_count(recs)}")
    print(f"  band after : {band_count(recs + new_recs)}")


if __name__ == "__main__":
    main()
