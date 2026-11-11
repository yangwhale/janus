"""mini 隔离实验：摘掉 MTP loss / 索引器 KD loss（各自系数设 0）后，TPU 主 loss 对 GPU 的偏差后段还往上走不往上走。

每种配置（含现行对照）给一组 GPU 运行与一组 TPU 运行（同一配置、同一批数据），算：
- 逐步相对差 (TPU − GPU 均值) ÷ GPU 均值，按 25 步分块；第 551–1000 步各块对块中心步数的最小二乘斜率（每 100 步），
  以及 TPU 每份各自的斜率；GPU 留一法（每份 GPU 对其余几份的均值）同样算一遍，当「GPU 自己会走出多大斜率」的尺子；
- 窗口均值：第 2–500、501–1000、551–1000、751–1000 步；
- 带符号均值的 r（janus L4 的定义，PROTOCOL 第 3 节）：被测对基线（GPU 组第一份）在第 1–1000 步的带符号均值，
  除以 GPU 组两两各对的带符号均值的最大绝对值（份数少时尺子粗：4 份只有 6 对）。

    python mini_loss_isolation.py --root <运行目录根> --config 名字:GPU1,GPU2,…:TPU1,TPU2 [...] [--field lm_loss] --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

BLOCK = 25
STEPS = 1000
WINDOWS = [(2, 500), (501, 1000), (551, 1000), (751, 1000)]


def series(root: str, run: str, field: str) -> np.ndarray:
    d = {}
    with open(os.path.join(root, run, "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get(field) is not None:
                d[r["step"]] = r[field]
    return np.array([d[s] for s in range(1, STEPS + 1)], float)


def blocks(rel: np.ndarray) -> np.ndarray:
    return rel[..., : STEPS // BLOCK * BLOCK].reshape(rel.shape[:-1] + (STEPS // BLOCK, BLOCK)).mean(-1)


def late_slope(b: np.ndarray, first: int = 551) -> float:
    """第 first 步以后各块（块中心步数）对步数的最小二乘斜率，单位：每 100 步。"""
    centers = np.arange(STEPS // BLOCK) * BLOCK + (BLOCK + 1) / 2
    m = centers >= first
    return float(np.polyfit(centers[m], b[m], 1)[0] * 100)


def signed_mean(test: np.ndarray, golden: np.ndarray) -> float:
    return float(np.mean((test - golden) / golden))


def one(root: str, name: str, gpu: list[str], tpu: list[str], field: str) -> dict:
    g = np.array([series(root, x, field) for x in gpu])
    t = np.array([series(root, x, field) for x in tpu])
    gm = g.mean(0)
    rel_t = (t - gm) / gm
    rel_loo = np.array([(g[i] - np.delete(g, i, 0).mean(0)) / gm for i in range(len(gpu))]) if len(gpu) > 1 else None
    bt = blocks(rel_t)
    bl = blocks(rel_loo) if rel_loo is not None else None
    pairs = [abs(signed_mean(g[i], g[j])) for i, j in itertools.combinations(range(len(gpu)), 2)]
    n = max(pairs) if pairs else None
    out = {
        "gpu": gpu,
        "tpu": tpu,
        "blocks_tpu_mean": [float(x) for x in bt.mean(0)],
        "late_slope_per100": {
            "tpu_mean": late_slope(bt.mean(0)),
            "tpu_each": [late_slope(x) for x in bt],
            "gpu_loo_each": [late_slope(x) for x in bl] if bl is not None else None,
        },
        "windows": {},
        "signed_mean_r": {
            "golden": gpu[0],
            "N_max_pair": n,
            "n_pairs": len(pairs),
            "tpu_each": [signed_mean(x, g[0]) / n for x in t] if n else None,
        },
    }
    for a, b in WINDOWS:
        out["windows"][f"{a}-{b}"] = {
            "tpu_each": [float(x) for x in rel_t[:, a - 1 : b].mean(-1)],
            "tpu_mean": float(rel_t[:, a - 1 : b].mean()),
            "gpu_loo_sd": float(np.std(rel_loo[:, a - 1 : b].mean(-1), ddof=1)) if rel_loo is not None else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", nargs="+", required=True, help="名字:GPU 运行（逗号分隔）:TPU 运行（逗号分隔）")
    ap.add_argument("--field", nargs="+", default=["lm_loss"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"what": "隔离实验：各配置 TPU 对同配置 GPU 均值的主 loss 偏差（25 步分块、后段斜率、窗口、带符号均值 r）", "configs": {}}
    for spec in a.config:
        name, gs, ts = spec.split(":")
        res["configs"][name] = {}
        for f in a.field:
            try:
                o = one(a.root, name, gs.split(","), ts.split(","), f)
            except KeyError:  # 这一项在某份运行里没记（如关 KD 后的 indexer_loss）
                continue
            if not np.all(np.isfinite(o["blocks_tpu_mean"])):  # 系数为 0 时记录值为 0（如关 MTP 后的 mtp_loss_mt）
                continue
            res["configs"][name][f] = o
            s = o["late_slope_per100"]
            w = o["windows"]["551-1000"]
            loo = s["gpu_loo_each"]
            rr = o["signed_mean_r"]["tpu_each"]
            print(
                f"{name:8s} {f:12s} slope/100 TPU {s['tpu_mean']:+.2e} (each {', '.join(f'{x:+.1e}' for x in s['tpu_each'])})"
                + (f" GPU-LOO [{min(loo):+.1e}, {max(loo):+.1e}]" if loo else "")
                + f" | 551-1000 {w['tpu_mean']:+.2e}" + (f" (sd {w['gpu_loo_sd']:.1e})" if w["gpu_loo_sd"] else "")
                + (f" | r {', '.join(f'{x:.2f}' for x in rr)} ({o['signed_mean_r']['n_pairs']} 对)" if rr else "")
            )
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
