"""mini 同一份权重上的前向对照：TPU 冻结前向、CPU FP64 参照、CPU BF16（GPU 写法）、真 GPU 重跑，主 loss 与 MTP loss 各对 FP64。

为什么：主 loss 的 TPU 偏高要先分开「每一步都有的前向差」与「训练轨迹的差」。同一份权重上让几方各算同几批的前向，
差别就只剩前向（同 06-rw3 第 151 / 900 步的做法）。

输入（每个状态一组，`--state 名字=<TPU 冻结运行目录>,<FP64 目录>,<CPU BF16 目录>`，后两个目录里每批一个 s<步>.log）：
- TPU：`train_hy4.py --param-override <HF 快照> --bias-override hf:param-override --first-step N`，`learning_rate=0
  routed_bias_update_rate=0`，metrics.jsonl 的 lm_loss / mtp_loss_mt（param_norm 不变）；
- FP64：`frameworks/megatron/step1_ref.py --forward-only --step N`（GPU 同一套代码，ForceDtype 改写成 FP64）的 MICRO 行；
- CPU BF16：`cpu_bf16_gpu_forward.py`（同一套代码、local 层、BF16 参数 + Float16Module），同格式的 MICRO 行；
- 真 GPU（可选）：`--gpu 名字:步=lm,mtp_mt[;lm,mtp_mt]`，同状态重跑的第 1 步记录。
MTP 一律按 MaxText 口径（M22：每个 microbatch 内按主 loss 有效 token 数加权的行均值，再对 microbatch 平均），
FP64 / CPU BF16 从 MICRO 行的 mtp_mean 与 n_tokens 现算；TPU 的 mtp_loss_mt 乘 10（系数 0.1）。

    python mini_main_loss_same_state.py --state s751=<tpu>,<fp64>,<bf16> ... [--gpu g900:900=6.5885,0.7515] --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def micro_global(log: str) -> tuple[float, float] | None:
    """MICRO 行 → (主 loss = Σ loss_sum ÷ Σ n_tokens, MTP（MaxText 口径）)；没写完（无 DONE）返回 None。"""
    text = open(log).read()
    if "DONE" not in text:
        return None
    rows = sorted((json.loads(x[6:]) for x in text.splitlines() if x.startswith("MICRO ")), key=lambda r: r["row"])
    lm = sum(r["loss_sum"] for r in rows) / sum(r["n_tokens"] for r in rows)
    mbs = [rows[m * 8 : (m + 1) * 8] for m in range(len(rows) // 8)]   # 第 m 个 microbatch = 全局批第 8m … 8m+7 行
    mt = float(np.mean([sum(r["mtp_mean"] * r["n_tokens"] for r in rr) / sum(r["n_tokens"] for r in rr) for rr in mbs]))
    return lm, mt


def mean_se(v: list[float]) -> dict | None:
    if not v:
        return None
    return {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / len(v) ** 0.5) if len(v) > 1 else None, "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs="+", required=True)
    ap.add_argument("--gpu", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gpu = {}
    for g in a.gpu:
        key, vals = g.split("=")
        st, step = key.split(":")
        gpu.setdefault(st, {})[int(step)] = [tuple(map(float, p.split(","))) for p in vals.split(";")]
    out = {"what": "同一份权重上各方前向对 CPU FP64 的相对差（主 loss、MTP loss MaxText 口径）", "states": {}}
    for spec in a.state:
        name, paths = spec.split("=")
        tpu_dir, f64_dir, b16_dir = paths.split(",")
        tpu = {json.loads(x)["step"]: json.loads(x) for x in open(os.path.join(tpu_dir, "metrics.jsonl"))}
        rows = []
        for log in sorted(glob.glob(os.path.join(f64_dir, "s*.log"))):
            step = int(os.path.basename(log)[1:-4])
            ref = micro_global(log)
            if ref is None:
                continue
            r = {"batch": step, "fp64_lm": ref[0], "fp64_mtp": ref[1]}
            if step in tpu:
                r["tpu_lm"] = (tpu[step]["lm_loss"] - ref[0]) / ref[0]
                r["tpu_mtp"] = (tpu[step]["mtp_loss_mt"] * 10 - ref[1]) / ref[1]
            b16 = os.path.join(b16_dir, f"s{step}.log")
            if os.path.exists(b16) and micro_global(b16) is not None:
                x = micro_global(b16)
                r["bf16_lm"] = (x[0] - ref[0]) / ref[0]
                r["bf16_mtp"] = (x[1] - ref[1]) / ref[1]
            if step in gpu.get(name, {}):
                r["gpu_lm"] = [(x - ref[0]) / ref[0] for x, _ in gpu[name][step]]
                r["gpu_mtp"] = [(y * 10 - ref[1]) / ref[1] for _, y in gpu[name][step]]
            rows.append(r)
        summ = {}
        for k in ("tpu_lm", "bf16_lm", "gpu_lm", "tpu_mtp", "bf16_mtp", "gpu_mtp"):
            summ[k] = mean_se([float(np.mean(x[k])) for x in rows if k in x])
        both = [x for x in rows if "tpu_lm" in x and "bf16_lm" in x]
        summ["tpu_minus_bf16_lm"] = mean_se([x["tpu_lm"] - x["bf16_lm"] for x in both])
        summ["tpu_minus_bf16_mtp"] = mean_se([x["tpu_mtp"] - x["bf16_mtp"] for x in both])
        out["states"][name] = {"rows": rows, "summary": summ}
        print(name, {k: (f"{v['mean']:+.2e} n{v['n']}" if v else None) for k, v in summ.items()})
    allb = [x for s in out["states"].values() for x in s["rows"] if "tpu_lm" in x and "bf16_lm" in x]
    out["all_states"] = {
        "tpu_minus_bf16_lm": mean_se([x["tpu_lm"] - x["bf16_lm"] for x in allb]),
        "tpu_minus_bf16_mtp": mean_se([x["tpu_mtp"] - x["bf16_mtp"] for x in allb]),
        "tpu_lm": mean_se([x["tpu_lm"] for s in out["states"].values() for x in s["rows"] if "tpu_lm" in x]),
        "bf16_lm": mean_se([x["bf16_lm"] for s in out["states"].values() for x in s["rows"] if "bf16_lm" in x]),
    }
    print("all", out["all_states"])
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
