"""同一份权重上逐张量梯度对 CPU FP64 的误差，按参数类别汇总，分母可以是几份「GPU 写法」样本。

为什么：同状态（如 GPU g3x-a 第 849 步）上只有一份真 GPU 梯度，单份样本当 R 的分母会把它自己的运气算进来（某类参数
在这一份上恰好离 FP64 近，R 就虚高）。所以另用 `cpu_bf16_gpu_forward.py --all-grads`（GPU 同一套代码、BF16、每行
落地一次再 FP32 累加）多出几份 GPU 写法的样本，每个被测对每一份分母各算一遍 R。

类别（HF 名字）：
- ihc：iHC 与 hc_head 的门控参数（`hc_pre.hc_fn / hc_base / hc_scale`、`hc_head.*`）；
- sink：注意力 sink（`learnable_sink_param`）；
- norm：RMSNorm 的权重（各层 / MTP 的 layernorm、enorm、hnorm、`model.norm`，索引器 k_norm 的权重）；
- rest：其余（路由专家不算：TPU 诊断梯度不存它们）。

逐张量 rel = ‖x − FP64‖ / ‖FP64‖；R(x 对 g) = rel(x) / rel(g)。每类报 rel 中位数、R 中位数、R > 1.2 的个数。
TPU 诊断梯度（`--diag-step1`）自动判断要不要乘全局有效 token 数（同 `same_state_grads.py`）。

    python same_state_grad_classes.py --ref <step1_ref.py 输出目录> --gpu g3x-a=<safetensors> proxy=<目录或文件> \\
        --x base=<TPU 运行目录> ihc=<TPU 运行目录> … --out <json>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from safetensors import safe_open

CLASSES = ("ihc", "sink", "norm", "rest")


def class_of(h: str) -> str:
    if ".hc_pre." in h or h.startswith("model.hc_head."):
        return "ihc"
    if "learnable_sink" in h:
        return "sink"
    if h.endswith("norm.weight"):
        return "norm"
    return "rest"


def open_grads(path: str):
    """运行目录（diag_grads.safetensors / grads_hf.safetensors）或 safetensors 文件。"""
    if os.path.isdir(path):
        for f in ("diag_grads.safetensors", "grads_hf.safetensors"):
            if os.path.exists(os.path.join(path, f)):
                return safe_open(os.path.join(path, f), "pt")
        raise FileNotFoundError(path)
    return safe_open(path, "pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--gpu", nargs="+", required=True, help="名字=GPU 写法样本（真 GPU 或 CPU BF16 代理）")
    ap.add_argument("--x", nargs="+", required=True, help="名字=被测（TPU 运行目录或文件）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = safe_open(os.path.join(a.ref, "grads_hf.safetensors"), "pt")
    gpu = {k: open_grads(v) for k, v in (s.split("=", 1) for s in a.gpu)}
    xs = {k: open_grads(v) for k, v in (s.split("=", 1) for s in a.x)}
    every = {**gpu, **xs}
    names = sorted(
        n for n in ref.keys() if ".experts." not in n and all(n in f.keys() for f in every.values())
    )
    refn = {n: ref.get_tensor(n).double() for n in names}
    # 缩放：被测的总范数与 FP64 的比若接近全局 token 数，说明是没除 token 数的诊断梯度
    summ = json.load(open(os.path.join(a.ref, "summary.json")))
    ntok = float(summ.get("global_tokens") or summ.get("n_tokens") or 0)
    rsq = sum(float(t.square().sum()) for t in refn.values())
    scale = {}
    for k, f in every.items():
        sq = sum(float(f.get_tensor(n).double().square().sum()) for n in names)
        ratio = (sq / rsq) ** 0.5
        scale[k] = 1.0 / ntok if ntok and abs(ratio / ntok - 1) < 0.2 else 1.0
    rows = []
    for n in names:
        r = refn[n].flatten()
        nr = float(r.norm())
        row = {"tensor": n, "class": class_of(n), "ref_norm": nr}
        for k, f in every.items():
            x = f.get_tensor(n).double().flatten() * scale[k]
            row[f"rel_{k}"] = float((x - r).norm()) / nr
        rows.append(row)
    summary = {}
    for c in CLASSES:
        rs = [r for r in rows if r["class"] == c]
        s = {"n": len(rs), "rel_median": {k: float(np.median([r[f"rel_{k}"] for r in rs])) for k in every}}
        for k in every:
            for g in gpu:
                if k == g:
                    continue
                R = [r[f"rel_{k}"] / r[f"rel_{g}"] for r in rs]
                s[f"R_{k}_over_{g}"] = {"median": float(np.median(R)), "n_gt_1.2": int(sum(x > 1.2 for x in R))}
        summary[c] = s
    out = {
        "what": "同状态逐张量梯度对 CPU FP64 的相对误差，按类别汇总；R 对每一份 GPU 写法样本各算一遍",
        "gpu_samples": list(gpu),
        "tested": list(xs),
        "scale": scale,
        "summary": summary,
        "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for c, s in summary.items():
        print(c, s["n"], {k: f"{v:.2e}" for k, v in s["rel_median"].items()})
        for k, v in s.items():
            if k.startswith("R_"):
                print("   ", k, f"{v['median']:.2f}", v["n_gt_1.2"])


if __name__ == "__main__":
    main()
