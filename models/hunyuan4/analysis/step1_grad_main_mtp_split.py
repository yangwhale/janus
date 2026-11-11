"""第 1 步梯度按「主 loss 部分 / MTP 部分」拆开，看各侧对两部分各放大多少。

FP64 两份（`frameworks/megatron/step1_ref.py`，GPU 同一套代码）：全量 g_full，与 `--mtp off` 的 g_main（主 loss 与主干索引器 KD；
索引器的输入 detach，所以主干参数上 g_main 就是主 loss 部分）；g_mtp = g_full − g_main。对同时收两部分梯度的张量，
把被测（TPU 训练步梯度、GPU 第 1 步全量抓数、或 CPU BF16 写法）对 FP64 的误差投到两部分上：

    g_X − g_full ≈ α · g_main + β · g_mtp   （最小二乘）

α、β 是被测对两部分各自的放大量；只有整体缩放时 α = β（Adam 与 Muon 对单个张量的整体缩放不敏感），α ≠ β 才改变两部分的
相对权重。索引器参数（只吃 KD）不算；两部分里任一部分不到总量 1e-3 的张量不算。

    python step1_grad_main_mtp_split.py --full <FP64 grads_hf> --main <FP64 --mtp off grads_hf> \\
        --x tpu=<safetensors> --x gpu_a=<safetensors> ... [--map-megatron] --out <json>
`--map-megatron`：被测文件用 Megatron 名字（`cpu_bf16_gpu_forward.py --backward` 的输出），只比末尾 hc_head、末层 norm、
输出头与词嵌入。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from safetensors import safe_open

MEGATRON_TO_HF = {
    "decoder.final_layernorm.hc_fn": "model.hc_head.hc_head_fn",
    "decoder.final_layernorm.hc_base": "model.hc_head.hc_head_base",
    "decoder.final_layernorm.hc_scale": "model.hc_head.hc_head_scale",
    "decoder.final_layernorm.norm.weight": "model.norm.weight",
    "output_layer.weight": "lm_head.weight",
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
}


def group(n: str) -> str:
    if "lm_head" in n:
        return "lm_head"
    if "embed" in n:
        return "embed"
    if "hc_head" in n or n == "model.norm.weight":
        return "hc_head+norm"
    if ".hc_" in n:
        return "ihc"
    if "experts" in n or ".mlp." in n:
        return "mlp"
    if "norm" in n:
        return "norm"
    return "attn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--x", nargs="+", required=True, help="名字=safetensors")
    ap.add_argument("--map-megatron", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(32)
    full, main_ = safe_open(a.full, "pt"), safe_open(a.main, "pt")
    xs = {k: safe_open(v, "pt") for k, v in (s.split("=", 1) for s in a.x)}
    rows = []
    for hn in sorted(set(main_.keys())):
        if "indexer" in hn:
            continue
        gf = full.get_tensor(hn).double().reshape(-1)
        gm = main_.get_tensor(hn).double().reshape(-1)
        if gm.shape != gf.shape:
            continue
        gt = gf - gm
        nf, nm, nt = gf.norm().item(), gm.norm().item(), gt.norm().item()
        if nt < 1e-3 * nf or nm < 1e-3 * nf:
            continue
        A = torch.stack([gm, gt], 1)
        row = {"tensor": hn, "group": group(hn), "mtp_frac": nt / nf, "cos_main_mtp": float(gm @ gt / (nm * nt))}
        for k, f in xs.items():
            name = hn
            if a.map_megatron:
                inv = {v: kk for kk, v in MEGATRON_TO_HF.items()}
                if hn not in inv:
                    continue
                name = inv[hn]
            if name not in f.keys():
                continue
            x = f.get_tensor(name).double().reshape(-1)
            if x.shape != gf.shape:
                continue
            c = torch.linalg.solve(A.T @ A, A.T @ (x - gf))
            row[k] = {"alpha": float(c[0]), "beta": float(c[1]), "rel_err": float((x - gf).norm() / nf)}
        rows.append(row)
    summary = {}
    for g in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == g]
        summary[g] = {"n": len(rs), "mtp_frac_median": float(np.median([r["mtp_frac"] for r in rs]))}
        for k in xs:
            v = [r[k] for r in rs if k in r]
            if v:
                summary[g][k] = {"alpha_median": float(np.median([x["alpha"] for x in v])),
                                 "beta_median": float(np.median([x["beta"] for x in v]))}
    out = {"what": "第 1 步梯度对 FP64 的误差投到主 loss 部分与 MTP 部分：g_X − g_full ≈ α g_main + β g_mtp",
           "full": a.full, "main": a.main, "x": dict(s.split("=", 1) for s in a.x), "summary": summary, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for g, s in summary.items():
        print(g, s)


if __name__ == "__main__":
    main()
