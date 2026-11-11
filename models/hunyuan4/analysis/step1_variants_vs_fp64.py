"""几份 TPU 第 1 步梯度（同权重、同 batch，只差编译选项等）各对 CPU FP64 参照、对 GPU 三份、彼此之间比。

逐张量（HF 名字；路由专家只取 GPU 导出旁边 JSON 里有的那几个）：
- rel_L2(X, FP64) = ‖X − FP64‖ ÷ ‖FP64‖，X 是每份 TPU 与三份 GPU；GPU 取三份的中位数；
- R = rel_L2(TPU, FP64) ÷ GPU 中位数；
- 两份 TPU 之间：‖T_i − T_j‖ ÷ ‖FP64‖，以及逐位相同的张量数；
- 对 GPU 的 r（与参照无关，同 step1_vs_fp64.py 的旧口径）：TPU 对最近一份 GPU 的 rel_L2（分母 ‖GPU‖）÷ GPU 三份两两的最大 rel_L2。
汇总：全体与分模块的中位数、R > 1.2 的个数、r > 1 / > 2 的个数。

    python models/hunyuan4/analysis/step1_variants_vs_fp64.py --ref <fp64 目录> \\
        --gpu g2s-a=<hf.safetensors> g2s-b=<…> g2s-c=<…> --tpu A=<diag_grads.safetensors> C=<…> D=<…> --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re

import numpy as np
import torch
from safetensors import safe_open

R_LIMIT = 1.2
MODULES = [  # (模块, 正则)；按顺序取第一个匹配（同 step1_vs_fp64.py）
    ("iHC", r"\.hc_(attn|mlp)_layer\.|^model\.hc_head\."),
    ("索引器", r"\.self_attn\.indexer\."),
    ("注意力", r"\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|linear_gate|learnable_sink_param)"),
    ("MLP / 专家", r"\.mlp\."),
    ("norm", r"layernorm\.weight$|^model\.norm\.|\.(enorm|hnorm|final_layernorm)\."),
    ("其他", r"."),
]


def module_of(name):
    return next(m for m, pat in MODULES if re.search(pat, name))


def med(v):
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="step1_ref.py 的输出目录（FP64，grads_hf.safetensors）")
    p.add_argument("--gpu", nargs=3, required=True, help="名字=HF 名字的梯度 safetensors")
    p.add_argument("--tpu", nargs="+", required=True, help="名字=diag_grads.safetensors，第一份是基线")
    p.add_argument("--only", default=None, help="只比名字匹配这个正则的张量（TPU 只存了一部分时用）")
    p.add_argument("--note", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    ref = safe_open(os.path.join(a.ref, "grads_hf.safetensors"), "pt")
    gnames, gfiles, gexp = [], [], []
    for s in a.gpu:
        n, path = s.split("=", 1)
        gnames.append(n)
        gfiles.append(safe_open(path, "pt"))
        side = os.path.splitext(path)[0] + ".json"
        gexp.append(json.load(open(side)).get("tensors", {}) if os.path.exists(side) else {})
    tnames = [s.split("=", 1)[0] for s in a.tpu]
    tfiles = [safe_open(s.split("=", 1)[1], "pt") for s in a.tpu]

    rows = []
    for h in sorted(gfiles[0].keys()):
        if a.only and not re.search(a.only, h):
            continue
        g = [f.get_tensor(h).to(torch.float64) for f in gfiles]
        fr = ref.get_tensor(h)
        sl = None
        if tuple(fr.shape) != tuple(g[0].shape):  # 路由专家：GPU 只有 rank 0 本地的几个
            ex = gexp[0].get(h, {}).get("experts")
            sl = torch.tensor(ex if ex is not None else list(range(g[0].shape[0])))
            fr = fr[sl]
        f64 = fr.to(torch.float64)
        den = float(f64.norm())
        ts = []
        for f in tfiles:
            x = f.get_tensor(h)
            ts.append((x[sl] if sl is not None else x).to(torch.float64))
        gn = [float(x.norm()) for x in g]
        gpu_ref = [float((x - f64).norm()) / den for x in g]
        ff = float(f64.pow(2).sum())

        def along(x):  # 误差沿 FP64 方向的分量：⟨X − FP64, FP64⟩ ÷ ‖FP64‖²（整体放大为正、缩小为负）
            return float(((x - f64) * f64).sum()) / ff if ff > 0 else float("nan")

        noise = max(float((g[i] - g[j]).norm()) / gn[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        r = {"tensor": h, "module": module_of(h), "numel": int(f64.numel()), "gpu_vs_ref": med(gpu_ref),
             "gpu_vs_ref_each": gpu_ref, "gpu_noise_max": noise, "gpu_along_ref_each": [along(x) for x in g], "tpu": {}}
        for n, t in zip(tnames, ts):
            tv = float((t - f64).norm()) / den
            over = min(float((t - x).norm()) / m for x, m in zip(g, gn)) / noise if noise > 0 else float("nan")
            r["tpu"][n] = {"vs_ref": tv, "R": tv / r["gpu_vs_ref"] if r["gpu_vs_ref"] > 0 else float("nan"),
                           "over_N": over, "along_ref": along(t)}
        r["pairs"] = {f"{tnames[i]}-{tnames[j]}": {"rel": float((ts[i] - ts[j]).norm()) / den,
                                                   "bit_identical": bool(torch.equal(ts[i], ts[j]))}
                      for i, j in itertools.combinations(range(len(ts)), 2)}
        rows.append(r)

    def summarize(rs):
        s = {"n": len(rs), "gpu_vs_ref_median": med([r["gpu_vs_ref"] for r in rs])}
        for n in tnames:
            s[n] = {"vs_ref_median": med([r["tpu"][n]["vs_ref"] for r in rs]),
                    "R_median": med([r["tpu"][n]["R"] for r in rs]),
                    "n_R_gt_1.2": sum(r["tpu"][n]["R"] > R_LIMIT for r in rs),
                    "over_N_median": med([r["tpu"][n]["over_N"] for r in rs]),
                    "n_over_N_gt_1": sum(r["tpu"][n]["over_N"] > 1 for r in rs),
                    "n_over_N_gt_2": sum(r["tpu"][n]["over_N"] > 2 for r in rs)}
        for k in rows[0]["pairs"]:
            s[k] = {"rel_median": med([r["pairs"][k]["rel"] for r in rs]),
                    "rel_max": max(r["pairs"][k]["rel"] for r in rs),
                    "n_bit_identical": sum(r["pairs"][k]["bit_identical"] for r in rs)}
        return s

    out = {
        "what": f"第 1 步全部参数梯度：TPU {'/'.join(tnames)} 与 GPU {'/'.join(gnames)} 各对 CPU FP64 参照，TPU 彼此之间",
        "reference": os.path.basename(os.path.normpath(a.ref)),
        "definitions": {
            "vs_ref": "‖X − FP64‖ ÷ ‖FP64‖",
            "R": "rel_L2(TPU, FP64) ÷ 三份 GPU 各自对 FP64 的中位数",
            "pair_rel": "‖T_i − T_j‖ ÷ ‖FP64‖",
            "over_N": "TPU 对最近一份 GPU 的 rel_L2（分母 ‖GPU‖）÷ GPU 三份两两的最大 rel_L2（与参照无关）",
            "experts": "路由专家只比 GPU 导出里有的那几个",
            "along_ref": "⟨X − FP64, FP64⟩ ÷ ‖FP64‖²：误差里沿 FP64 方向的整体缩放（负 = 整体偏小）",
        },
        "note": a.note,
        "all": summarize(rows),
        "by_module": {m: summarize([r for r in rows if r["module"] == m]) for m, _ in MODULES
                      if any(r["module"] == m for r in rows)},
        "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["all"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
