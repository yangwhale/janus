"""L1 mini 单算子回放：开关全开的一次（被测）收成结果文件，另加「按位置对比」一节——同一批算子在开关关的那次
（基线）与被测上的读数并列，按四要素写（比什么、GPU 绝对值、TPU 绝对值、R），外加 TPU 对 GPU 的 rel_L2。
数字只从两份 op_replay_mini.py --compare 的输出里取，不重算。真实桶名换成 gs://<POC 桶>。

  python models/hunyuan4/analysis/l1_mini_site_compare.py <被测 cmp.json> <基线 cmp.json> <输出 json> [--note <说明>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SITES = [  # (位置, 中文, 算子名正则)
    ("sigmoid_attn_gate", "注意力整块（含输出门的 sigmoid）", r"^(\d|mtp0)\.self_attn$"),
    ("sigmoid_ihc", "iHC 混合（pre 门 sigmoid）", r"(attn_hc|ffn_hc)\.out$"),
    ("sigmoid_ihc_post", "iHC 写回（post 门 sigmoid）", r"(attn_residual|ffn_residual)$"),
    ("sigmoid_hc_head", "hc_head", r"^hc_head$"),
    ("silu_dense", "稠密 MLP（第 0 层）", r"^0\.mlp$"),
    ("silu_shared", "共享专家", r"shared_experts$"),
    ("sigmoid_router+silu_routed", "MoE 整块（自选路由）", r"mlp\(own routing\)$"),
    ("silu_routed", "路由专家（MoE 整块减共享专家）", r"routed_experts$"),
    ("cross_entropy", "交叉熵", r"^cross_entropy$"),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test")
    p.add_argument("base")
    p.add_argument("out")
    p.add_argument("--note", default=None)
    a = p.parse_args(argv)
    t = json.loads(Path(a.test).read_text())
    b = json.loads(Path(a.base).read_text())
    T = {o["op"]: o for o in t["ops"]}
    B = {o["op"]: o for o in b["ops"]}
    sites = []
    for site, zh, pat in SITES:
        ops = [k for k in T if re.search(pat, k) and "R" in T[k] and k in B and "R" in B[k]]
        if not ops:
            continue
        worst_t = max(ops, key=lambda k: T[k]["R"])
        worst_b = max(ops, key=lambda k: B[k]["R"])
        vs_gpu = [k for k in ops if "rel_l2_tpu_vs_gpu" in T[k]]
        sites.append(
            {
                "site": site,
                "ops": zh,
                "n_ops": len(ops),
                "metric": "输出 · rel_L2 对 FP64（同一算子、同一 GPU 输入；逐算子取最差）",
                "base": {
                    "op": worst_b,
                    "gpu": B[worst_b]["rel_l2_gpu_vs_fp64"],
                    "tpu": B[worst_b]["rel_l2_tpu_vs_fp64"],
                    "R": B[worst_b]["R"],
                },
                "test": {
                    "op": worst_t,
                    "gpu": T[worst_t]["rel_l2_gpu_vs_fp64"],
                    "tpu": T[worst_t]["rel_l2_tpu_vs_fp64"],
                    "R": T[worst_t]["R"],
                },
                "tpu_vs_gpu_rel_l2_mean": {
                    "base": float(np.mean([B[k]["rel_l2_tpu_vs_gpu"] for k in vs_gpu])) if vs_gpu else None,
                    "test": float(np.mean([T[k]["rel_l2_tpu_vs_gpu"] for k in vs_gpu])) if vs_gpu else None,
                },
            }
        )
    out = {
        **({"note": a.note} if a.note else {}),
        "per_site_vs_base": {"base_run": b.get("tpu"), "rule": t.get("rule"), "sites": sites},
        **t,
    }
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(out, ensure_ascii=False, indent=1))
    Path(a.out).write_text(text + "\n")
    for s in sites:
        print(
            f"{s['site']:28s} R {s['base']['R']:.3f} → {s['test']['R']:.3f}   TPU-vs-GPU {s['tpu_vs_gpu_rel_l2_mean']['base']} → {s['tpu_vs_gpu_rel_l2_mean']['test']}"
        )


if __name__ == "__main__":
    main()
