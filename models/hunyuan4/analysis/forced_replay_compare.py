"""GPU 第 1 步的离散选择灌进 TPU（MaxText M45 路由回放、M46 索引器 top-k 回放）后，第 1 步梯度对 GPU 同一次运行
（g8-chg-a，选择与梯度出自同一次运行）的逐张量 rel_L2，几种回放并列（不回放 / 只路由 / 路由 + 索引器 / 只索引器）。

另给两样参照：
- GPU 自己重跑：g2s-a / b / c（同设置的 GPU 重跑，只存了 rank 0 的路由专家）各自对 g8-chg-a，取中位数，作噪声尺度；
- 回放的审计（diag_step1.json 的 forced_routing.audit）：实际用的下标 / 位置对文件的不符数，与本侧自选一致的比例。
只报 TPU 对 GPU，不算 R（R 的分母要 CPU FP64 参照，另由参照一侧给）。

  python models/hunyuan4/analysis/forced_replay_compare.py --runs <TPU 运行目录的上一级> --names tw-fr0,tw-fr1,tw-fri1,tw-fi1 \\
      --gpu <g8-chg-a.safetensors> --gpu-rerun <目录，含 g2s-{a,b,c}.safetensors> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _group(name: str) -> str:
    for key, g in (
        ("mlp.experts.", "routed_experts"),
        ("mlp.shared_experts", "shared_experts"),
        ("mlp.gate", "router"),
        ("indexer", "indexer"),
        ("self_attn", "attention"),
        ("hc_", "ihc"),
        ("embed_tokens", "embedding"),
        ("lm_head", "head"),
    ):
        if key in name:
            return g
    return "other"


def compare(T: dict, G: dict) -> dict[str, float]:
    out = {}
    for h, g in G.items():
        if h not in T:
            continue
        t = T[h]
        if t.shape != g.shape and t.shape[1:] == g.shape[1:]:
            t = t[: g.shape[0]]
        if t.shape == g.shape:
            out[h] = rel(t, g)
    return out


def summarize(r: dict[str, float]) -> dict:
    v = np.array(list(r.values()))
    groups = {}
    for h, x in r.items():
        groups.setdefault(_group(h), []).append(x)
    return {
        "n_tensors": len(r),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "max_tensor": max(r, key=r.get),
        "group_median": {g: float(np.median(x)) for g, x in sorted(groups.items())},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--names", default="tw-fr0,tw-fr1,tw-fri1,tw-fi1")
    p.add_argument("--gpu", required=True)
    p.add_argument("--gpu-rerun", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from safetensors.numpy import load_file

    G = load_file(a.gpu)
    res = {"what": __doc__.split("\n\n")[0], "gpu": Path(a.gpu).name, "runs": {}}
    noise = None
    if a.gpu_rerun:
        per = {}
        for r in ("g2s-a", "g2s-b", "g2s-c"):
            f = Path(a.gpu_rerun) / f"{r}.safetensors"
            if f.is_file():
                for h, x in compare(load_file(str(f)), G).items():
                    per.setdefault(h, []).append(x)
        noise = {h: float(np.median(x)) for h, x in per.items()}
        res["gpu_rerun_vs_g8"] = summarize(noise) if noise else None
    per_run = {}
    for n in a.names.split(","):
        d = Path(a.runs) / n
        if not (d / "diag_grads.safetensors").is_file():
            res["runs"][n] = {"missing": True}
            continue
        r = compare(load_file(str(d / "diag_grads.safetensors")), G)
        per_run[n] = r
        diag = json.loads((d / "diag_step1.json").read_text())
        fr = diag.get("forced_routing") or {}
        res["runs"][n] = {
            "loss": diag.get("loss"),
            "grad_norm": diag.get("grad_norm"),
            "replay": {k: fr[k].get("variant") for k in ("routing", "indexer") if k in fr},
            "replay_sha256": {k: fr[k].get("sha256") for k in ("routing", "indexer") if k in fr},
            "audit": (fr.get("audit") or {}).get("summary"),
            "tpu_vs_gpu": summarize(r),
        }
        if noise:
            ratio = {h: r[h] / noise[h] for h in r if h in noise and noise[h] > 0}
            res["runs"][n]["tpu_vs_gpu_over_gpu_rerun"] = {
                "median": float(np.median(list(ratio.values()))),
                "n_gt_1.2": int(sum(x > 1.2 for x in ratio.values())),
                "top": sorted(ratio.items(), key=lambda kv: -kv[1])[:10],
            }
    base = per_run.get("tw-fr0")
    if base:
        for n, r in per_run.items():
            if n == "tw-fr0":
                continue
            better = {h: r[h] / base[h] for h in r if h in base and base[h] > 0}
            res["runs"][n]["vs_no_replay"] = {
                "median_ratio": float(np.median(list(better.values()))),
                "group_median_ratio": {
                    g: float(np.median([x for h, x in better.items() if _group(h) == g]))
                    for g in sorted({_group(h) for h in better})
                },
                "note": "每个张量 rel_L2(回放, GPU) ÷ rel_L2(不回放, GPU) 的中位数；< 1 = 回放后更接近 GPU",
            }
    res["rule"] = "只报 TPU 对 GPU；R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64) 等 CPU FP64 参照"
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({n: (v.get("tpu_vs_gpu") or {}).get("median") for n, v in res["runs"].items()}))


if __name__ == "__main__":
    main()
