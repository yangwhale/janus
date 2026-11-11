"""同状态单步：索引器 KD 与 grad_norm，TPU、GPU 各算一步，对同一状态的 CPU FP64 真值比。

状态：第 1 步是共同的初始权重；第 k 步（8 / 30 / 100）是 GPU `g2w-b` 该步更新前的 FP32 主权重与路由偏置。
- GPU：原运行该步的记录（状态即原状态），另加从重建状态同设置重跑的几份（量 GPU 自己在同一状态上的散布）；
- TPU：`--diag-step1 --first-step k --param-override <主权重> --bias-override <偏置>`，只算梯度不更新，路由自选；
- FP64：`frameworks/megatron/step1_ref.py --dtype float64`，权重按 GPU 模型里参数的实际 dtype（GPU 留 FP32 的
  输出头、索引器 k_norm、iHC / hc_head、sink、路由偏置取 FP32 主权重原值，其余取 bf16(主权重)），路由与 DSA 自选。

比值四件套：rel = 某侧 ÷ FP64 − 1；R = |TPU rel| ÷ |GPU rel|（GPU 取原运行）。grad_norm 另核口径：三侧逐张量平方和
（HF 名字，205 个张量）之和开方是否等于记录的 grad_norm（裁剪前），并按参数组拆开对 FP64 的差；索引器各张量梯度对
FP64 的 rel_L2 与 R，看 KD 标量的偏差是否进了梯度。

    python models/hunyuan4/analysis/kd_gn_same_state.py --spec spec.json --out <json>
spec.json：{"<步>": {"tpu_diag": 目录, "gpu_metrics": 原运行 metrics.jsonl, "gpu_reruns": [metrics.jsonl…],
             "gpu_grads": HF 名字的 FP32 梯度, "fp64": step1_ref 输出目录}, …}
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

LAYER_KEY = (("moe_layers_remainder", "5"), ("moe_layers/", "1"), ("dense_layers", "0"), ("mtp", "mtp0"))
GROUPS = (("indexer", r"indexer"), ("experts", r"mlp\.experts\."), ("router", r"mlp\.gate\.weight"),
          ("shared", r"shared_experts"), ("ihc_hc_head", r"hc_"), ("sink", r"sink"), ("embed", r"embed_tokens"),
          ("lm_head", r"lm_head"), ("mtp_proj_norm", r"mtp_layers\.0\.(eh_proj|enorm|hnorm|final_layernorm)"),
          ("norm", r"norm"), ("attn", r"self_attn"), ("dense_mlp", r"mlp\."))


def grp(n):
    for g, pat in GROUPS:
        if re.search(pat, n):
            return g
    return "other"


def lid(n):
    for a, b in LAYER_KEY:
        if a in n:
            return b
    raise KeyError(n)


def metrics_at(path, step):
    for line in open(path):
        r = json.loads(line)
        if r["step"] == step:
            return r
    raise KeyError(step)


def sumsq(path):
    from safetensors import safe_open

    f = safe_open(path, "np")
    return {k: f.get_tensor(k).astype(np.float64) for k in f.keys()}


def one_step(k, s):
    tpu = json.load(open(os.path.join(s["tpu_diag"], "diag_step1.json")))
    fp = json.load(open(os.path.join(s["fp64"], "summary.json")))
    g = metrics_at(s["gpu_metrics"], k)
    reruns = [metrics_at(p, 1) for p in s.get("gpu_reruns", [])]
    ga, raw = tpu["gradient_accumulation_steps"], tpu["indexer_loss_raw"]
    tl = {lid(n): v[0] / ga / len(raw) for n, v in raw.items()}
    out = {"step": k, "kd": {}, "gpu_reruns": {}}
    for L in ("0", "1", "5", "mtp0"):
        f, gv, t = fp["indexer_loss_per_layer"][L], g[f"indexer_loss.{L}"], tl[L]
        out["kd"][L] = {"fp64": f, "gpu": gv, "tpu": t, "gpu_rel": gv / f - 1, "tpu_rel": t / f - 1}
    f, gv, t = fp["indexer_loss"], g["indexer_loss"], sum(tl.values())
    out["kd"]["total"] = {"fp64": f, "gpu": gv, "tpu": t, "gpu_rel": gv / f - 1, "tpu_rel": t / f - 1,
                          "R": abs(t / f - 1) / abs(gv / f - 1), "tpu_minus_gpu_rel": t / gv - 1}
    f, gv, t = fp["grad_norm"], g["grad_norm"], tpu["grad_norm"]
    out["grad_norm"] = {"fp64": f, "gpu": gv, "tpu": t, "gpu_rel": gv / f - 1, "tpu_rel": t / f - 1,
                        "R": abs(t / f - 1) / abs(gv / f - 1), "tpu_minus_gpu_rel": (t - gv) / ((t + gv) / 2)}
    out["lm_loss"] = {"fp64": fp["lm_loss"], "gpu": g["lm_loss"]}
    if reruns:
        out["gpu_reruns"] = {
            "n": len(reruns),
            "lm_loss_rel_vs_orig": [r["lm_loss"] / g["lm_loss"] - 1 for r in reruns],
            "kd_rel_vs_fp64": [r["indexer_loss"] / fp["indexer_loss"] - 1 for r in reruns],
            "grad_norm_rel_vs_fp64": [r["grad_norm"] / fp["grad_norm"] - 1 for r in reruns],
        }
    if not s.get("gpu_grads"):  # 第 1 步：GPU 那份全量抓数只有本 rank 的路由专家，不做逐张量
        return out
    # 逐张量
    G = sumsq(s["gpu_grads"])
    T = sumsq(os.path.join(s["tpu_diag"], "diag_grads.safetensors"))
    F = sumsq(os.path.join(s["fp64"], "grads_hf.safetensors"))
    names = sorted(F)
    assert sorted(G) == names and sorted(T) == names, "三侧张量名不一致"
    sq = {side: {n: float((d[n] ** 2).sum()) for n in names} for side, d in (("fp64", F), ("gpu", G), ("tpu", T))}
    tot = {side: sum(v.values()) for side, v in sq.items()}
    out["grad_norm_scope"] = {
        "n_tensors": len(names),
        "sqrt_sum_per_tensor": {side: v ** 0.5 for side, v in tot.items()},
        "logged": {"gpu": g["grad_norm"], "tpu": tpu["grad_norm"], "fp64": fp["grad_norm"]},
        "groups": {},
    }
    for gname in sorted({grp(n) for n in names}):
        r = sum(sq["fp64"][n] for n in names if grp(n) == gname)
        row = {"fp64_sumsq": r}
        for side in ("gpu", "tpu"):
            v = sum(sq[side][n] for n in names if grp(n) == gname)
            row[f"{side}_sumsq_rel"] = v / r - 1
            row[f"{side}_contrib_to_grad_norm_rel"] = (v - r) / (2 * tot["fp64"])
        out["grad_norm_scope"]["groups"][gname] = row
    rows = []
    for n in names:
        x = F[n]
        nx = np.linalg.norm(x)
        if nx == 0:
            continue
        eg, et = float(np.linalg.norm(G[n] - x) / nx), float(np.linalg.norm(T[n] - x) / nx)
        rows.append((n, eg, et, et / eg if eg > 0 else float("inf")))
    idx = [r for r in rows if "indexer" in r[0]]
    out["grad_tensors_vs_fp64"] = {
        "n": len(rows), "R_median": float(np.median([r[3] for r in rows])),
        "gpu_rel_l2_median": float(np.median([r[1] for r in rows])),
        "tpu_rel_l2_median": float(np.median([r[2] for r in rows])),
        "n_R_gt_1_2": int(sum(r[3] > 1.2 for r in rows)),
        "indexer": {"n": len(idx), "R_median": float(np.median([r[3] for r in idx])),
                    "R_max": float(max(r[3] for r in idx)),
                    "gpu_rel_l2_median": float(np.median([r[1] for r in idx])),
                    "tpu_rel_l2_median": float(np.median([r[2] for r in idx]))},
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    spec = json.load(open(a.spec))
    res = {}
    for k, s in spec.items():
        r = one_step(int(k), s)
        res[k] = r
        kd, gn = r["kd"]["total"], r["grad_norm"]
        print(f"step {k}: KD fp64 {kd['fp64']:.7e} GPU {kd['gpu_rel']:+.2e} TPU {kd['tpu_rel']:+.2e} R {kd['R']:.2f} | "
              f"grad_norm fp64 {gn['fp64']:.6f} GPU {gn['gpu_rel']:+.2e} TPU {gn['tpu_rel']:+.2e} R {gn['R']:.2f}")
    out = {"what": __doc__.split("\n")[0], "steps": res}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
