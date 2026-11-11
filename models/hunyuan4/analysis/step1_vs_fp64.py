"""第 1 步逐张量梯度对 CPU FP64 参照（frameworks/megatron/step1_ref.py 的输出）的三方距离与定责。

逐张量（HF 名字；路由专家只取 GPU 全量抓数有的那几个，见 GPU 导出旁边的 JSON）：
- rel_L2(GPU_i, FP64)、rel_L2(TPU, FP64)、rel_L2(GPU_i, TPU)，分母都是 ‖FP64‖；GPU 三份两两的噪声同一分母；
- R = rel_L2(TPU, FP64) ÷ 中位数_i rel_L2(GPU_i, FP64)（同旧口径：GPU 取三份的中位数），另记每份的 R；
- 误差向量夹角余弦 cos_i = ⟨GPU_i − FP64, TPU − FP64⟩ ÷ (‖GPU_i − FP64‖·‖TPU − FP64‖)，直接用完整张量算；
- 「超出 N」沿用旧口径（与参照无关）：TPU 对最近一份 GPU 的 rel_L2（分母 ‖GPU‖）÷ GPU 三份两两的最大 rel_L2；
- 判定同 dashboard/build.py 的 `_step1_judge`（PROTOCOL 第 5 节 + 2026-11-08 分辨率下限）。
- 给了 `--tpu-diag` 就先核 TPU 梯度文件：逐张量平方和对设备上算的（`per_tensor.sq`），对不上的记为落盘损坏、不进统计与判定。
另报：旧参照（TPU FP32 + HIGHEST）与 CPU FP32 自检各自离 FP64 多远；旧参照下与新参照下定责结论的差异；
离散选择（第一行）FP64 对 GPU 的翻转，按 FP64 分差分桶；灌 GPU 选择的灵敏度（若给了 --inject）。

    python models/hunyuan4/analysis/step1_vs_fp64.py --ref <fp64 目录> --gpu g2s-a=<hf.safetensors> ... \\
        --tpu t1zdiag2=<diag_grads.safetensors> [--old-ref t1diag18b=<…>] [--cpu32 <fp32 目录>] \\
        [--gpu-l1 <GPU l1 抓数 step_1_fwd.safetensors>] [--gpu-route-dump <GPU 全量抓数 step_1 目录>] --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
from safetensors import safe_open

R_LIMIT = 1.2
RES_FLOOR = 2.0**-8
FILE_SQ_TOL = 1e-10  # TPU 梯度文件对设备平方和的相对差上限（干净的实测 1e-14 量级，落盘置零的 ≥ 1e-7）
MODULES = [  # (模块, 正则)；按顺序取第一个匹配
    ("iHC", r"\.hc_(attn|mlp)_layer\.|^model\.hc_head\."),
    ("索引器", r"\.self_attn\.indexer\."),
    (
        "注意力",
        r"\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|linear_gate|learnable_sink_param)",
    ),
    ("MLP / 专家", r"\.mlp\."),
    ("norm", r"layernorm\.weight$|^model\.norm\.|\.(enorm|hnorm|final_layernorm)\."),
    ("其他", r"."),
]


def module_of(name):
    for m, pat in MODULES:
        if re.search(pat, name):
            return m
    return "其他"


def load(path):
    f = safe_open(path, "pt")
    return f, set(f.keys())


def t64(f, k, sl=None):
    x = f.get_tensor(k)
    if sl is not None and x.shape[0] != len(sl):
        x = x[sl]
    return x.to(torch.float64)


def rel(a, b, den):
    return float((a - b).norm() / den) if den > 0 else float("nan")


def judge(rows, key_r="R_vs_ref", key_g="gpu_vs_ref", key_t="tpu_vs_ref"):
    """同 dashboard/build.py 的 _step1_judge。"""

    def below(r):
        return r["gpu_noise_max"] == 0 and r[key_t] < RES_FLOOR and r[key_g] < RES_FLOOR

    bad = [r["tensor"] for r in rows if r.get("tpu_file_corrupt")]
    rows = [r for r in rows if not r.get("tpu_file_corrupt")]
    zero = [r for r in rows if r["gpu_noise_max"] == 0 and not below(r)]
    pos = [(r, r["tpu_vs_gpu_min"] / r["gpu_noise_max"]) for r in rows if r["gpu_noise_max"] > 0]
    un2 = [r["tensor"] for r, x in pos if x > 2 and r[key_r] > R_LIMIT]
    un1 = [r["tensor"] for r, x in pos if 1 < x <= 2 and r[key_r] > R_LIMIT]
    zbad = [r["tensor"] for r in zero if r[key_r] > R_LIMIT]
    st = "fail" if un2 or zbad else ("inconclusive" if un1 or bad else "pass")
    return {
        "status": st,
        "tpu_file_corrupt_not_judged": bad,
        "unattributed_over_2N": un2,
        "unattributed_1_2N": un1,
        "unattributed_N0": zbad,
        "n_over_N": sum(x > 1 for _, x in pos),
        "n_over_2N": sum(x > 2 for _, x in pos),
        "n_below_resolution_floor": sum(below(r) for r in rows),
    }


def q(v, p):
    return float(np.quantile(np.asarray(v), p))


def flip_buckets(gaps):
    edges = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    lab = ["≤1e-6", "1e-6–1e-5", "1e-5–1e-4", "1e-4–1e-3", "1e-3–1e-2", ">1e-2"]
    out = dict.fromkeys(lab, 0)
    for g in gaps:
        i = sum(g > e for e in edges)
        out[lab[i]] += 1
    return out


def route_flips(ref_idx, ref_scores, gpu_idx):
    """逐 token 比选择集合；翻转的 token 记 FP64 分差：GPU 换进来的专家要比换掉的高出多少（相对第 k 名的分数）。"""
    a = np.sort(ref_idx, 1)
    b = np.sort(gpu_idx, 1)
    rows = np.nonzero((a != b).any(1))[0]
    gaps = []
    k = ref_idx.shape[1]
    for t in rows:
        s = ref_scores[t]
        out_ = sorted(set(a[t]) - set(b[t]))  # FP64 选了、GPU 没选
        in_ = sorted(set(b[t]) - set(a[t]))  # GPU 选了、FP64 没选
        sk = np.sort(s)[::-1][k - 1]
        gaps.append(float((max(s[out_]) - min(s[in_])) / abs(sk)))
    return int(len(rows)), gaps


def topk_mask(tk):
    S = tk.shape[0]
    m = np.zeros((S, S + 1), bool)
    np.put_along_axis(m, np.where(tk >= 0, tk, S), True, 1)
    return m[:, :S] & np.tri(S, dtype=bool)


def dsa_flips(ref_topk, other_mask, q_, k_, w_):
    """逐查询行比 top-k 集合（因果内）；翻转的行记 FP64 分差：同上，但除以这一行索引分数的最大绝对值
    （索引分数有正有负，第 k 名常在 0 附近，不能拿它作分母）。"""
    ma, mb = topk_mask(ref_topk), other_mask
    rows = np.nonzero((ma != mb).any(1))[0]
    gaps, nel = [], 0
    for s in rows:
        sc = torch.relu(torch.einsum("hd,td->ht", q_[s, 0], k_[: s + 1, 0])) * w_[s, 0][:, None]
        sc = sc.sum(0).numpy()
        out_ = np.nonzero(ma[s, : s + 1] & ~mb[s, : s + 1])[0]
        in_ = np.nonzero(mb[s, : s + 1] & ~ma[s, : s + 1])[0]
        nel += len(out_)
        gaps.append(float((sc[out_].max() - sc[in_].min()) / np.abs(sc).max()))
    return int(len(rows)), int(nel), gaps


def gpu_only(a, ref, refsum, gnames, gfiles, gexp):
    """TPU 第 1 步梯度还没有时：GPU 三份各对 FP64、GPU 两两噪声，按模块汇总；不判定。"""
    rows = []
    for h in sorted(gfiles[0].keys()):
        gs = [f.get_tensor(h).to(torch.float64) for f in gfiles]
        ex = gexp[0].get(h, {}).get("experts")
        f64 = t64(ref, h, torch.tensor(ex if ex is not None else list(range(gs[0].shape[0]))))
        den = float(f64.norm())
        r = {
            "tensor": h,
            "module": module_of(h),
            "numel": int(f64.numel()),
            "ref_norm": den,
            "gpu_vs_ref_each": [rel(x, f64, den) for x in gs],
            "gpu_noise_each": [rel(gs[i], gs[j], den) for i, j in ((0, 1), (0, 2), (1, 2))],
        }
        r["gpu_vs_ref"] = float(np.median(r["gpu_vs_ref_each"]))
        e = [x - f64 for x in gs]
        r["gpu_err_cos_pairs"] = [
            float((e[i] * e[j]).sum() / (float(e[i].norm()) * float(e[j].norm())))
            if float(e[i].norm()) * float(e[j].norm()) > 0
            else float("nan")
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        rows.append(r)
    out = {
        "what": f"第 1 步全部参数梯度：GPU 第 1 步参照（{'/'.join(gnames)}）各对 CPU FP64 参照（TPU 第 1 步梯度还没有，不判定）",
        "reference": {
            k: refsum.get(k)
            for k in (
                "dtype",
                "lm_loss",
                "mtp_loss",
                "indexer_loss",
                "grad_norm",
                "global_tokens",
                "rows",
                "code",
                "audit",
                "kept_experts",
            )
        },
        "n_tensors": len(rows),
        "median": {
            "gpu_vs_ref": float(np.median([r["gpu_vs_ref"] for r in rows])),
            "gpu_noise": float(np.median([max(r["gpu_noise_each"]) for r in rows])),
            "gpu_err_cos": float(np.median([np.median(r["gpu_err_cos_pairs"]) for r in rows])),
        },
        "by_module": {
            m: {
                "n": len(rs),
                "gpu_vs_ref_median": float(np.median([r["gpu_vs_ref"] for r in rs])),
                "gpu_noise_median": float(np.median([max(r["gpu_noise_each"]) for r in rs])),
            }
            for m, _ in MODULES
            if (rs := [r for r in rows if r["module"] == m])
        },
        "note": a.note,
        "rows": rows,
    }
    if a.gpu_meta:
        m = json.loads(open(a.gpu_meta).readline())
        out["loss_vs_gpu"] = {
            k: {"gpu": m.get(k), "fp64": refsum[k], "diff": (m.get(k) - refsum[k]) if m.get(k) is not None else None}
            for k in ("lm_loss", "mtp_loss", "indexer_loss", "grad_norm")
        }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["median"], ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="step1_ref.py 的输出目录（FP64）")
    p.add_argument("--gpu", nargs=3, required=True, help="名字=HF 名字的梯度 safetensors（export_hf_grads.py）")
    p.add_argument(
        "--tpu", default="", help="名字=diag_grads.safetensors；不给时只报 GPU 三份对 FP64（TPU 第 1 步梯度还没有时）"
    )
    p.add_argument("--old-ref", default="", help="名字=旧参照的 diag_grads.safetensors（TPU FP32 + HIGHEST）")
    p.add_argument("--cpu32", default="", help="step1_ref.py --dtype float32 的输出目录（自检）")
    p.add_argument("--gpu-l1", default="", help="GPU l1 抓数 step_1_fwd.safetensors（第一行的 DSA top-k 与路由）")
    p.add_argument("--gpu-route-dump", default="", help="GPU 全量抓数 step_1 目录（layers.<L>.safetensors 的路由下标）")
    p.add_argument("--tpu-kd-inputs", default="", help="TPU diag_kd_inputs.safetensors（第一行各 full 层的 DSA 掩码）")
    p.add_argument(
        "--tpu-kd-layers",
        default="0=decoder/dense_layers,1=decoder/moe_layers/layers_0,5=decoder/moe_layers_remainder/layers_0,"
        "mtp0=mtp_block/mtp_layer_1/mtp_1_transformer_layer",
        help="层名=MaxText 抓数的前缀（hy4-mini 的取值）",
    )
    p.add_argument("--gpu-meta", default="", help="GPU 第 1 步的 metrics.jsonl（loss 对照）")
    p.add_argument(
        "--tpu-diag",
        default="",
        help="TPU diag_step1.json（loss 对照；其 per_tensor.sq 是设备上算的平方和，用来核 TPU 梯度文件落盘有没有坏）",
    )
    p.add_argument("--note", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ref, refk = load(os.path.join(a.ref, "grads_hf.safetensors"))
    refsum = json.load(open(os.path.join(a.ref, "summary.json")))
    gnames, gfiles, gexp = [], [], []
    for s in a.gpu:
        n, path = s.split("=", 1)
        gnames.append(n)
        gfiles.append(load(path)[0])
        side = os.path.splitext(path)[0] + ".json"
        side = side if os.path.exists(side) else os.path.splitext(path)[0] + ".sidecar.json"
        gexp.append(json.load(open(side))["tensors"] if os.path.exists(side) else {})
    if not a.tpu:
        return gpu_only(a, ref, refsum, gnames, gfiles, gexp)
    tname, tpath = a.tpu.split("=", 1)
    tpu = load(tpath)[0]
    old = load(a.old_ref.split("=", 1)[1])[0] if a.old_ref else None
    oname = a.old_ref.split("=", 1)[0] if a.old_ref else None
    c32 = load(os.path.join(a.cpu32, "grads_hf.safetensors"))[0] if a.cpu32 else None

    # TPU 梯度文件的完整性：文件里每个张量的平方和对设备上算的（diag_step1.json 的 per_tensor.sq）。
    # 干净的张量两者相对差 1e-14 量级；落盘时一段被写成 0 的张量差 1e-7 以上（文件偏小）。超过 FILE_SQ_TOL 的不可判。
    dev_sq = (json.load(open(a.tpu_diag)).get("per_tensor") or {}) if a.tpu_diag else {}
    rows = []
    for h in sorted(gfiles[0].keys()):
        gs = [f.get_tensor(h) for f in gfiles]
        shape = tuple(gs[0].shape)
        fr = ref.get_tensor(h)
        ex = gexp[0].get(h, {}).get("experts")  # 路由专家：GPU 只有 rank 0 本地的几个
        sl = torch.tensor(ex if ex is not None else list(range(shape[0])))
        if tuple(fr.shape) != shape:  # 参照可能已只留这几个（step1_ref.py --keep-experts）
            fr = fr[sl]
        f64 = fr.to(torch.float64)
        g = [x.to(torch.float64) for x in gs]
        t = t64(tpu, h, sl)
        den = float(f64.norm())
        gn = [float(x.norm()) for x in g]
        eg = [x - f64 for x in g]
        et = t - f64
        net = float(et.norm())
        r = {
            "tensor": h,
            "module": module_of(h),
            "numel": int(f64.numel()),
            "ref_norm": den,
            "gpu_vs_ref_each": [rel(x, f64, den) for x in g],
            "tpu_vs_ref": net / den if den else float("nan"),
            "gpu_vs_tpu_each": [rel(x, t, den) for x in g],
            "gpu_noise_each": [rel(g[i], g[j], den) for i, j in ((0, 1), (0, 2), (1, 2))],
            # 旧口径（与参照无关）：TPU 对最近一份 GPU、GPU 两两，分母是 GPU 一侧
            "tpu_vs_gpu_min": min(rel(t, x, n) for x, n in zip(g, gn)),
            "gpu_noise_max": max(rel(g[i], g[j], gn[j]) for i, j in ((0, 1), (0, 2), (1, 2))),
            "cos_each": [
                float((e * et).sum() / (float(e.norm()) * net)) if float(e.norm()) * net > 0 else float("nan")
                for e in eg
            ],
        }
        r["gpu_vs_ref"] = float(np.median(r["gpu_vs_ref_each"]))
        r["R_vs_ref"] = r["tpu_vs_ref"] / r["gpu_vs_ref"] if r["gpu_vs_ref"] > 0 else float("nan")
        r["R_each"] = [r["tpu_vs_ref"] / x if x > 0 else float("nan") for x in r["gpu_vs_ref_each"]]
        r["cos"] = float(np.median(r["cos_each"]))
        r["gpu_vs_tpu"] = float(np.median(r["gpu_vs_tpu_each"]))
        # 余弦定理核对：(g² + t² − d²) / (2gt)，GPU 第 1 份
        g0, t0, d0 = r["gpu_vs_ref_each"][0], r["tpu_vs_ref"], r["gpu_vs_tpu_each"][0]
        r["cos_law_check_a"] = (g0**2 + t0**2 - d0**2) / (2 * g0 * t0) if g0 * t0 > 0 else float("nan")
        if h in dev_sq:
            fs = float(tpu.get_tensor(h).to(torch.float64).pow(2).sum())  # 整个张量（不只比的那几个专家）
            r["tpu_file_sq_rel"] = (fs - dev_sq[h]["sq"]) / dev_sq[h]["sq"] if dev_sq[h]["sq"] else 0.0
            r["tpu_file_corrupt"] = abs(r["tpu_file_sq_rel"]) > FILE_SQ_TOL
        if old is not None:
            o = t64(old, h, sl)
            r["old_ref_vs_fp64"] = rel(o, f64, den)
            on = float(o.norm())
            go = float(np.median([rel(x, o, on) for x in g]))
            r["old_gpu_vs_ref"], r["old_tpu_vs_ref"] = go, rel(t, o, on)
            r["old_R_vs_ref"] = r["old_tpu_vs_ref"] / go if go > 0 else float("nan")
        if c32 is not None:
            c = t64(c32, h, sl)
            r["cpu32_vs_fp64"] = rel(c, f64, den)
            r["gpu_vs_cpu32"] = float(np.median([rel(x, c, den) for x in g]))
        rows.append(r)
        del g, gs, eg, et, f64, t

    # 落盘损坏的张量（见上）留在 rows 里备查，但不进任何统计与判定
    corrupt = [(r["tensor"], r["tpu_file_sq_rel"]) for r in rows if r.get("tpu_file_corrupt")]
    allrows, rows = rows, [r for r in rows if not r.get("tpu_file_corrupt")]
    R = [r["R_vs_ref"] for r in rows]
    new = judge(allrows)
    out = {
        "what": f"T1 第 1 步全部参数梯度：TPU（{tname}）与 GPU 第 1 步参照（{'/'.join(gnames)}）各对 CPU FP64 参照",
        "reference": {
            "kind": "CPU FP64，GPU golden 同一套 Megatron-Core 代码（frameworks/megatron/step1_ref.py），与 TPU 不同源",
            "dir": os.path.basename(os.path.normpath(a.ref)),
            **{
                k: refsum[k]
                for k in ("dtype", "lm_loss", "mtp_loss", "indexer_loss", "grad_norm", "global_tokens", "rows", "code")
            },
            "audit": refsum.get("audit"),
            "dtype_rewrites": refsum.get("dtype_rewrites"),
        },
        "definitions": {
            "rel_L2": "‖a − b‖ ÷ ‖FP64‖",
            "gpu_vs_ref": "三份 GPU 各自对 FP64 的 rel_L2 的中位数",
            "R_vs_ref": "rel_L2(TPU, FP64) ÷ gpu_vs_ref",
            "cos": "误差向量 (GPU_i − FP64) 与 (TPU − FP64) 的夹角余弦，三份取中位数（完整张量直接点积）",
            "over_N": "tpu_vs_gpu_min ÷ gpu_noise_max（旧口径，分母 ‖GPU‖，与参照无关）",
            "judge": "同 dashboard/build.py 的 _step1_judge：超 2N 且 R > 1.2 不过；1–2N 且 R > 1.2 待定；N = 0 的按分辨率下限",
            "experts": "路由专家只比 GPU 全量抓数里有的（rank 0 本地 0–3 号）",
        },
        "n_tensors": len(allrows),
        "tpu_file_check": {
            "checked": bool(dev_sq),
            "rule": f"TPU 梯度文件里每个张量的平方和对设备上算的（diag_step1.json 的 per_tensor.sq），相对差 > {FILE_SQ_TOL:g} 记为落盘损坏、不可判",
            "corrupt": [{"tensor": t, "file_sq_rel": x} for t, x in corrupt],
            "stats_exclude_corrupt": True,
        },
        "median": {
            "gpu_vs_ref_each": [float(np.median([r["gpu_vs_ref_each"][i] for r in rows])) for i in range(3)],
            "gpu_vs_ref": float(np.median([r["gpu_vs_ref"] for r in rows])),
            "tpu_vs_ref": float(np.median([r["tpu_vs_ref"] for r in rows])),
            "gpu_vs_tpu": float(np.median([r["gpu_vs_tpu"] for r in rows])),
            "gpu_noise": float(np.median([max(r["gpu_noise_each"]) for r in rows])),
            "cos": float(np.median([r["cos"] for r in rows])),
            "R_vs_ref": float(np.median(R)),
        },
        "R_distribution": {
            "q10": q(R, 0.1),
            "q25": q(R, 0.25),
            "q50": q(R, 0.5),
            "q75": q(R, 0.75),
            "q90": q(R, 0.9),
            "max": max(R),
            "n_lt_0.8": sum(x < 0.8 for x in R),
            "n_0.8_1.0": sum(0.8 <= x <= 1.0 for x in R),
            "n_1.0_1.2": sum(1.0 < x <= 1.2 for x in R),
            "n_gt_1.2": sum(x > 1.2 for x in R),
        },
        "R_gt_1.2": [
            {
                k: r[k]
                for k in (
                    "tensor",
                    "module",
                    "R_vs_ref",
                    "gpu_vs_ref",
                    "tpu_vs_ref",
                    "cos",
                    "tpu_vs_gpu_min",
                    "gpu_noise_max",
                )
            }
            for r in sorted(rows, key=lambda r: -r["R_vs_ref"])
            if r["R_vs_ref"] > R_LIMIT
        ],
        "by_module": {},
        "judge_fp64": new,
    }
    for m, _ in MODULES:
        rs = [r for r in rows if r["module"] == m]
        if not rs:
            continue
        out["by_module"][m] = {
            "n": len(rs),
            "gpu_vs_ref_median": float(np.median([r["gpu_vs_ref"] for r in rs])),
            "tpu_vs_ref_median": float(np.median([r["tpu_vs_ref"] for r in rs])),
            "gpu_vs_tpu_median": float(np.median([r["gpu_vs_tpu"] for r in rs])),
            "cos_median": float(np.median([r["cos"] for r in rs])),
            "R_median": float(np.median([r["R_vs_ref"] for r in rs])),
            "R_max": max(r["R_vs_ref"] for r in rs),
            "n_R_gt_1.2": sum(r["R_vs_ref"] > R_LIMIT for r in rs),
        }
    if old is not None:
        oldj = judge(allrows, "old_R_vs_ref", "old_gpu_vs_ref", "old_tpu_vs_ref")
        un = lambda j: set(j["unattributed_over_2N"] + j["unattributed_1_2N"] + j["unattributed_N0"])  # noqa: E731
        oR = [r["old_R_vs_ref"] for r in rows]
        out["old_reference"] = {
            "name": oname,
            "kind": "TPU 一侧 MaxText，dtype float32、matmul_precision highest、sparse_matmul false（被测一侧产出）",
            "old_ref_vs_fp64_median": float(np.median([r["old_ref_vs_fp64"] for r in rows])),
            "old_ref_vs_fp64_over_gpu_vs_fp64_median": float(
                np.median([r["old_ref_vs_fp64"] / r["gpu_vs_ref"] for r in rows if r["gpu_vs_ref"] > 0])
            ),
            "n_old_ref_err_gt_half_gpu_err": sum(r["old_ref_vs_fp64"] > 0.5 * r["gpu_vs_ref"] for r in rows),
            "R_median_old": float(np.median(oR)),
            "n_R_gt_1.2_old": sum(x > R_LIMIT for x in oR),
            "judge_old": oldj,
            "newly_unattributed": sorted(un(new) - un(oldj)),
            "no_longer_unattributed": sorted(un(oldj) - un(new)),
            "R_crossed_1.2": [
                {"tensor": r["tensor"], "R_old": r["old_R_vs_ref"], "R_fp64": r["R_vs_ref"]}
                for r in rows
                if (r["old_R_vs_ref"] > R_LIMIT) != (r["R_vs_ref"] > R_LIMIT)
            ],
        }
    if c32 is not None:
        out["self_check_cpu32"] = {
            "cpu32_vs_fp64_median": float(np.median([r["cpu32_vs_fp64"] for r in rows])),
            "cpu32_vs_fp64_max": max(r["cpu32_vs_fp64"] for r in rows),
            "gpu_vs_cpu32_over_gpu_vs_fp64_median": float(
                np.median([r["gpu_vs_cpu32"] / r["gpu_vs_ref"] for r in rows if r["gpu_vs_ref"] > 0])
            ),
            "summary": {
                k: json.load(open(os.path.join(a.cpu32, "summary.json")))[k]
                for k in ("lm_loss", "mtp_loss", "indexer_loss", "grad_norm")
            },
        }
    if a.gpu_meta:
        m = json.loads(open(a.gpu_meta).readline())
        out["loss_vs_gpu"] = {
            k: {"gpu": m.get(k2), "fp64": refsum[k], "diff": (m.get(k2) - refsum[k]) if m.get(k2) is not None else None}
            for k, k2 in (
                ("lm_loss", "lm_loss"),
                ("mtp_loss", "mtp_loss"),
                ("indexer_loss", "indexer_loss"),
                ("grad_norm", "grad_norm"),
            )
        }
    if a.tpu_diag:
        d = json.load(open(a.tpu_diag))
        out["tpu_diag"] = {
            "objective": d.get("loss"),
            "objective_note": "TPU 训练目标（diag_step1.json 的 loss：主 loss 加上本模型有的 aux，如 0.1 × MTP、索引器 KD），不是主 loss",
            "grad_norm": d.get("grad_norm"),
        }

    # 离散选择的翻转（第一行；GPU 只抓了 rank 0 第 0 个 microbatch）
    ch_path = os.path.join(a.ref, "choices.safetensors")
    if os.path.exists(ch_path) and (a.gpu_l1 or a.gpu_route_dump):
        ch = safe_open(ch_path, "pt")
        keys = set(ch.keys())
        row = min(int(k.split(".")[0][1:]) for k in keys if ".dsa." in k)
        flips = {"row": row, "route": {}, "dsa": {}}
        l1 = safe_open(a.gpu_l1, "pt") if a.gpu_l1 else None
        tkd = safe_open(a.tpu_kd_inputs, "pt") if a.tpu_kd_inputs else None
        tkd_map = dict(x.split("=", 1) for x in a.tpu_kd_layers.split(",") if x)
        names = sorted({k.split(".")[2] for k in keys if k.startswith(f"r{row}.route.")})
        for L in names:
            ri = ch.get_tensor(f"r{row}.route.{L}.idx").numpy().astype(np.int64)
            rs = ch.get_tensor(f"r{row}.route.{L}.scores").to(torch.float64).numpy()
            ent = {}
            hn = f"hf.model.layers.{L}" if L != "mtp0" else "hf.model.mtp_layers.0"
            if l1 is not None:
                rm = l1.get_tensor(f"{hn}.mlp.gate.routing_map").numpy()
                gi = np.argsort(-rm.astype(np.int8), 1, kind="stable")[:, : ri.shape[1]]
                n, gaps = route_flips(ri, rs, gi)
                ent["vs_gpu_l1"] = {
                    "tokens_flipped": n,
                    "tokens": int(ri.shape[0]),
                    "gap_buckets": flip_buckets(gaps),
                    "gap_max": max(gaps) if gaps else None,
                }
            dp = os.path.join(a.gpu_route_dump, f"layers.{L}.safetensors") if a.gpu_route_dump else ""
            if dp and os.path.exists(dp):
                gi = safe_open(dp, "pt").get_tensor(f"layers.{L}.mlp.router.topk_idx").numpy().astype(np.int64)
                n, gaps = route_flips(ri, rs, gi)
                ent["vs_gpu_dump"] = {
                    "tokens_flipped": n,
                    "tokens": int(ri.shape[0]),
                    "gap_buckets": flip_buckets(gaps),
                    "gap_max": max(gaps) if gaps else None,
                }
            flips["route"][L] = ent
        if l1 is not None:
            topk = None
            for L in sorted({k.split(".")[2] for k in keys if k.startswith(f"r{row}.dsa.")}):
                rt = ch.get_tensor(f"r{row}.dsa.{L}.topk").numpy().astype(np.int64)[0]
                topk = rt.shape[-1]
                hn = f"hf.model.layers.{L}" if L != "mtp0" else "hf.model.mtp_layers.0"
                gt = l1.get_tensor(f"{hn}.self_attn.topk").numpy()[0]
                qq = ch.get_tensor(f"r{row}.dsa.{L}.q").to(torch.float64)
                kk = ch.get_tensor(f"r{row}.dsa.{L}.k").to(torch.float64)
                ww = ch.get_tensor(f"r{row}.dsa.{L}.w").to(torch.float64)
                ent = {"rows_with_choice": int(rt.shape[0] - topk)}
                n, nel, gaps = dsa_flips(rt, topk_mask(gt), qq, kk, ww)
                ent["vs_gpu_l1"] = {
                    "rows_flipped": n,
                    "elements_swapped": nel,
                    "gap_buckets": flip_buckets(gaps),
                    "gap_max": max(gaps) if gaps else None,
                }
                if tkd is not None and L in tkd_map:
                    m = tkd.get_tensor(tkd_map[L] + "/self_attention/debug_kd_indexer_mask/0").reshape(rt.shape[0], -1)
                    mt = (m == 0).numpy() & np.tri(rt.shape[0], dtype=bool)
                    n, nel, gaps = dsa_flips(rt, mt, qq, kk, ww)
                    ent["vs_tpu"] = {
                        "rows_flipped": n,
                        "elements_swapped": nel,
                        "gap_buckets": flip_buckets(gaps),
                        "gap_max": max(gaps) if gaps else None,
                    }
                flips["dsa"][L] = ent
        flips["gap_definition"] = (
            "FP64 分数里，FP64 选了而对方没选的最高分减去对方选了而 FP64 没选的最低分；"
            "路由除以 FP64 第 k 名的分数（sigmoid + 偏置，约 0.5），DSA 除以这一行索引分数的最大绝对值"
        )
        out["flips_row0"] = flips
    inj = os.path.join(a.ref, "inject")
    row0 = out.get("flips_row0", {}).get("row", 0)
    self_row = os.path.join(a.ref, f"grads_row{row0}_self_unscaled_hf.safetensors")
    if os.path.isdir(inj) and os.path.exists(self_row):
        si = safe_open(self_row, "pt")
        ii = [x for x in os.listdir(inj) if x.endswith("_inject_unscaled_hf.safetensors")][0]
        ij = safe_open(os.path.join(inj, ii), "pt")
        N = refsum["global_tokens"]
        sens = []
        for r in rows:
            h = r["tensor"]
            sl = None
            ex = gexp[0].get(h, {}).get("experts")
            d = ij.get_tensor(h).to(torch.float64) - si.get_tensor(h).to(torch.float64)
            if ex is not None and d.shape[0] != len(ex):
                d = d[torch.tensor(ex)]
            dn = float(d.norm()) / N / r["ref_norm"] if r["ref_norm"] else float("nan")
            sens.append(
                {
                    "tensor": h,
                    "row_flip_effect": dn,
                    "est_full_batch": dn * 4.0,
                    "over_gpu_vs_ref": dn * 4.0 / r["gpu_vs_ref"] if r["gpu_vs_ref"] else None,
                }
            )
        out["inject_sensitivity"] = {
            "what": (
                "第一行用 GPU l1 抓数的选择（DSA top-k 与路由）重跑 FP64，与本侧自选的同一行梯度之差 ÷ 全局 token 数 ÷ ‖FP64‖；"
                "全批估算 × √16 = 4（推断：各行的翻转互相独立）"
            ),
            "median_row_effect": float(np.median([s["row_flip_effect"] for s in sens])),
            "median_est_full_over_gpu_vs_ref": float(
                np.median([s["over_gpu_vs_ref"] for s in sens if s["over_gpu_vs_ref"]])
            ),
            "max_est_full_over_gpu_vs_ref": max(s["over_gpu_vs_ref"] for s in sens if s["over_gpu_vs_ref"]),
            "rows": sens,
        }
        by = {x["tensor"]: x for x in sens}
        for r in rows:
            r["flip_est_over_gpu_vs_ref"] = by[r["tensor"]]["over_gpu_vs_ref"]
        un = new["unattributed_over_2N"] + new["unattributed_1_2N"] + new["unattributed_N0"]
        out["judge_fp64"]["unattributed_flip_estimate"] = {h: by[h]["over_gpu_vs_ref"] for h in un}
        out["inject_sensitivity"]["n_tensors_flip_est_ge_gpu_err"] = sum((x["over_gpu_vs_ref"] or 0) >= 1 for x in sens)
    out["note"] = a.note
    out["rows"] = allrows
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("median", "R_distribution")}, ensure_ascii=False))
    print("JUDGE", json.dumps(new, ensure_ascii=False))
    if old is not None:
        print(
            "OLD",
            json.dumps({k: v for k, v in out["old_reference"].items() if k != "R_crossed_1.2"}, ensure_ascii=False),
        )


if __name__ == "__main__":
    main()
