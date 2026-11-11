"""核 TPU 训练第 1 步全量抓数（`--dump-steps 1 --full` 的 step_1/）有没有落盘损坏，并估「由动量反推梯度」的舍入误差。

`export_train_grads.py` 由 step_1/ 的动量反推训练第 1 步实际用的梯度（Adam g = m / (1 − β1)，Muon g = buf / (1 − β)，
再乘回裁剪系数）。诊断程序的梯度文件有 `diag_step1.json` 的设备上逐张量平方和可核（`step1_vs_fp64.py --tpu-diag`），
全量抓数没有逐张量的设备统计，这里换成四条核对：

1. sha256：index.json 里每个张量的 sha256 是 Pod 上写文件时（janus.store.write_tensors，主机内存里的数组）算的；
   逐个重算比对。已知的损坏方式（从 Pod 分块取回时流被截断，拼好的文件里留下整段 0，PROTOCOL 变更记录 2026-11-08）
   改的是写完之后的字节，这一条逐位能抓到。
2. 全局平方和：反推出的全部梯度平方和开方，对设备上算的 raw_grad_norm（metrics.jsonl 第 1 步）。管的是写文件之前
   （设备取回主机）那一段；只能抓到占总平方和够大的损坏（门限见结果里的 detect_floor）。
3. 零的位置对 GPU：同一批输入下 GPU 第 1 步梯度不是 0、TPU 反推的梯度恰好是 0 的元素，逐张量数个数和最长连续段
   （按存储顺序展平；路由专家只比 GPU 导出里有的那几个）。孤立的 0 不算损坏：两边各自会把很小的值算成恰好 0
   （结果里记了这些位置 GPU 的值有多大）；连续 ≥ SEG_MIN 个才记为「整段写成 0」。词嵌入另核全 0 的行（这一步
   没出现的 token）两边是不是同一批。
4. Adam 参数的 m 与 v 零的位置一致：m = 0 而 v ≠ 0（或反过来）的元素数。

舍入误差：Adam 参数另有一条独立的路——|g| = sqrt(v / (1 − β2))，与 |m| / (1 − β1) 逐元素比，相对差的分布直接量出
两条路合起来的舍入（单位 u = 2^-24，FP32 的单位舍入）；另写解析上界。

    python models/hunyuan4/analysis/step1_train_grad_file_check.py --run <TPU 运行目录> \\
        [--gpu g2s-a=<GPU 第 1 步梯度 safetensors>] --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpu_self_compare as tsc  # noqa: E402

U = 2.0**-24  # FP32 单位舍入
SEG_MIN = 16  # 连续这么多个「TPU 为 0、GPU 不为 0」才算整段写零（64 字节；已知的截断损坏是成块的）
B2 = 0.95  # Adam β2（T1：adam_b2 0.95，config-alignment.md「标量参数的 Adam」）


def longest_true_run(mask: np.ndarray) -> int:
    """一维布尔数组里最长的连续 True 段长度。"""
    m = np.concatenate(([False], mask.ravel(), [False])).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max()) if starts.size else 0


def sha_check(run: str) -> dict:
    import torch
    from safetensors import safe_open

    sdir = os.path.join(run, "step_1")
    idx = json.load(open(os.path.join(sdir, "index.json")))["tensors"]
    handles, bad = {}, []
    for e in idx:
        f = handles.setdefault(e["file"], safe_open(os.path.join(sdir, e["file"]), "pt"))
        x = f.get_tensor(e["name"])
        raw = (
            (x.view(torch.int16) if x.dtype == torch.bfloat16 else x).contiguous().numpy()
        )  # 同 tpu_self_compare.Dump.get
        if hashlib.sha256(raw.reshape(-1).view(np.uint8)).hexdigest() != e["sha256"]:
            bad.append(e["name"])
    return {
        "n_tensors": len(idx),
        "n_files": len({e["file"] for e in idx}),
        "mismatch": bad,
        "rule": "index.json 的 sha256（Pod 上写文件时由主机内存里的数组算）对本地文件重算的 sha256，逐位",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="TPU 运行目录（含 step_1/、metrics.jsonl、meta.json）")
    p.add_argument(
        "--gpu", default="", help="名字=GPU 第 1 步梯度 safetensors（HF 名字，旁边的 .json / .sidecar.json 列路由专家）"
    )
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    run = os.path.normpath(a.run)
    meta_sha = hashlib.sha256(open(os.path.join(run, "meta.json"), "rb").read()).hexdigest()
    with open(os.path.join(run, "metrics.jsonl")) as f:
        m1 = next(json.loads(line) for line in f if json.loads(line)["step"] == 1)
    raw_gn, gn_clipped = float(m1["raw_grad_norm"]), m1.get("grad_norm_clipped")
    clip = max(1.0, raw_gn)

    sha = sha_check(run)

    gpu, gexp, gname = None, {}, None
    if a.gpu:
        from safetensors import safe_open

        gname, gpath = a.gpu.split("=", 1)
        gpu = safe_open(gpath, "pt")
        for side in (os.path.splitext(gpath)[0] + ".json", os.path.splitext(gpath)[0] + ".sidecar.json"):
            if os.path.exists(side):
                gexp = json.load(open(side)).get("tensors", {})
                break

    d = tsc.Dump(run)
    names = sorted(n for (k, n) in d.idx if k == "exp_avg")
    f32 = np.float32
    c_fp64 = {"adam": 1 - tsc.B1, "muon": 1 - tsc.BETA}  # export_train_grads.py / tpu_self_compare 用的常数
    c_fp32 = {"adam": float(f32(1) - f32(tsc.B1)), "muon": float(f32(1) - f32(tsc.BETA))}  # 若设备用 FP32 常数
    sq = {"adam": 0.0, "muon": 0.0}
    per_tensor, rel_u = {}, []
    for n in names:
        kind = "adam" if d.has("exp_avg_sq", n) else "muon"
        m = d.get("exp_avg", n)  # FP32 存储，读成 FP64
        sq[kind] += float((m * m).sum())
        g = m / c_fp64[kind] * clip  # 与 export_train_grads.py 同一式子
        row = {"opt": kind, "numel": int(g.size), "tpu_zeros": int((g == 0).sum())}
        if kind == "adam":
            v = d.get("exp_avg_sq", n)
            row["adam_m_v_zero_mismatch"] = int(((m == 0) != (v == 0)).sum())
            nz = (m != 0) & (v > 0)
            if nz.any():
                g_m = np.abs(m[nz]) / c_fp64["adam"]
                g_v = np.sqrt(v[nz] / (1 - B2))
                rel_u.append(np.abs(g_m - g_v) / g_v / U)
        if gpu is not None and n in gpu.keys():
            gg = gpu.get_tensor(n).double().numpy()
            ex = gexp.get(n, {}).get("experts")
            gt = g[np.array(ex)] if ex is not None and g.shape != gg.shape else g
            if gt.shape != gg.shape:
                row["gpu_shape_mismatch"] = [list(gt.shape), list(gg.shape)]
            else:
                bad = (gt == 0) & (gg != 0)
                row["gpu_zeros"] = int((gg == 0).sum())
                row["tpu_zero_gpu_nonzero"] = int(bad.sum())
                row["tpu_zero_gpu_nonzero_longest_run"] = longest_true_run(bad)
                row["tpu_nonzero_gpu_zero"] = int(((gt != 0) & (gg == 0)).sum())
                row["compared_experts"] = ex
                if bad.any():
                    row["gpu_abs_at_tpu_zero_max"] = float(np.abs(gg[bad]).max())
                    row["gpu_abs_median_nonzero"] = float(np.median(np.abs(gg[gg != 0])))
                if gt.ndim == 2 and n == "model.embed_tokens.weight":
                    tz, gz = set(np.flatnonzero((gt == 0).all(1))), set(np.flatnonzero((gg == 0).all(1)))
                    row["zero_rows"] = {"tpu": len(tz), "gpu": len(gz), "symmetric_diff": len(tz ^ gz)}
        row["corrupt_suspect"] = bool(
            row.get("adam_m_v_zero_mismatch", 0) > 0
            or row.get("tpu_zero_gpu_nonzero_longest_run", 0) >= SEG_MIN
            or row.get("zero_rows", {}).get("symmetric_diff", 0) > 0
            or n in sha["mismatch"]
        )
        per_tensor[n] = row

    tot = {
        lab: (sq["adam"] / c["adam"] ** 2 + sq["muon"] / c["muon"] ** 2) ** 0.5
        for lab, c in (("fp64", c_fp64), ("fp32", c_fp32))
    }
    rel_glob = {lab: (x * clip - raw_gn) / raw_gn for lab, x in tot.items()}
    floor = 2 * max(abs(x) for x in rel_glob.values())
    r = np.concatenate(rel_u) if rel_u else np.array([np.nan])
    suspects = sorted(n for n, x in per_tensor.items() if x["corrupt_suspect"])
    sha_bad = sha["mismatch"]
    out = {
        "schema_version": 1,
        "stage": "t1zo-step1-train-grad-file-check",
        "model": "hunyuan4-mini",
        "catalog": {
            "path": "models/hunyuan4/analysis/step1_train_grad_file_check.py",
            "sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        },
        "what": f"TPU {os.path.basename(run)} 第 1 步全量抓数（反推训练步梯度的来源）的落盘损坏核对与反推舍入估计",
        "inputs": {
            "run": os.path.basename(run),
            "meta_sha256": meta_sha,
            "gpu": gname,
            "raw_grad_norm": raw_gn,
            "grad_norm_clipped": gn_clipped,
            "clip_factor": clip,
            "n_grad_tensors": len(names),
            "n_adam": sum(x["opt"] == "adam" for x in per_tensor.values()),
            "n_muon": sum(x["opt"] == "muon" for x in per_tensor.values()),
        },
        "sha256": sha,
        "global_norm": {
            "rule": "反推的全部梯度平方和开方 ÷ 设备上算的 raw_grad_norm − 1；两种常数各算一次（Python FP64 的 1 − β，"
            "export_train_grads.py 所用；与 FP32 里算的 1 − β）",
            "reconstructed_raw_grad_norm": {k: v * clip for k, v in tot.items()},
            "reconstructed_clipped_norm": tot,
            "rel_vs_device_raw_grad_norm": rel_glob,
            "detect_floor_sq_fraction": floor,
            "detect_floor_note": "一段被置零、占总平方和比例 f 时，范数相对少约 f / 2；f 大于 detect_floor_sq_fraction"
            "（两倍于现在的最大相对差，换算成平方和比例）才能从这一条看出来，小张量的损坏靠第 3、4 条",
        },
        "rounding": {
            "unit": "u = 2^-24（FP32 单位舍入）",
            "analytic": "设备一侧：裁剪 g·s（s ≈ 1 / raw_grad_norm）舍一次、乘 (1 − β) 舍一次、s 本身舍一次，逐元素相对误差 ≤ 3u"
            "（≈ 1.8e-7）；反推在 FP64 里做，可忽略。另有一处整体缩放：若设备的 1 − β 是 FP32 里算的，"
            "与反推用的 FP64 常数差 (1 − β)_fp32 ÷ (1 − β)_fp64 − 1，见 const_scale_rel；整张量同一个倍数，"
            "不改方向，rel_L2 对 FP64 最多跟着变这么多（相对）",
            "const_scale_rel": {k: c_fp32[k] / c_fp64[k] - 1 for k in ("adam", "muon")},
            "empirical_adam": {
                "rule": "Adam 参数逐元素 | |m| / (1 − β1) − sqrt(v / (1 − β2)) | ÷ sqrt(v / (1 − β2))，单位 u；"
                "两条路各自的舍入合起来（v 那条多一次平方的舍入），是反推误差的上界量级",
                "n_elements": int(r.size),
                "q50_u": float(np.nanquantile(r, 0.5)),
                "q99_u": float(np.nanquantile(r, 0.99)),
                "q999_u": float(np.nanquantile(r, 0.999)),
                "max_u": float(np.nanmax(r)),
            },
        },
        "tpu_file_check": {
            "checked": True,
            "rule": "全量抓数逐张量 sha256 对 Pod 上写文件时记的值；Adam 参数 m 与 v 零的位置一致；"
            f"反推梯度为 0 而 GPU（{gname}）同位置不为 0 的元素没有连续 ≥ {SEG_MIN} 个的段；词嵌入全 0 的行两边相同；"
            "任一条不过记为落盘损坏、不可判",
            "corrupt": suspects,
            "sha256_mismatch": sha_bad,
        },
        "zero_pattern_summary": {
            "seg_min": SEG_MIN,
            "n_tensors_compared": sum("tpu_zero_gpu_nonzero" in x for x in per_tensor.values()),
            "n_tensors_with_isolated_tpu_zero": sum(x.get("tpu_zero_gpu_nonzero", 0) > 0 for x in per_tensor.values()),
            "total_tpu_zero_gpu_nonzero": sum(x.get("tpu_zero_gpu_nonzero", 0) for x in per_tensor.values()),
            "max_longest_run": max(
                (x.get("tpu_zero_gpu_nonzero_longest_run", 0) for x in per_tensor.values()), default=0
            ),
            "max_gpu_abs_over_median_at_tpu_zero": max(
                (
                    x["gpu_abs_at_tpu_zero_max"] / x["gpu_abs_median_nonzero"]
                    for x in per_tensor.values()
                    if "gpu_abs_at_tpu_zero_max" in x
                ),
                default=None,
            ),
            "adam_m_v_zero_mismatch_total": sum(x.get("adam_m_v_zero_mismatch", 0) for x in per_tensor.values()),
            "embed_zero_rows": per_tensor.get("model.embed_tokens.weight", {}).get("zero_rows"),
        },
        "per_tensor": per_tensor,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(
        f"sha256 {sha['n_tensors']} 个张量不符 {len(sha_bad)}；全局范数相对差 {rel_glob}；"
        f"可疑张量 {len(suspects)}；Adam 反推舍入 q50 {out['rounding']['empirical_adam']['q50_u']:.3g}u、"
        f"max {out['rounding']['empirical_adam']['max_u']:.3g}u"
    )
    for n in suspects:
        print("  ", n, per_tensor[n])


if __name__ == "__main__":
    main()
