"""汇总 06-rw3 第 1 步索引器 KD 的 GPU 偏差定位：各工具的输出 → 一份结果文件（数字都从输入文件里取，不手写）。

输入（`--in-dir` 下，各工具的原样输出）：
- `kd_input_rounding_r0.json`（`kd_input_rounding.py`，第 0 行，只在 KD 一段加舍入）
- `kd_gpu_dump_vs_fp64.json`（`kd_gpu_dump_vs_fp64.py`，GPU 实抓的第 0 行输入逐个换进 FP64）
- `kd_cpu16_vs_gpu_dump.json`（`kd_cpu16_vs_gpu_dump.py`，CPU BF16 前向对 GPU 实抓）
- `kd_noise_scaling.json`（`kd_noise_scaling.py`，随机噪声的响应）
- `bf16v64/rows.jsonl`（`kd_bf16_vs_fp64.py`，逐行 FP64 参照 + CPU BF16 前向与各组合）
- `kpe/summary.json`（可选，`kpe_round_emul.py`：各注意力层 k_pe 舍一次 BF16 的主 loss / MTP loss）
另：GPU 8 份 golden 第 1 步的 metrics（`--gpu-metrics` 给每份的 metrics.jsonl），FP64 参照的 summary.json（`--fp64`）。

用法：
    python kd_gpu_bias_summary.py --in-dir <目录> --fp64 <summary.json> --gpu-metrics a/metrics.jsonl b/metrics.jsonl ... \
        --out models/hunyuan4/alignments/06-rw3/results/step1-kd-gpu-bias.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def rel(a, b):
    return a / b - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--fp64", required=True)
    p.add_argument("--gpu-metrics", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from kd_bf16_vs_fp64 import summarize

    d = a.in_dir
    fp = json.load(open(a.fp64))
    ref = dict(fp["indexer_loss_per_layer"])
    ref["all"] = fp["indexer_loss"]
    out = {
        "what": "06-rw3 第 1 步索引器 KD 对 CPU FP64 的 GPU 偏差：逐层、逐输入、逐算子的拆分",
        "fp64_ref": {"indexer_loss": fp["indexer_loss"], "per_layer": fp["indexer_loss_per_layer"]},
    }
    gpu = {}
    for path in a.gpu_metrics:
        m = json.loads(open(path).readline())
        assert m["step"] == 1
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        g = {
            "0": m["indexer_loss.0"],
            "1": m["indexer_loss.1"],
            "mtp0": m["indexer_loss.mtp0"],
            "all": m["indexer_loss"],
        }
        gpu[name] = {k: rel(g[k], ref[k]) for k in g}
        # 各层对总量相对差的贡献（绝对差 ÷ FP64 总 KD）
        gpu[name]["contrib_to_all"] = {k: (g[k] - ref[k]) / ref["all"] for k in ("0", "1", "mtp0")}
    out["gpu_step1_rel_vs_fp64"] = gpu
    out["kd_local_rounding_row0"] = {
        L: {v: r.get("rel_vs_fp64") for v, r in lay.items() if v != "fp64"}
        for L, lay in json.load(open(os.path.join(d, "kd_input_rounding_r0.json")))["layers"].items()
    }
    gd = json.load(open(os.path.join(d, "kd_gpu_dump_vs_fp64.json")))
    out["gpu_dump_row0"] = {
        L: {"rel_vs_ref": lay["rel_vs_ref"], "rel_l2_gpu": lay["rel_l2_gpu"], "topk_overlap": lay["topk_overlap"]}
        for L, lay in gd["layers"].items()
    }
    cg = json.load(open(os.path.join(d, "kd_cpu16_vs_gpu_dump.json")))
    out["cpu_bf16_vs_gpu_dump_row0"] = cg["layers"]
    ns = json.load(open(os.path.join(d, "kd_noise_scaling.json")))
    out["noise_scaling_row0"] = [
        {k: r[k] for k in ("layer", "target", "sigma", "mean", "mean_over_sigma2")} | {"rel_shift": r["rel_shift"]}
        for r in ns["runs"]
    ]
    sm = summarize(os.path.join(d, "bf16v64"))
    out["cpu_bf16_factorial"] = sm
    kp = os.path.join(d, "kpe", "summary.json")
    if os.path.exists(kp):
        k = json.load(open(kp))
        out["kpe_round_all_layers"] = {
            "rows": len(k["rows"]),
            "lm_loss_rel_vs_fp64": rel(k["lm_loss"], fp["lm_loss"]),
            "mtp_loss_rel_vs_fp64": rel(k["mtp_loss"], fp["mtp_loss"]),
            "lm_loss": k["lm_loss"],
            "mtp_loss": k["mtp_loss"],
        }
    with open(a.out, "w") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)
    print(
        json.dumps(
            {
                "all": sm["all"].get("bf16run"),
                "per_layer": {L: v["rel_vs_ref"].get("bf16run") for L, v in sm["per_layer"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
