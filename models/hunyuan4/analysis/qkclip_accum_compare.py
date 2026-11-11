"""低阈值 QK-Clip 真触发时，TPU「梯度累积 2」对「不累积」（每设备微批 2，全局批不变）：逐步逐层逐头的最大 logit、
缩放 η、被裁头数与主 loss；尺子是 GPU 两份同设置重跑（只差 GPU 自己的求和顺序）之间同样的差。另把第 1 步的逐头
最大 logit 对 CPU FP64 参照（`t1zo-qkclip-maxlogit.results.json` 里存的 FP64 值，同一份第 1 步数据）算 R。

    python models/hunyuan4/analysis/qkclip_accum_compare.py \\
        --ga2 <累积 2 的 metrics.jsonl> --ga1 <不累积的 metrics.jsonl> \\
        --gpu-a <GPU 第一份 metrics.jsonl> --gpu-b <GPU 第二份 metrics.jsonl> \\
        --fp64 models/hunyuan4/alignments/01-mini/results/t1zo-qkclip-maxlogit.results.json \\
        --tau 0.1 --names ga2=…,ga1=…,gpu=…/… --out <json>

判读：梯度累积把最大 logit 求和（M19 修之前的 bug）时，第 1 步累积 2 的最大 logit 约为不累积的 2 倍；只差求和顺序时，
两者之差与 GPU 两份重跑之差同一量级。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

LAYERS = ["0", "1", "2", "3", "4", "5", "6", "mtp0"]


def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def heads(row, kind):
    return np.array([row[f"qk.{kind}.{layer}"] for layer in LAYERS], dtype=np.float64)


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def pair_series(a, b, n):
    """逐步：最大 logit 的逐头最大相对差与整体相对 L2、η 的逐头最大相对差、被裁头数之差、主 loss 的相对差。"""
    out = []
    for s in range(n):
        ma, mb = heads(a[s], "max_logit"), heads(b[s], "max_logit")
        ea, eb = heads(a[s], "eta"), heads(b[s], "eta")
        la, lb = a[s]["lm_loss"], b[s]["lm_loss"]
        out.append(
            {
                "step": s + 1,
                "max_logit_max_abs_rel": float(np.max(np.abs(ma - mb) / np.abs(ma))),
                "max_logit_rel_l2": rel_l2(mb, ma),
                "max_logit_ratio_max": float(np.max(ma / mb)),
                "max_logit_bitwise_equal": int((ma == mb).sum()),
                "eta_max_abs_rel": float(np.max(np.abs(ea - eb) / ea)),
                "clipped": [int(a[s]["qk.clipped_total"]), int(b[s]["qk.clipped_total"])],
                "lm_loss_rel": float((lb - la) / la),
            }
        )
    return out


def summarize(series, lo, hi):
    sel = [r for r in series if lo <= r["step"] <= hi]
    return {
        "window": f"{lo}:{hi + 1}",
        "max_logit_max_abs_rel": {
            "max": max(r["max_logit_max_abs_rel"] for r in sel),
            "median": float(np.median([r["max_logit_max_abs_rel"] for r in sel])),
        },
        "max_logit_rel_l2": {
            "max": max(r["max_logit_rel_l2"] for r in sel),
            "median": float(np.median([r["max_logit_rel_l2"] for r in sel])),
        },
        "eta_max_abs_rel_max": max(r["eta_max_abs_rel"] for r in sel),
        "lm_loss_abs_rel": {
            "max": max(abs(r["lm_loss_rel"]) for r in sel),
            "mean": float(np.mean([abs(r["lm_loss_rel"]) for r in sel])),
        },
        "steps_clipped_count_differs": sum(r["clipped"][0] != r["clipped"][1] for r in sel),
        "steps_with_clipping": sum(r["clipped"][0] > 0 for r in sel),
        "clipped_heads_total": [sum(r["clipped"][0] for r in sel), sum(r["clipped"][1] for r in sel)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ga2", required=True)
    p.add_argument("--ga1", required=True)
    p.add_argument("--gpu-a", required=True)
    p.add_argument("--gpu-b", required=True)
    p.add_argument(
        "--fp64", required=True, help="t1zo-qkclip-maxlogit.results.json（第 1 步逐头 FP64 值与 GPU 中位数）"
    )
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--names", required=True, help="ga2=<运行名>,ga1=<运行名>,gpu=<a>/<b>")
    p.add_argument("--stage", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    names = dict(kv.split("=", 1) for kv in a.names.split(","))
    ga2, ga1, ga, gb = load(a.ga2), load(a.ga1), load(a.gpu_a), load(a.gpu_b)
    n = min(len(ga2), len(ga1))
    ng = min(len(ga), len(gb))
    tpu = pair_series(ga2, ga1, n)
    gpu = pair_series(ga, gb, ng)

    # 第 1 步对 FP64：只有 τ 0.1 的第 1 步数据与 FP64 参照同源（第 1 步的前向不受 τ 影响，任何 τ 都可用）
    ref = json.load(open(a.fp64))
    rows = ref["checks"][0]["details_inline"]["rows"]
    step1 = []
    worst = None
    for r in rows:
        layer = r["layer"]
        f64 = np.array(r["fp64"])
        x2 = np.array(ga2[0][f"qk.max_logit.{layer}"])
        x1 = np.array(ga1[0][f"qk.max_logit.{layer}"])
        g = r["gpu_median"]
        e = {
            "layer": layer,
            "rel_l2_ga2_vs_fp64": rel_l2(x2, f64),
            "rel_l2_ga1_vs_fp64": rel_l2(x1, f64),
            "gpu_median_vs_fp64": g,
            "R_ga2": rel_l2(x2, f64) / g if g > 0 else None,
            "R_ga1": rel_l2(x1, f64) / g if g > 0 else None,
            "rel_l2_ga1_vs_ga2": rel_l2(x1, x2),
        }
        step1.append(e)
        if worst is None or (e["R_ga1"] or 0) > (worst["R_ga1"] or 0):
            worst = e
    # 整体（8 层拉直）对 FP64
    f_all = np.concatenate([np.array(r["fp64"]) for r in rows])
    x2_all = np.concatenate([np.array(ga2[0][f"qk.max_logit.{r['layer']}"]) for r in rows])
    x1_all = np.concatenate([np.array(ga1[0][f"qk.max_logit.{r['layer']}"]) for r in rows])

    res = {
        "schema_version": 1,
        "stage": a.stage,
        "model": "hunyuan4-mini",
        "inputs": {
            "tpu_ga2": names.get("ga2"),
            "tpu_ga1": names.get("ga1"),
            "gpu_pair": names.get("gpu"),
            "tau": a.tau,
            "fp64": "t1zo-qkclip-maxlogit.results.json（第 1 步）",
            "note": "累积 2 = 每设备微批 1 × 累积 2（现行）；累积 1 = 每设备微批 2 × 累积 1；全局批 16、其余设置相同，回放同一份 GPU batch",
        },
        "step1": {
            "ratio_ga2_over_ga1_max": float(np.max(x2_all / x1_all)),
            "ratio_ga2_over_ga1_min": float(np.min(x2_all / x1_all)),
            "bitwise_equal_heads": int((x2_all == x1_all).sum()),
            "heads": int(x2_all.size),
            "rel_l2_all_ga2_vs_fp64": rel_l2(x2_all, f_all),
            "rel_l2_all_ga1_vs_fp64": rel_l2(x1_all, f_all),
            "per_layer": step1,
            "worst_layer_by_R_ga1": worst["layer"],
        },
        "tpu_ga1_vs_ga2": {
            "steps": n,
            "first_30": summarize(tpu, 1, min(30, n)),
            "all": summarize(tpu, 1, n),
            "series": tpu,
        },
        "gpu_rerun_pair": {"steps": ng, "first_30": summarize(gpu, 1, min(30, ng)), "series": gpu},
    }
    t30, g30 = res["tpu_ga1_vs_ga2"]["first_30"], res["gpu_rerun_pair"]["first_30"]
    res["ratio_to_gpu_pair_first_30"] = {
        "max_logit_max_abs_rel_max": t30["max_logit_max_abs_rel"]["max"] / g30["max_logit_max_abs_rel"]["max"],
        "max_logit_rel_l2_median": t30["max_logit_rel_l2"]["median"] / g30["max_logit_rel_l2"]["median"],
        "lm_loss_abs_rel_mean": t30["lm_loss_abs_rel"]["mean"] / g30["lm_loss_abs_rel"]["mean"],
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("step1", "ratio_to_gpu_pair_first_30")}, ensure_ascii=False)[:2000])
    print(json.dumps({"tpu30": t30, "tpu_all": res["tpu_ga1_vs_ga2"]["all"], "gpu30": g30}, ensure_ascii=False))


if __name__ == "__main__":
    main()
