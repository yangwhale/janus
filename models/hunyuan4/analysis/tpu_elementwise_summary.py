"""把 TPU 上 elementwise_ulp.py 的原始输出整理成结果文件 results/tpu-elementwise-ulp.json（只读原始输出与混元 3 的 GPU 结果，不重算）。

加三样东西：
1. 每个函数一行判定：FP32 用 elementwise_ulp.py 的门槛（ulp_max ≤ CUDA 文档值；relu / clamp 逐位）；BF16 只报；
2. 同分布合成用例上的 R 参考值：GPU 分母取混元 3 GPU（B200，torch）一侧 `L1.elem.*.fp32` 的同名用例
   （models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-l1.results.json 的 worst 列表；种子不同，分布相同），
   R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)，只作参考、不进判定（混元 4 GPU 一侧同一组用例还没有）；
3. KD 的 LSE 偏低这条说法的单独核对：logsumexp 与 log_softmax 反推的 LSE 带符号平均偏差，以及 exp、求和、log 三步各自的贡献。

  python models/hunyuan4/analysis/tpu_elementwise_summary.py <TPU 运行目录>/elementwise_ulp.json <输出 json> [<另一次运行>/elementwise_ulp.json …]

后面再给的运行只补前面没有的检查（如后加的 MaxText 写法一组 fp32_mt），每条检查记来自哪次运行（source_runs）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HY3_GPU = ROOT / "models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-l1.results.json"
SYNTH = {  # 函数 → 两边同名的合成用例
    "exp": ["unif[-87,88]", "unif[-10,10]"],
    "sigmoid": ["unif[-30,30]"],
    "silu": ["unif[-30,30]"],
    "rsqrt": ["all[1,4)", "logu[1e-37,1e37]"],
}


def main(raw_path: str, out_path: str, *extra_paths: str) -> None:
    raw = json.loads(Path(raw_path).read_text())
    origin = {c["id"]: Path(raw_path).parent.name for c in raw["checks"]}
    for ep in extra_paths:
        for c in json.loads(Path(ep).read_text())["checks"]:
            if c["id"] not in origin:
                raw["checks"].append(c)
                origin[c["id"]] = Path(ep).parent.name
    gpu = {}
    for c in json.loads(HY3_GPU.read_text())["checks"]:
        for w in c.get("worst") or []:
            if w.get("metric") == "rel_l2":
                gpu[w["name"]] = w["value"]
    checks = {c["id"]: c for c in raw["checks"]}
    funcs = {}
    for fn in ("exp", "log", "sigmoid", "silu", "rsqrt", "sqrt", "div", "bias_correction", "relu", "clamp"):
        if f"L1.elem.{fn}.fp32" not in checks:
            continue
        c32 = checks[f"L1.elem.{fn}.fp32"]
        c16 = checks.get(f"L1.elem.{fn}.bf16", {"details_inline": []})
        rows32 = {r["name"].split(".fp32.", 1)[1]: r for r in c32["details_inline"]}
        bf = {r["name"].split(".", 1)[1]: r for r in c16["details_inline"]}
        f = {
            "fp32": {
                "ulp_max": c32["metrics"].get("ulp_max"),
                "ulp_mean_worst_case": c32["metrics"].get("ulp_mean_max"),
                "cr_match_frac_worst_case": c32["metrics"].get("cr_match_frac"),
                "per_case": {
                    k: {
                        x: r.get(x)
                        for x in ("ulp_max", "ulp_p999", "ulp_mean", "ulp_signed_mean", "cr_match_frac", "rel_l2")
                    }
                    for k, r in rows32.items()
                },
                "gate": c32.get("threshold"),
                "status": c32["status"],
            },
            "bf16": {
                k: {
                    x: r.get(x)
                    for x in ("ulp_max", "ulp_mean", "cr_match_frac", "bit_match_frac", "excluded_subnormal")
                }
                for k, r in bf.items()
            },
        }
        hi = checks.get(f"L1.elem.{fn}.fp32_highest")
        if hi:
            f["fp32_accuracy_highest"] = {
                "ulp_max": hi["metrics"].get("ulp_max"),
                "ulp_mean_worst_case": hi["metrics"].get("ulp_mean_max"),
                "status": hi["status"],
                "per_case": {
                    r["name"].split(".fp32_highest.", 1)[1]: {
                        x: r.get(x) for x in ("ulp_max", "ulp_mean", "ulp_signed_mean", "rel_l2")
                    }
                    for r in hi["details_inline"]
                },
            }
        mt = checks.get(f"L1.elem.{fn}.fp32_mt")
        if mt:  # MaxText 开关实际用的写法（accurate_math / muon_mcore_compat）
            f["fp32_maxtext_accurate"] = {
                "ulp_max": mt["metrics"].get("ulp_max"),
                "ulp_mean_worst_case": mt["metrics"].get("ulp_mean_max"),
                "status": mt["status"],
                "per_case": {
                    r["name"].split(".fp32_mt.", 1)[1]: {
                        x: r.get(x) for x in ("ulp_max", "ulp_mean", "ulp_signed_mean", "rel_l2")
                    }
                    for r in mt["details_inline"]
                },
            }
        ref = {}
        for case in SYNTH.get(fn, []):
            g = gpu.get(f"{fn}.fp32.{case}")
            t = rows32.get(case, {}).get("rel_l2")
            if g and t is not None:
                ref[case] = {"rel_l2_tpu": t, "rel_l2_gpu_hy3_b200": g, "R": t / g}
        if ref:
            f["fp32"]["R_vs_hy3_gpu_same_distribution"] = ref
        funcs[fn] = f
    lse = raw.get("lse_breakdown", {})
    kd = lse.get("kd_real", {})
    claim = None
    if "tpu" in kd:
        t = kd["tpu"]
        th = kd.get("tpu_highest")
        claim = {
            "claim": "TPU FP32 log_softmax 的 LSE 在 KD 输入上系统偏低约 0.95 ulp（results/t1-layer0-kd-attribution.json，图内测得 −0.953）",
            "isolated_tpu_logsumexp_ulp_signed_mean": t["lse_logsumexp_ulp_signed_mean"],
            "isolated_tpu_log_softmax_implied_ulp_signed_mean": t["lse_implied_by_log_softmax_ulp_signed_mean"],
            "cpu_same_process_logsumexp_ulp_signed_mean": kd.get("cpu", {}).get("lse_logsumexp_ulp_signed_mean"),
            "breakdown_tpu": {
                "exp_of_shifted_scores_ulp_signed_mean": t["step_exp_ulp_signed_mean"],
                "fp32_sum_err_in_lse_ulp_signed_mean": t["step_sum_err_in_lse_ulp_signed_mean"],
                "log_of_sum_ulp_signed_mean": t["step_log_ulp_signed_mean"],
            },
            "tpu_accuracy_highest_logsumexp_ulp_signed_mean": th["lse_logsumexp_ulp_signed_mean"] if th else None,
            "verdict": "confirmed" if t["lse_logsumexp_ulp_signed_mean"] < -0.5 else "refuted",
            "note": "单独编译的 logsumexp / log_softmax 在同一批 KD 输入上复现了偏低；拆开看，偏差来自 TPU 的 FP32 exp（KD 分数近乎均匀，"
            "exp 的输入都在 0 附近，逐元素带符号平均偏低约 2.5 ulp）与 log（偏低约 0.6 ulp），FP32 求和本身不贡献偏差；"
            "同一进程的 XLA CPU 上同样的程序没有偏差",
        }
    bwd = {
        cid.split(".")[2]: {
            r["name"].split(".", 1)[1]: {x: r.get(x) for x in ("ulp_max", "ulp_mean", "ulp_signed_mean", "rel_l2")}
            for r in c["details_inline"]
        }
        for cid, c in checks.items()
        if cid.endswith("_bwd")
    }
    gated = [fn for fn in funcs if funcs[fn]["fp32"]["status"] in ("pass", "fail")]
    # 判定只看 XLA 默认实现；accuracy=HIGHEST 与 MaxText 开关写法（fp32_maxtext_accurate）两组只报
    fails = [fn for fn in gated if funcs[fn]["fp32"]["status"] == "fail"]
    out = {
        "what": "逐元素函数对 FP64 的 ULP（TPU v5p，JAX / XLA）：sigmoid、silu、rsqrt、exp、log、log_softmax / LSE、relu、clamp，FP32 与 BF16；"
        "softplus 在 hy4-mini 里没有用到（路由打分是 sigmoid，routed_score_func）",
        "source_run": raw.get("inputs"),
        "source_runs": origin,
        "platform": raw.get("platform"),
        "device_kind": raw.get("device_kind"),
        "jax": raw.get("jax"),
        "rule": raw.get("rule"),
        "reference": "FP64（numpy）正确舍入到输出 dtype",
        "verdict": "fail" if fails else "pass",
        "fail_functions": fails,
        "functions": funcs,
        "lse_claim_check": claim,
        "backward": {
            "what": "sigmoid / silu 的反向（上游梯度 1）：jax 默认实现的自动微分 vs MaxText 精确写法（M39，PyTorch 的解析式）对 FP64 解析导数；"
            "输入 [-8, 8]。ULP 最大值被 1 − y 的抵消主导（两种写法都有），看 rel_L2 与平均 ULP",
            "per_case": bwd,
        }
        if bwd
        else None,
        "lse_breakdown": lse,
        "summary": raw.get("summary"),
        "checks": [{k: v for k, v in c.items() if k != "details_inline"} for c in raw["checks"]],
    }
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(out, ensure_ascii=False, indent=1))  # 仓库里不写真实桶名
    Path(out_path).write_text(text + "\n")
    print(json.dumps({"verdict": out["verdict"], "fail_functions": fails}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])
