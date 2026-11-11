"""低阈值 QK-Clip（τ 0.1，30 步）TPU 对 GPU：逐层逐头的最大 logit / η、被裁的头、裁剪后的 q / kv 上投影 FP32 主权重。

输入（都是本地目录）：
- TPU 运行 `--tpu`（train_hy4.py，MaxText `qk_clip_log_per_head`：metrics.jsonl 里 qk.max_logit.<层> / qk.eta.<层>；
  第 1–5 步全量抓数只留 q_b_proj / kv_b_proj：更新前的 FP32 主权重 param.* 与更新（含裁剪）后的 main_param_after.*）；
- TPU 的 FP32 参照 `--tpu-ref`（同一程序 dtype=float32、matmul_precision=highest 跑第 1 步，只用它的逐头最大 logit）；
- GPU 两份 `--gpu-a`、`--gpu-b`（g6-qk-a / b：metrics.jsonl 同名列；qk_clip/step_<N>.safetensors 存各层裁剪前后的主权重与 BF16 权重）。

四项：
1. 裁剪的算法（判定，R 之外的「公式」项）：TPU 裁剪后的主权重 ≟ GPU 裁剪前（已含这一步的优化器更新）的主权重 × TPU 自己记下的 η
   按 Megatron-Core 的行规则（q 的 nope 行 × η^α、rope 行 × η，k 的 nope 行 × η^(1−α)，v 行不动，α 0.5）。
   残差 / 裁剪本身的改变量（‖GPU 裁剪后 − GPU 裁剪前‖）只剩两边这一步优化器更新的差；对照 GPU 自己的同一量（≈ 1e-6）。
   过线：≤ 1e-3（推断：第 1 步一次更新约 lr·O(1) = 1e-5 量级、裁剪改变量约 (1−η)·|W| ≈ 1e-1·6e-3，比值 ~1e-4；取 10 倍余量）；
2. 第 1 步逐层逐头最大 logit（= η 的分母）对 TPU FP32 参照的 R = rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照)（≤ 1.2 过）；
   注意这是整网前向的量（第 k 层的输入带着前 k 层的差），不是隔离的单算子；
3. 被裁的头（η < 1）：30 步里逐步逐头的集合差，TPU 对 GPU-a，对照 GPU-b 对 GPU-a（噪声）；只报；
4. 已知语义差：GPU 另把 BF16 模型权重单独按 bf16(W_bf16 · 因子) 裁（不是 bf16(裁剪后主权重)），下一步前向用它；
   MaxText 只有 FP32 主权重。量出两者的差（GPU 抓数里），并看 30 步 lm_loss 是否在 GPU 两份的散布内；只报。

  python models/hunyuan4/analysis/qk_clip_compare.py --tpu <tw-qk-a> --tpu-ref <tw-qk-ref> --gpu-a <g6-qk-a> --gpu-b <g6-qk-b> --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

H, NOPE, PE, V, ALPHA, TAU = 8, 192, 64, 256, 0.5, 0.1
LABS = ["0", "1", "2", "3", "4", "5", "6", "mtp0"]
FORMULA_LIMIT = 1e-3
R_LIMIT = 1.2


def metrics(d: str) -> dict:
    return {int(json.loads(x)["step"]): json.loads(x) for x in Path(d, "metrics.jsonl").read_text().splitlines() if x.strip()}


def st(path: str) -> dict:
    import torch  # noqa: F401  (safetensors 的 pt 读法要 torch：GPU 文件里有 BF16)
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return {k: f.get_tensor(k).float().numpy().astype(np.float64) for k in f.keys()}


def factors(eta, proj: str) -> np.ndarray:
    eta = np.asarray(eta, np.float64)
    if proj == "q":
        f = np.concatenate([(eta**ALPHA)[:, None] * np.ones((H, NOPE)), eta[:, None] * np.ones((H, PE))], 1)
    else:
        f = np.concatenate([(eta ** (1 - ALPHA))[:, None] * np.ones((H, NOPE)), np.ones((H, V))], 1)
    return f.reshape(-1, 1)


def hf_name(L: str, proj: str) -> str:
    base = "hf.model.mtp_layers.0" if L == "mtp0" else f"hf.model.layers.{L}"
    return f"{base}.self_attn.{'q_b_proj' if proj == 'q' else 'kv_b_proj'}.weight"


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--tpu", "--tpu-ref", "--gpu-a", "--gpu-b", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--formula-steps", default="1", help="做公式核对的步（TPU 与 GPU 更新前的权重只在第 1 步相同）")
    a = p.parse_args(argv)
    T, F, A, B = (metrics(d) for d in (a.tpu, a.tpu_ref, a.gpu_a, a.gpu_b))
    steps = sorted(set(T) & set(A) & set(B))

    # 1. 裁剪的算法
    formula = []
    for s in [int(x) for x in a.formula_steps.split(",")]:
        g = st(f"{a.gpu_a}/qk_clip/step_{s}.safetensors")
        t = {}
        for fn in glob.glob(f"{a.tpu}/step_{s}/main_param_after.*.safetensors"):
            t.update(st(fn))
        for L in LABS:
            for proj, mg in (("q", "linear_q_up_proj"), ("kv", "linear_kv_up_proj")):
                gb, ga = g[f"{L}.{mg}.main_param.before"], g[f"{L}.{mg}.main_param.after"]
                ta = t["main_param_after." + hf_name(L, proj)]
                clip = np.linalg.norm(ga - gb)
                if clip == 0:
                    continue
                formula.append({
                    "step": s, "layer": L, "proj": proj, "clip_change_norm": float(clip),
                    "tpu_resid_over_clip": float(np.linalg.norm(ta - gb * factors(T[s][f"qk.eta.{L}"], proj)) / clip),
                    "gpu_resid_over_clip": float(np.linalg.norm(ga - gb * factors(A[s][f"qk.eta.{L}"], proj)) / clip),
                    "tpu_vs_gpu_after_over_clip": float(np.linalg.norm(ta - ga) / clip),
                })
    f_worst = max(formula, key=lambda r: r["tpu_resid_over_clip"])

    # 2. 第 1 步逐头最大 logit 对 FP32 参照
    ml = []
    for L in LABS:
        f, t, g, b = (np.asarray(D[1][f"qk.max_logit.{L}"], np.float64) for D in (F, T, A, B))
        et, eg = rel(t, f), rel(g, f)
        ml.append({"layer": L, "rel_l2_tpu_vs_ref": et, "rel_l2_gpu_vs_ref": eg, "R": et / max(eg, 1e-300),
                   "rel_l2_gpu_b_vs_ref": rel(b, f), "tpu_max_logit": t.tolist(), "gpu_max_logit": g.tolist(), "ref_max_logit": f.tolist()})
    m_worst = max(ml, key=lambda r: r["R"])

    # 3. 被裁的头
    def clipped(D, s):
        return np.concatenate([np.asarray(D[s][f"qk.eta.{L}"]) < 1 for L in LABS])

    per_step = []
    for s in steps:
        per_step.append({"step": s, "clipped_tpu": int(clipped(T, s).sum()), "clipped_gpu_a": int(clipped(A, s).sum()),
                         "clipped_gpu_b": int(clipped(B, s).sum()),
                         "set_diff_tpu_vs_a": int((clipped(T, s) != clipped(A, s)).sum()),
                         "set_diff_b_vs_a": int((clipped(B, s) != clipped(A, s)).sum()),
                         "lm_loss_rel_tpu_vs_a": (T[s]["lm_loss"] - A[s]["lm_loss"]) / A[s]["lm_loss"],
                         "lm_loss_rel_b_vs_a": (B[s]["lm_loss"] - A[s]["lm_loss"]) / A[s]["lm_loss"]})

    # 4. GPU 的 BF16 权重单独裁剪
    sem = []
    for s in range(1, 6):
        pth = Path(f"{a.gpu_a}/qk_clip/step_{s}.safetensors")
        if not pth.exists():
            continue
        import ml_dtypes
        g = st(str(pth))
        for L in LABS:
            for proj, mg in (("q", "linear_q_up_proj"), ("kv", "linear_kv_up_proj")):
                ma, pa, mb = (g[f"{L}.{mg}.{k}"] for k in ("main_param.after", "param.after", "main_param.before"))
                if not (ma != mb).any():
                    continue
                b16 = ma.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
                rows = (ma != mb).any(1)
                d = pa != b16
                ulp = np.abs(pa - b16) / np.maximum(np.spacing(np.abs(b16).astype(ml_dtypes.bfloat16)).astype(np.float64), 1e-300)
                sem.append({"step": s, "layer": L, "proj": proj, "frac_differ": float(d.mean()),
                            "frac_differ_in_clipped_rows": float(d[rows].mean()), "max_bf16_ulp": float(ulp[d].max()) if d.any() else 0.0,
                            "rel_l2": rel(pa, b16)})
    mare = lambda k: float(np.mean([abs(r[k]) for r in per_step]))  # noqa: E731
    checks = {
        "formula": {"rule": f"‖TPU 裁剪后 − GPU 裁剪前 × f(η_TPU)‖ ÷ ‖GPU 裁剪后 − GPU 裁剪前‖ ≤ {FORMULA_LIMIT}",
                    "worst": f_worst, "verdict": "pass" if f_worst["tpu_resid_over_clip"] <= FORMULA_LIMIT else "fail"},
        "max_logit_step1": {"rule": "R = rel_L2(TPU, FP32 参照) ÷ rel_L2(GPU, FP32 参照) ≤ 1.2（每层 8 个头）",
                            "worst": {k: m_worst[k] for k in ("layer", "R", "rel_l2_tpu_vs_ref", "rel_l2_gpu_vs_ref")},
                            "verdict": "pass" if m_worst["R"] <= R_LIMIT else "fail"},
        "clipped_heads": {"rule": "只报：30 步逐步逐头被裁集合的差，TPU 对 GPU-a 与 GPU-b 对 GPU-a",
                          "total_set_diff_tpu_vs_a": sum(r["set_diff_tpu_vs_a"] for r in per_step),
                          "total_set_diff_b_vs_a": sum(r["set_diff_b_vs_a"] for r in per_step),
                          "step1_clipped": [per_step[0]["clipped_tpu"], per_step[0]["clipped_gpu_a"], per_step[0]["clipped_gpu_b"]],
                          "verdict": "info"},
        "gpu_bf16_separate_clip": {"rule": "只报（已知语义差）", "max_frac_differ": max(r["frac_differ"] for r in sem) if sem else None,
                                   "max_frac_differ_in_clipped_rows": max(r["frac_differ_in_clipped_rows"] for r in sem) if sem else None,
                                   "max_bf16_ulp": max(r["max_bf16_ulp"] for r in sem) if sem else None,
                                   "lm_loss_mare_tpu_vs_a": mare("lm_loss_rel_tpu_vs_a"), "lm_loss_mare_b_vs_a": mare("lm_loss_rel_b_vs_a"),
                                   "verdict": "info"},
    }
    verdict = "fail" if any(c["verdict"] == "fail" for c in checks.values()) else "pass"
    out = {
        "what": "低阈值 QK-Clip（τ 0.1，30 步，hy4-mini）TPU 对 GPU：裁剪算法、逐层逐头最大 logit / η、被裁的头、裁剪后的权重",
        "runs": {"tpu": Path(a.tpu).name, "tpu_ref": Path(a.tpu_ref).name, "gpu": [Path(a.gpu_a).name, Path(a.gpu_b).name]},
        "tau": TAU, "alpha": ALPHA, "verdict": verdict, "checks": checks,
        "formula_rows": formula, "max_logit_step1": ml, "per_step": per_step, "gpu_bf16_separate_clip_rows": sem,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"verdict": verdict, **{k: v.get("verdict") for k, v in checks.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
