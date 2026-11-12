"""issue #10：各上下文并行（CP）方案第 1 步整网梯度对 FP64 的距离，和同配置不开 CP 的基线比。

被测与基线都是 train_hy3.py 第 1 步全量状态读（--full --no-hooks）；梯度由一阶动量反推（同 grads_from_moments.py：
第 1 步 g = m₁ / (1 − β1) / 裁剪系数，裁剪系数 = clip / grad_norm（grad_norm ≥ clip 时，否则 1），只用 exp_avg.* 与 metrics）；
真值是 fp64_step.py 的输出（同一份 BF16 权重、golden 第 1 步的全局 batch，FP64 前向 + 自动求导）。
对每个张量 n：e(x) = rel_l2(x_n, fp64_n)；R(臂) = e(臂) / e(基线)。另报按 FP64 梯度能量合起来的整体 e 与 R，
以及「只改执行方式」的对照臂给出的 R 范围（同一套算法，只换编译开关），CP 臂落在这个范围里就和换编译分不开。
同时报第 1 步 loss（前向）对 FP64 平均 loss 的差。

  python cp_step1_fp64.py --fp64 <fp64_step 输出目录> --base <运行根>/i10-st-b --arm 名=<运行根>/i10-st-rg2 ... --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from janus.store import Run  # noqa: E402


def group(n: str) -> str:
    if "experts.local_experts" in n or ".experts." in n:
        return "routed_expert"
    for k, g in [("router", "router"), ("shared_experts", "shared_expert"), ("linear_qkv.layer_norm", "input_norm"),
                 ("q_layernorm", "qk_norm"), ("k_layernorm", "qk_norm"), ("pre_mlp_layernorm", "mlp_norm"),
                 ("final_layernorm", "final_norm"), ("linear_qkv", "attn_qkv"), ("linear_proj", "attn_out"),
                 ("mlp.linear_fc", "dense_mlp")]:
        if k in n:
            return g
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp64", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", default=[], help="名=派生运行目录")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.fp64)
    fmeta = json.loads((d / "meta.json").read_text())
    f = safe_open(str(d / "grads.safetensors"), "numpy")
    fkeys = set(f.keys())
    base = Run.open(a.base)
    arms = {k: Run.open(v) for k, v in (x.split("=", 1) for x in a.arm)}
    def gnames(run):
        return {"grad." + n[len("exp_avg."):] for n in run.names(1, "exp_avg.*")}

    def coeff(run):
        oc = run.meta.get("optimizer_config") or {}
        clip, gn = float(oc.get("clip_grad") or 0.0), float(run.series("grad_norm")[1])
        return (1.0 if (clip <= 0 or gn < clip) else clip / gn), float(oc["adam_beta1"])

    def grad(run, n):
        c, b1 = coeff(run)
        return run.tensor(1, "exp_avg." + n[len("grad."):]).astype(np.float64) / (1.0 - b1) / c

    names = sorted(n for n in gnames(base) if n in fkeys and all(n in gnames(r) for r in arms.values()))
    loss64 = float(np.mean(fmeta["loss_per_micro"]))
    res = {"fp64_loss_mean": loss64, "n_tensors": len(names), "base": Path(a.base).name,
           "arm_runs": {k: Path(v).name for k, v in (x.split("=", 1) for x in a.arm)}, "arms": {}}

    def loss1(run):
        m = run.metrics().get(1, {})
        return m.get("lm_loss", m.get("loss"))

    acc = {k: {"num": 0.0, "den": 0.0, "groups": {}} for k in ["base", *arms]}
    per = {k: {} for k in ["base", *arms]}
    for n in names:
        ref = f.get_tensor(n).astype(np.float64)
        en = float(np.sum(ref * ref))
        for k, run in [("base", base), *arms.items()]:
            x = grad(run, n).reshape(ref.shape)
            dd = float(np.sum((x - ref) ** 2))
            per[k][n] = np.sqrt(dd / en) if en else float("nan")
            acc[k]["num"] += dd
            acc[k]["den"] += en
            g = acc[k]["groups"].setdefault(group(n), [0.0, 0.0])
            g[0] += dd
            g[1] += en
    eb = np.sqrt(acc["base"]["num"] / acc["base"]["den"])
    res["base_pooled_rel_l2"] = float(eb)
    res["base_loss1_minus_fp64"] = loss1(base) - loss64 if loss1(base) is not None else None
    for k in arms:
        e = np.sqrt(acc[k]["num"] / acc[k]["den"])
        rs = {n: per[k][n] / per["base"][n] for n in names if per["base"][n] > 0}
        worst = sorted(rs.items(), key=lambda t: -t[1])[:5]
        grp = {}
        for gname, (num, den) in acc[k]["groups"].items():
            bnum, bden = acc["base"]["groups"][gname]
            grp[gname] = {"rel_l2": float(np.sqrt(num / den)), "base_rel_l2": float(np.sqrt(bnum / bden)),
                          "R": float(np.sqrt(num / bnum)) if bnum else None}
        l1 = loss1(arms[k])
        res["arms"][k] = {"pooled_rel_l2": float(e), "R_pooled": float(e / eb), "R_max_tensor": worst[0][1],
                          "worst": [[n, float(r), float(per[k][n]), float(per["base"][n])] for n, r in worst],
                          "groups": grp, "loss1": l1, "loss1_minus_fp64": (l1 - loss64) if l1 is not None else None,
                          "loss1_minus_base": (l1 - loss1(base)) if l1 is not None and loss1(base) is not None else None}
        print(f"{k:12s} pooled e {e:.3e} (base {eb:.3e}) R {e/eb:.3f}  R_max {worst[0][1]:.2f} {worst[0][0]}  "
              f"loss1-fp64 {res['arms'][k]['loss1_minus_fp64']}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
