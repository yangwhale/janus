"""前向激活对 FP64 的 R-ratio：被测（TPU）离 FP64 真值不比 GPU 远，就算对齐。写成与 `janus run` 同一格式的结果文件。

为什么不用 calibrated：GPU 默认设置下第 1 步前向逐位可复现，GPU 自己两两比的噪声上限是 0（docs/reference/regression.md 第五节「前向激活不用这一条」），
任何跨平台的前向差都会被判超出 0 的无穷多倍。这里换一把尺子：每个观测点用 fp64_forward.py 从同一份输入、同一份 BF16 权重在精确算术里
算出的真值，比
    R = rel_l2(被测, FP64) / rel_l2(GPU, FP64)
R ≤ 1.2 过（op_replay 的数值门槛，推断，和 GPU 侧 phase1-l1 的 L1.rratio 同一个起点：只差舍入顺序时 R ≈ 1，多一次同量级的独立舍入 R ≈ √2）。

两种用法（--mode）：
  isolated：逐层隔离回放（layer_replay.py 的产物，每层输入是 golden 的 layers.k.in）。真值从同一个 golden 层输入算，层与层之间不串，
            得到 L2.layer{k}.fwd（op_replay，每个观测点一个用例）与路由两条：
              L2.layer{k}.router：被测的 top-8 对 FP64 真值的集合一致比例，门槛 1 − 1.2·(1 − GPU 对 FP64 的一致比例)（同上 1.2 倍）；
              L1.router.topk.fp64：被测自己的路由输入（mlp_norm.out）→ FP64 logits → 加偏置取 top-8，与被测选中的比（只看路由这一个算子），
              门槛 0.97，同 GPU 侧 phase1-full（推断）。
  ops：单算子，各用各自的输入（被测是逐层隔离回放的产物）。每个算子的真值从「这一侧自己抓到的算子输入」在 FP64 里算：
       被测离「自己输入上的真值」多远，对比 GPU 离「GPU 自己输入上的真值」多远，R ≤ 1.2。这样上游差异不进来，只看算子本身：
         L1.ops.op.qkv（输入归一化 + qkv 投影，输入是 golden 的层输入，两侧相同）、L1.ops.op.qk_norm（qkv.out 里的 q / k → 每头 RMSNorm）、
         L1.ops.op.attn_proj（core.out → 输出投影）、L1.ops.op.mlp_norm（MoE 层：bf16(层输入 + attn.out) → MLP 前 RMSNorm）、
         L1.ops.op.router_gemm（mlp_norm.out → 路由 logits）、L1.ops.op.dense_fc1（第 0 层：残差和 → MLP 前 RMSNorm + gate|up 投影）、
         L1.ops.op.dense_act_fc2（第 0 层：fc1.out → silu(gate)·up → down）；
         L1.ops.kernel.attn_core（q_norm / k_norm / v → RoPE、1/√d、因果注意力）、L1.ops.kernel.moe（mlp_norm.out → 路由 + 分组专家 + 合并 + 共享专家）
  chain：整网串跑（被测是加钩子的训练步 / 观测构建抓到的第 1 步）。真值从 golden 的 embed.out（L0 已逐位）一路 FP64 算到 logits 与逐 token loss，
         得到 L3.step1.logits、L3.step1.loss.per_token（op_replay）。只比 rank 0 第一个 microbatch（golden 只抓了这一份）。

  bwd：逐层隔离回放的反向（layer_replay.py --grad 的产物，每层的输入与输出端余切都是 golden 的 layers.k.in / layers.k.out.grad）。
       真值 = fp64_backward.layer_vjp 在 FP64 里求的 dx，得到 L2.layer{k}.bwd（op_replay，用例 layers.k.in.grad）。
       为什么换尺子：GPU 自己重跑第 1 步反向只差原子加顺序（第 3 层 1.3e-5），TPU 只改执行参数的重跑差 1e-5 – 1.7e-4，
       两边都不翻路由，量不出两种硬件各自的 BF16 舍入顺序与近似平局路由（2b 的调查，case 01）。
  step：第 1 步整网梯度与由它决定的状态（被测是第 1 步全量状态读 + grads_from_moments.py 反推的梯度）。真值是 fp64_step.py 的输出
        （--fp64-step 目录）：同一份权重、同一个全局 batch 在 FP64 里求的梯度，和每层每个专家的选中 token 数。得到
          L2.layer{i}.grads、L3.step1.grads.all：逐张量 R；
          L3.step1.optim_state、L3.step1.updated：两侧各自的 FP32 主权重 + FP64 梯度，用这一侧自己的裁剪系数在 FP64 里重放一步 AdamW，
            得到这一侧的真值，再比 R（裁剪系数是 grad_norm 这一个标量的后果，由 L3 / L4 的 grad_norm 检查单独判；这里只看张量本身）；
          L3.step1.bias_after：FP64 选中数按 Megatron 规则（偏置 += 速率 · sign(平均选中数 − 本专家选中数)）算的更新量，比更新量的 R
            （更新量是 ±速率，R 实际是「更新方向和真值相反的专家数」之比的平方根）。

  python fwd_rratio.py --mode isolated --hf <HF 目录> --golden <运行根>/g1k-a --test <运行根>/r-l2-norm-perop --out <结果目录> --stage <名>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import fp64_forward as F  # noqa: E402

from janus import __version__  # noqa: E402
from janus import metrics as M  # noqa: E402
from janus.catalog import Catalog  # noqa: E402
from janus.checks import REGISTRY, Threshold  # noqa: E402
from janus.checks.base import ERROR, FAIL, INCONCLUSIVE, PASS  # noqa: E402
from janus.report.markdown import render  # noqa: E402
from janus.store import Run  # noqa: E402

R_MAX = 1.2
R_SOURCE = ("推断：同 GPU 侧 phase1-l1 的 L1.rratio 起点。只差舍入顺序时 R≈1；多一次同量级的独立舍入误差方差约翻倍、R≈√2≈1.41，取两者之间")
ROUTER_FP64_MIN = 0.97
ROUTER_FP64_SOURCE = "推断：沿用 phase1-full 的 L1.router.topk.fp64（GPU FP32 路由对 FP64 的选中集合，先给宽松下限）"
SKIP = (".topk_idx", ".probs", ".counts", ".in", ".grad")      # 离散量另判；层输入两边相同；反向不在这里


def float_cases(names, test: Run, golden: Run, ref: dict) -> list:
    cases = []
    for n in names:
        if n.endswith(SKIP) or n not in ref or n not in golden.names(1) or n not in test.names(1):
            continue
        out = test.tensor(1, n)
        g = golden.tensor(1, n)
        if out.shape != g.shape:
            raise SystemExit(f"{n}：被测 {out.shape} 与 golden {g.shape} 形状不同")
        cases.append({"name": n, "out": out, "fp64": ref[n].reshape(out.shape), "ref": g,
                      "extra": {"rel_l2_test_golden": M.rel_l2(out.astype(np.float64), g.astype(np.float64))}})
    return cases


def ops_checks(hf, g: Run, t: Run, layers, pos, op, th, same: Run | None = None) -> list:
    """--mode ops：每个算子两侧各用自己的输入算 FP64 真值。返回 op_replay 结果列表。
    参考档另比「被测对 GPU」（vs_ref_*，只报）：只在两侧算子输入相同时比——输入本来就是 golden 的（qkv），
    或 same（layer_replay.py --l1 的运行，算子输入都是 golden 的）里有同名输出 l1.<名>；其余算子不比（输入不同，差里混着上游）。"""
    import ml_dtypes
    c = hf.cfg
    H, KV, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta, fd = c["rms_norm_eps"], c["rope_parameters"]["rope_theta"], c["first_k_dense_replace"]
    rep = H // KV
    cases: dict[str, list] = {}

    same_names = set(same.names(1)) if same is not None else set()

    def add(key, name, f64_of, side_in, same_input=False):
        """f64_of(run) → 这一侧输入上的 FP64 真值；被测与 GPU 各算一次。same_input：两侧这个算子的输入相同。"""
        out, ref = t.tensor(1, name), g.tensor(1, name)
        ft = f64_of(t).reshape(out.shape)
        fg = f64_of(g).reshape(ref.shape)
        rt, rg = M.rel_l2(out.astype(np.float64), ft), M.rel_l2(ref.astype(np.float64), fg)
        vs = {}
        if same_input:
            vs = {"vsref": ref, "vsref_note": "两侧输入相同，被测输出直接对 GPU"}
        elif f"l1.{name}" in same_names:
            vs = {"vsref": ref, "vsref_out": same.tensor(1, f"l1.{name}").reshape(ref.shape),
                  "vsref_note": f"{same.name} 的 l1.{name}：同一算子在 golden 输入上的 TPU 输出，对 GPU"}
        cases.setdefault(key, []).append({
            "name": name, "out": out, "fp64": ft, **vs,
            "extra": {"rel_l2_test_vs_own_fp64": rt, "rel_l2_gpu_vs_own_fp64": rg, "r_ratio_own_inputs": rt / rg if rg else float("inf"),
                      "input": side_in}})

    def T(run, n):
        return run.tensor(1, n)[:, 0].astype(np.float64)

    for i in layers:
        L, p = f"layers.{i}", f"model.layers.{i}."
        w_in = hf.get(p + "input_layernorm.weight")
        wq, wk, wv = (hf.get(p + f"self_attn.{x}_proj.weight") for x in "qkv")

        def qkv64(run, L=L, w_in=w_in, wq=wq, wk=wk, wv=wv):
            h1 = F.rmsnorm(T(run, f"{L}.in"), w_in, eps)
            S = h1.shape[0]
            q, k, v = (h1 @ wq.T).reshape(S, H, d), (h1 @ wk.T).reshape(S, KV, d), (h1 @ wv.T).reshape(S, KV, d)
            return np.concatenate([np.concatenate([q[:, gi * rep:(gi + 1) * rep].reshape(S, rep * d), k[:, gi], v[:, gi]], -1)
                                   for gi in range(KV)], -1)
        add("qkv", f"{L}.attn.qkv.out", qkv64, "golden 的层输入（两侧相同）", same_input=True)

        def split(run, L=L):
            x = T(run, f"{L}.attn.qkv.out").reshape(-1, KV, (rep + 2) * d)
            return x[:, :, :rep * d].reshape(-1, H, d), x[:, :, rep * d:(rep + 1) * d], x[:, :, (rep + 1) * d:]
        wqn, wkn = hf.get(p + "self_attn.q_norm.weight"), hf.get(p + "self_attn.k_norm.weight")
        add("qk_norm", f"{L}.attn.q_norm.out", lambda run, s=split, w=wqn: F.rmsnorm(s(run)[0], w, eps), "本侧 qkv.out 里的 q")
        add("qk_norm", f"{L}.attn.k_norm.out", lambda run, s=split, w=wkn: F.rmsnorm(s(run)[1], w, eps), "本侧 qkv.out 里的 k")

        def core64(run, L=L, s=split):
            qn = run.tensor(1, f"{L}.attn.q_norm.out")[:, 0].astype(np.float64)
            kn = run.tensor(1, f"{L}.attn.k_norm.out")[:, 0].astype(np.float64)
            return F.attention(F.rope(qn, pos, theta, d), F.rope(kn, pos, theta, d), s(run)[2], H, KV)
        add("attn", f"{L}.attn.core.out", core64, "本侧 q_norm.out、k_norm.out 与 qkv.out 里的 v")
        wo = hf.get(p + "self_attn.o_proj.weight")
        add("attn_proj", f"{L}.attn.proj.out", lambda run, L=L, wo=wo: T(run, f"{L}.attn.core.out") @ wo.T, "本侧 core.out")
        w_post = hf.get(p + "post_attention_layernorm.weight")
        if i < fd:
            wg, wu, wd = hf.get(p + "mlp.gate_proj.weight"), hf.get(p + "mlp.up_proj.weight"), hf.get(p + "mlp.down_proj.weight")

            def fc1_64(run, L=L, w_post=w_post, wg=wg, wu=wu):
                hn = F.rmsnorm(T(run, f"{L}.mlp_norm.out"), w_post, eps)          # 第 0 层 mlp_norm.out 是残差和
                return np.concatenate([hn @ wg.T, hn @ wu.T], -1)
            add("dense_fc1", f"{L}.mlp.fc1.out", fc1_64, "本侧残差和（第 0 层的 mlp_norm.out）")

            def fc2_64(run, L=L, wd=wd):
                f = T(run, f"{L}.mlp.fc1.out")
                half = f.shape[-1] // 2
                return (F.silu(f[:, :half]) * f[:, half:]) @ wd.T
            add("dense_act_fc2", f"{L}.mlp.fc2.out", fc2_64, "本侧 fc1.out")
        else:
            def norm64(run, L=L, w_post=w_post):
                res = (run.tensor(1, f"{L}.in")[:, 0].astype(np.float32) + run.tensor(1, f"{L}.attn.out")[:, 0].astype(np.float32))
                return F.rmsnorm(res.astype(ml_dtypes.bfloat16), w_post, eps)
            add("mlp_norm", f"{L}.mlp_norm.out", norm64, "bf16(层输入 + 本侧 attn.out)")
            wr = hf.get(p + "mlp.router.gate.weight")
            add("router", f"{L}.mlp.router.logits", lambda run, L=L, wr=wr: T(run, f"{L}.mlp_norm.out") @ wr.T, "本侧 mlp_norm.out")
            add("moe", f"{L}.mlp.out", lambda run, L=L, i=i: F.moe_forward(hf, i, T(run, f"{L}.mlp_norm.out"))["mlp.out"][:, 0],
                "本侧 mlp_norm.out（真值按 FP64 分数重新选专家）")
        print(f"[ops] 第 {i} 层", flush=True)
    out = []
    # L1.ops.op.*：投影、归一化、路由 GEMM 这类单个算子；L1.ops.kernel.*：注意力 kernel、MoE 块（分组 GEMM + 合并）
    names = {"qkv": "L1.ops.op.qkv", "qk_norm": "L1.ops.op.qk_norm", "attn_proj": "L1.ops.op.attn_proj",
             "mlp_norm": "L1.ops.op.mlp_norm", "router": "L1.ops.op.router_gemm", "dense_fc1": "L1.ops.op.dense_fc1",
             "dense_act_fc2": "L1.ops.op.dense_act_fc2", "attn": "L1.ops.kernel.attn_core", "moe": "L1.ops.kernel.moe"}
    nr = {"r_ratio_own_inputs": R_MAX}
    for key, cid in names.items():
        if key not in cases:
            continue
        r = op(cid, "L1", cases[key], None)                    # 先按 op_replay 算各用例的对 FP64 指标（被测）
        worst = max(cs["extra"]["r_ratio_own_inputs"] for cs in cases[key])
        r.metrics["r_ratio_own_inputs"] = worst
        r.metrics["rel_l2_gpu_vs_own_fp64_max"] = max(cs["extra"]["rel_l2_gpu_vs_own_fp64"] for cs in cases[key])
        from janus.checks.base import judge_numeric
        r.status, bad = judge_numeric({"r_ratio_own_inputs": worst}, nr)
        r.threshold = {"kind": "numeric", "values": nr, "source": R_SOURCE + "；两侧各用自己抓到的算子输入算真值，只比算子本身"}
        r.message = "全部在门槛内" if r.status == "pass" else f"超线：{bad}"
        r.worst = sorted(({"name": cs["name"], "metric": "r_ratio_own_inputs", "value": cs["extra"]["r_ratio_own_inputs"],
                           "location": None} for cs in cases[key]), key=lambda x: -x["value"])[:5]
        out.append(r)
        print(f"[{cid}] {r.status} R={worst:.3f}", flush=True)
    return out


def step_checks(hf, g: Run, t: Run, d: Path, op, th) -> list:
    """--mode step：见文件头。"""
    from safetensors.numpy import load_file

    from janus.replay import adamw_step, clip_coeff, no_weight_decay
    f64 = load_file(str(d / "grads.safetensors"))
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("rows") != meta.get("global_rows"):
        raise SystemExit(f"{d}：只算了 {meta.get('rows')} / {meta.get('global_rows')} 行，不是整步")
    names = sorted(n for n in f64 if n in set(g.names(1, "grad.*")) and n in set(t.names(1, "grad.*")))

    def gcase(n):
        out, ref = t.tensor(1, n), g.tensor(1, n)
        return {"name": n, "out": out, "ref": ref, "fp64": f64[n].reshape(out.shape),
                "extra": {"rel_l2_test_golden": M.rel_l2(out.astype(np.float64), ref.astype(np.float64))}}
    checks = []

    def add(cid, level, cases):
        r = op(cid, level, cases, th, params={"worst_by": ["r_ratio"]})
        # 另报按 FP64 梯度能量加权的整体 R（所有用例拼成一个向量）：小张量的 R 抖得大，整体看的是能量大头
        num = sum(cs["_e_t"] for cs in cases if "_e_t" in cs)
        den = sum(cs["_e_g"] for cs in cases if "_e_g" in cs)
        if den:
            r.metrics["r_ratio_pooled"] = float(np.sqrt(num / den))
        checks.append(r)
        print(f"[{cid}] {r.status} R max {r.metrics.get('r_ratio')} pooled {r.metrics.get('r_ratio_pooled')}", flush=True)

    def pooled(cs):
        o, ref, tr = (np.asarray(cs[k], np.float64) for k in ("out", "ref", "fp64"))
        cs["_e_t"], cs["_e_g"] = float(((o - tr) ** 2).sum()), float(((ref - tr) ** 2).sum())
        return cs

    nl = hf.cfg["num_hidden_layers"]
    for i in range(nl):
        add(f"L2.layer{i}.grads", "L2", [pooled(gcase(n)) for n in names if n.startswith(f"grad.decoder.layers.{i}.")])
    add("L3.step1.grads.all", "L3", [pooled(gcase(n)) for n in names])

    # 优化器一步：两侧各用自己的主权重与裁剪系数，梯度换成 FP64 的，重放出这一侧的真值
    oc = g.meta.get("optimizer_config") or {}
    hp = dict(lr=g.series("lr").get(1), beta1=float(oc.get("adam_beta1", 0.9)), beta2=float(oc.get("adam_beta2", 0.999)),
              eps=float(oc.get("adam_eps", 1e-8)), step=1)
    wd_cfg = float(g.series("wd").get(1) if "wd" in g.metrics().get(1, {}) else oc.get("weight_decay", 0.0))
    coeff = {run.name: clip_coeff(run.series("grad_norm").get(1), oc.get("clip_grad")) for run in (g, t)}
    upd, st = [], []
    for n in names:
        pn = n[len("grad."):]
        wd = 0.0 if no_weight_decay(pn, f64[n].shape) else wd_cfg
        side = {}
        for run in (t, g):
            p = run.tensor(1, f"main_param.{pn}").astype(np.float64)
            new, m, v = adamw_step(p, f64[n], wd=wd, clip_coeff=coeff[run.name], **hp)
            side[run.name] = {"upd": (run.tensor(1, f"main_param_after.{pn}").astype(np.float64) - p, new - p),
                              "exp_avg": (run.tensor(1, f"exp_avg.{pn}"), m), "exp_avg_sq": (run.tensor(1, f"exp_avg_sq.{pn}"), v)}
        for key, bucket, label in (("upd", upd, f"update.{pn}"), ("exp_avg", st, f"exp_avg.{pn}"), ("exp_avg_sq", st, f"exp_avg_sq.{pn}")):
            (ot, tt), (og, tg) = side[t.name][key], side[g.name][key]
            ot, og = np.asarray(ot, np.float64), np.asarray(og, np.float64)
            et, eg = M.rel_l2(ot, tt), M.rel_l2(og, tg)
            bucket.append({"name": label, "out": np.asarray(side[t.name][key][0]), "fp64": tt,
                           "_e_t": float(((ot - tt) ** 2).sum()), "_e_g": float(((og - tg) ** 2).sum()),
                           "extra": {"rel_l2_test_vs_own_fp64": et, "rel_l2_gpu_vs_own_fp64": eg,
                                     "r_ratio_own": et / eg if eg else float("inf")}})
    for cid, cases in (("L3.step1.updated", upd), ("L3.step1.optim_state", st)):
        r = op(cid, "L3", cases, None)
        worst = max(cs["extra"]["r_ratio_own"] for cs in cases)
        r.metrics.update(r_ratio_own=worst, r_ratio_pooled=float(np.sqrt(sum(c["_e_t"] for c in cases) / sum(c["_e_g"] for c in cases))),
                         clip_coeff_test=coeff[t.name], clip_coeff_golden=coeff[g.name])
        from janus.checks.base import judge_numeric
        r.status, bad = judge_numeric({"r_ratio_own": worst}, {"r_ratio_own": R_MAX})
        r.threshold = {"kind": "numeric", "values": {"r_ratio_own": R_MAX},
                       "source": R_SOURCE + "；真值 = 这一侧的主权重 + FP64 梯度 + 这一侧的裁剪系数，FP64 重放一步 AdamW"}
        r.message = "全部在门槛内" if r.status == PASS else f"超线：{bad}"
        r.worst = sorted(({"name": c["name"], "metric": "r_ratio_own", "value": c["extra"]["r_ratio_own"], "location": None}
                          for c in cases), key=lambda x: -x["value"])[:5]
        checks.append(r)
        print(f"[{cid}] {r.status} R max {worst:.3f} pooled {r.metrics['r_ratio_pooled']:.3f}", flush=True)

    # 路由偏置一步：FP64 选中数 → 更新量真值
    rate = float((g.meta.get("model_config") or {}).get("moe_router_bias_update_rate", 1e-3))
    cases = []
    for L, cnt in sorted(meta["counts"].items()):
        cnt = np.asarray(cnt, np.float64)
        n_b, n_a = f"buffer.decoder.layers.{L}.mlp.router.expert_bias", f"buffer_after.decoder.layers.{L}.mlp.router.expert_bias"
        if n_b not in set(g.names(1)) or n_a not in set(t.names(1)):
            continue
        truth = rate * np.sign(cnt.mean() - cnt)
        dt = t.tensor(1, n_a).astype(np.float64) - t.tensor(1, n_b).astype(np.float64)
        dg = g.tensor(1, n_a).astype(np.float64) - g.tensor(1, n_b).astype(np.float64)
        cases.append(pooled({"name": f"update.{n_b[len('buffer.'):]}", "out": dt, "ref": dg, "fp64": truth,
                             "extra": {"wrong_sign_test": int((np.sign(dt) != np.sign(truth)).sum()),
                                       "wrong_sign_golden": int((np.sign(dg) != np.sign(truth)).sum()),
                                       "near_mean_experts": int((np.abs(cnt - cnt.mean()) < 0.01 * cnt.mean()).sum())}}))
    if cases:
        add("L3.step1.bias_after", "L3", cases)
    return checks


def write(out: Path, checks, runs, stage: str, script: Path, started: str, env: dict, test_name: str, golden_name: str) -> int:
    (out / "details").mkdir(parents=True, exist_ok=True)
    js = []
    for r in checks:
        d = r.to_json()
        if r.details:
            rel = f"details/{r.id}.jsonl"
            with open(out / rel, "w") as f:
                for row in r.details:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            d["details"] = rel
        js.append(d)
    summary = {k: sum(1 for c in js if c["status"] == k) for k in (PASS, FAIL, INCONCLUSIVE, ERROR)}
    summary["hard_fail"] = any(c["status"] == FAIL and c.get("hard", True) for c in js)
    rel_script = f"models/hunyuan3/analysis/{script.name}"
    res = {"schema_version": 1, "janus": {"version": __version__, "commit": None},
           "catalog": {"path": rel_script, "sha256": hashlib.sha256(script.read_bytes()).hexdigest()},
           "runs": {r.name: {"path": str(r.path), "meta_sha256": r.meta_sha256()} for r in runs},
           "started_at": started, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "environment": env, "summary": summary, "checks": js}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    cat = Catalog(Path(rel_script), {"model": "hunyuan3-4l", "stage": stage, "runs": {"test": test_name, "golden": [golden_name]}})
    (out / "report.md").write_text(render(cat, res))
    print(f"pass {summary['pass']} / fail {summary['fail']} / inconclusive {summary['inconclusive']} / error {summary['error']} → {out}")
    return 1 if summary["hard_fail"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["isolated", "ops", "chain", "bwd", "step"], required=True)
    ap.add_argument("--fp64-step", help="--mode step：fp64_step.py 的输出目录")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--same-input", help="--mode ops：layer_replay.py --l1 的运行（算子输入都是 golden 的），给参考档的「被测对 GPU」用")
    a = ap.parse_args(argv)
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    hf, g, t = F.HF(a.hf), Run.open(a.golden), Run.open(a.test)
    for r in (g, t):
        r.enable_cache()
    pos = g.tensor(1, "input.position_ids")[0]
    op, da = REGISTRY["op_replay"], REGISTRY["discrete_agreement"]
    th = Threshold("numeric", values={"r_ratio": R_MAX}, source=R_SOURCE)
    checks = []
    layers = [int(x) for x in a.layers.split(",") if x]
    fd = hf.cfg["first_k_dense_replace"]
    if a.mode == "isolated":
        for k in layers:
            ref = F.layer_forward(hf, k, g.tensor(1, f"layers.{k}.in")[:, 0], pos)
            cases = float_cases(t.names(1, f"layers.{k}.*"), t, g, ref)
            r = op(f"L2.layer{k}.fwd", "L2", cases, th, params={"worst_by": ["r_ratio"]})
            checks.append(r)
            print(f"[L2.layer{k}.fwd] {r.status} r_ratio max {r.metrics.get('r_ratio')}", flush=True)
            if k < fd:
                continue
            n = f"layers.{k}.mlp.router.topk_idx"
            fp = ref[n]
            gpu_frac = M.topk_agreement(g.tensor(1, n), fp).set_frac
            lo = 1.0 - R_MAX * (1.0 - gpu_frac)
            r = da(f"L2.layer{k}.router", "L2", [(n, t.tensor(1, n), fp)],
                   Threshold("numeric", values={"set_frac": lo},
                             source=f"实测 GPU（golden）对 FP64 真值的集合一致比例 {gpu_frac:.6f}，被测的翻转率不超过它的 {R_MAX} 倍（推断，同 R-ratio）"),
                   params={"mode": "topk"})
            r.metrics["gpu_vs_fp64_set_frac"] = gpu_frac
            checks.append(r)
            # 只看路由这一个算子：被测自己的路由输入 → FP64 logits → 加偏置取 top-8，对被测选中的
            x = t.tensor(1, f"layers.{k}.mlp_norm.out").astype(np.float64).reshape(-1, hf.cfg["hidden_size"])
            w = hf.get(f"model.layers.{k}.mlp.router.gate.weight")
            b = hf.get(f"model.layers.{k}.mlp.expert_bias")
            sel = F.topk_sets(1.0 / (1.0 + np.exp(-(x @ w.T))) + b, hf.cfg["num_experts_per_tok"]).astype(np.int32)
            r = da(f"L1.router.topk.fp64.layer{k}", "L1", [(n, t.tensor(1, n), sel)],
                   Threshold("numeric", values={"set_frac": ROUTER_FP64_MIN}, source=ROUTER_FP64_SOURCE), params={"mode": "topk"})
            checks.append(r)
            print(f"[router {k}] 对 FP64 {checks[-2].metrics} 单算子 {r.metrics}", flush=True)
    elif a.mode == "bwd":
        import fp64_backward as B
        for k in layers:
            bias = g.tensor(1, f"buffer.decoder.layers.{k}.mlp.router.expert_bias").astype(np.float64) if k >= fd else None
            dx = B.layer_vjp(hf, k, g.tensor(1, f"layers.{k}.in")[:, 0], g.tensor(1, f"layers.{k}.out.grad")[:, 0], pos, bias)
            n = f"layers.{k}.in.grad"
            out, ref = t.tensor(1, n), g.tensor(1, n)
            cases = [{"name": n, "out": out, "ref": ref, "fp64": dx.reshape(out.shape),
                      "extra": {"rel_l2_test_golden": M.rel_l2(out.astype(np.float64), ref.astype(np.float64))}}]
            r = op(f"L2.layer{k}.bwd", "L2", cases, th, params={"worst_by": ["r_ratio"]})
            checks.append(r)
            print(f"[L2.layer{k}.bwd] {r.status} r_ratio {r.metrics.get('r_ratio')}", flush=True)
    elif a.mode == "step":
        checks = step_checks(hf, g, t, Path(a.fp64_step), op, th)
    elif a.mode == "ops":
        checks = ops_checks(hf, g, t, layers, pos, op, th, Run.open(a.same_input) if a.same_input else None)
    else:
        x = g.tensor(1, "embed.out")[:, 0]
        if t.names(1, "embed.out"):
            te = t.tensor(1, "embed.out")
            if te.tobytes() != g.tensor(1, "embed.out").tobytes():
                raise SystemExit("被测的 embed.out 与 golden 不逐位：真值的起点对不上")
        for k in range(hf.cfg["num_hidden_layers"]):
            ref = F.layer_forward(hf, k, x, pos)
            x = ref[f"layers.{k}.out"][:, 0]
            print(f"[chain] 第 {k} 层", flush=True)
        ref = F.head_forward(hf, x, g.tensor(1, "input.labels")[0])
        for n, cid in (("logits", "L3.step1.logits"), ("loss.per_token", "L3.step1.loss.per_token"), ("final_norm.out", "L3.step1.final_norm")):
            cases = float_cases([n], t, g, ref)
            if not cases:                                 # 被测没有这个观测点（如整网前向回放只记 logits 与逐 token loss）
                continue
            r = op(cid, "L3", cases, th, params={"worst_by": ["r_ratio"]})
            if n == "logits":                             # 另报 KL(FP64 ‖ ·)：被测与 GPU 各自离真值的分布差
                p = ref[n].reshape(cases[0]["out"].shape)
                r.metrics["kl_test_vs_fp64"] = M.kl_div(cases[0]["out"], p)
                r.metrics["kl_golden_vs_fp64"] = M.kl_div(cases[0]["ref"], p)
                r.metrics["kl_test_vs_golden"] = M.kl_div(cases[0]["out"], cases[0]["ref"])
            checks.append(r)
            print(f"[{cid}] {r.status} {r.metrics}", flush=True)
    env = {"hf_config_sha256": hashlib.sha256((Path(a.hf) / "config.json").read_bytes()).hexdigest(), "numpy": np.__version__,
           "reference": "models/hunyuan3/analysis/fp64_forward.py（FP64，中间不舍入）", "mode": a.mode}
    if a.mode == "bwd":
        env["reference"] = "models/hunyuan3/analysis/fp64_backward.py（FP64 自动求导，中间不舍入）"
    if a.mode == "step":
        sm = json.loads((Path(a.fp64_step) / "meta.json").read_text())
        env["reference"] = "models/hunyuan3/analysis/fp64_step.py（FP64 整网前向 + 自动求导，全局 batch）"
        env["fp64_step"] = {"sha256": hashlib.sha256((Path(a.fp64_step) / "grads.safetensors").read_bytes()).hexdigest(),
                            "scale_fit": sm.get("scale_fit"), "rows": sm.get("rows"), "semantics": sm.get("semantics")}
    runs = [t, g] + ([Run.open(a.same_input)] if a.mode == "ops" and a.same_input else [])
    return write(Path(a.out), checks, runs, a.stage, Path(__file__).resolve(), started, env, t.name, g.name)


if __name__ == "__main__":
    sys.exit(main())
