"""索引器 KD 在 TPU 上偏高：拆到原语。在 TPU 上照 MaxText `calculate_indexer_loss`（`indexer_kd_mcore_log` +
`indexer_fp32_scores` + `transcendental_accuracy_highest`）的写法逐步算，每一步的 FP32 中间量取回主机，用 FP64 只重做
这一步（输入用 TPU 上一步的 FP32 结果），量这一步自己的误差与系统偏差：

  教师：s_h = Σ exp(x_h − max)（exp、求和）→ p_h = e_h / s_h（除法）→ a = Σ_h p_h（跨头求和）→ S = Σ a（求和）
        → p = a / max(S, 1e-10)（除法）
  学生：e = exp(z − max)（exp）→ s = Σ e（求和）→ l = log(s)（log）→ log q = (z − max) − l

KD 的行值 KL = Σ p (log p − log q) ≪ 1（第 0 层约 1.1e-4），所以 p 每行之和偏离 1 的 δ、log q 整体的平移 ε 都直接加到 KL 上
（KL ≈ 真 KL + δ − ε），1e-8 的偏差就是 1e-4 的相对偏差。另算两种候选写法：除法换成一步牛顿修正的倒数
（与 `muon_adam_accurate` 同一写法），求和换成先在 FP64 主机上算（只作对照，量求和这一步的贡献）。

输入（`--make-inputs`）：TPU `debug_sow_indexer` 运行的 diag_kd_inputs.safetensors 里第 0 行的教师 q / k、学生分数与掩码。

  python kd_norm_probe.py --make-inputs --kd <运行>/diag_kd_inputs.safetensors --out kd_probe_inputs.npz
  python kd_norm_probe.py --inputs gs://…/kd_probe_inputs.npz --out kd_norm_probe.json      # TPU 上（SCRIPT=kd_norm_probe.py）
  python kd_norm_probe.py --inputs kd_probe_inputs.npz --out x.json --cpu                     # CPU 上只验证脚本
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

S = 4096
MASK = -0.7 * float(np.finfo(np.float32).max)  # MaxText DEFAULT_MASK_VALUE


def make_inputs(kd: str, out: str):
    from safetensors import safe_open

    f = safe_open(kd, "np")
    layers = sorted({k.rsplit("/debug_kd", 1)[0] for k in f.keys()})
    d = {}
    for i, L in enumerate(layers):
        g = lambda n: f.get_tensor(f"{L}/debug_kd_{n}/0")
        d[f"l{i}_q"] = g("query").reshape(S, -1, g("query").shape[-1]).astype(np.float32)
        d[f"l{i}_k"] = g("key").reshape(S, -1, g("key").shape[-1]).astype(np.float32)
        d[f"l{i}_z"] = g("indexer_score").reshape(S, S).astype(np.float32)
        d[f"l{i}_m"] = g("indexer_mask").reshape(S, S).astype(np.float32)
        d[f"l{i}_kl_rows_sow"] = g("kl_rows").reshape(S).astype(np.float32)
        d[f"l{i}_target_sow"] = g("target").reshape(S, S).astype(np.float32)
    d["layers"] = np.array(layers)
    np.savez(out, **d)
    print({k: v.shape for k, v in d.items()})


def load_inputs(path: str) -> dict:
    if path.startswith("gs://"):
        loc = Path(tempfile.mkdtemp()) / "inputs.npz"
        subprocess.run(["gcloud", "storage", "cp", "-q", path, str(loc)], check=True)
        path = str(loc)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def tpu_steps(q, k, z, m):
    """TPU 上逐步算，返回各步 FP32 中间量。写法与 attention_mla.calculate_indexer_loss（mcore_log 分支）相同。"""
    import jax
    import jax.numpy as jnp
    from jax import lax

    HI = lax.AccuracyMode.HIGHEST if hasattr(lax, "AccuracyMode") else None

    def exp(x):
        return lax.exp(x, accuracy=HI) if HI is not None else jnp.exp(x)

    def log(x):
        return lax.log(x, accuracy=HI) if HI is not None else jnp.log(x)

    def recip(x):  # 一步牛顿修正（muon_adam_accurate 的写法）
        r = 1.0 / x
        return r * (2.0 - x * r)

    @jax.jit
    def f(q, k, z, m):
        x = jnp.einsum("thd,shd->hts", q, k, precision=lax.Precision.HIGHEST) + m[None]
        xs = x - lax.stop_gradient(jnp.max(x, -1, keepdims=True))
        e = exp(xs)
        s = jnp.sum(e, -1, keepdims=True)
        p_h = e / s
        p_h_fix = e * recip(s)
        a = jnp.sum(p_h, 0)
        a_fix = jnp.sum(p_h_fix, 0)
        tot = jnp.sum(a, -1, keepdims=True)
        tot_fix = jnp.sum(a_fix, -1, keepdims=True)
        p = a / jnp.maximum(tot, 1e-10)
        p_fix = a_fix * recip(jnp.maximum(tot_fix, 1e-10))
        zz = z + m
        zs = zz - lax.stop_gradient(jnp.max(zz, -1, keepdims=True))
        e2 = exp(zs)
        s2 = jnp.sum(e2, -1, keepdims=True)
        l2 = log(s2)
        lq = zs - l2

        def kl(pp):
            nz = pp > 0
            return jnp.sum(jnp.where(nz, pp * (log(jnp.maximum(pp, 1e-10)) - jnp.where(nz, lq, 0.0)), 0.0), -1)

        return dict(xs=xs, e=e, s=s, p_h=p_h, a=a, tot=tot, p=p, p_fix=p_fix, zs=zs, e2=e2, s2=s2, l2=l2, lq=lq,
                    kl=kl(p), kl_fix=kl(p_fix))

    return {n: np.asarray(v) for n, v in f(q, k, z, m).items()}


def ulp32(x):
    return np.spacing(np.abs(x).astype(np.float32)).astype(np.float64)


def stats_rel(t, ref, sel=None):
    """t（FP32）对 ref（FP64）的相对误差：均值（带符号）、均方根；以及 ULP 带符号均值。"""
    t = t.astype(np.float64)
    if sel is not None:
        t, ref = t[sel], ref[sel]
    ok = ref != 0
    r = (t[ok] - ref[ok]) / np.abs(ref[ok])
    u = (t[ok] - ref[ok]) / ulp32(ref[ok])
    return {"rel_signed_mean": float(r.mean()), "rel_rms": float(np.sqrt((r * r).mean())),
            "ulp_signed_mean": float(u.mean()), "ulp_abs_mean": float(np.abs(u).mean()), "n": int(ok.sum())}


def analyse(o: dict, kl_sow: np.ndarray, target_sow: np.ndarray, causal: np.ndarray) -> dict:
    f64 = np.float64
    sel = causal & (o["p"] > 0)
    res = {}
    # 教师各步
    xs = o["xs"].astype(f64)
    hsel = np.broadcast_to(causal[None], xs.shape) & (xs > -1e30)
    res["teacher_exp"] = stats_rel(o["e"], np.exp(xs), hsel)
    res["teacher_head_sum"] = stats_rel(o["s"][..., 0], o["e"].astype(f64).sum(-1))
    s_t = o["s"].astype(f64)
    res["teacher_head_div"] = stats_rel(o["p_h"], o["e"].astype(f64) / s_t, hsel)
    res["teacher_head_rowsum_minus1_mean"] = float((o["p_h"].astype(f64).sum(-1) - 1).mean())
    res["teacher_cross_head_sum"] = stats_rel(o["a"], o["p_h"].astype(f64).sum(0), causal)
    res["teacher_row_total"] = stats_rel(o["tot"][..., 0], o["a"].astype(f64).sum(-1))
    res["teacher_norm_div"] = stats_rel(o["p"], o["a"].astype(f64) / o["tot"].astype(f64), sel)
    rs = o["p"].astype(f64).sum(-1) - 1
    res["target_rowsum_minus1"] = {"mean": float(rs.mean()), "frac_pos": float((rs > 0).mean())}
    rs_fix = o["p_fix"].astype(f64).sum(-1) - 1
    res["target_rowsum_minus1_refined_div"] = {"mean": float(rs_fix.mean()), "frac_pos": float((rs_fix > 0).mean())}
    # 只让某一步「精确」时的行和偏差（推算：只把这一步换成 FP64 再走完剩下的 FP32 步不现实，这里用一阶分解）
    # Σp − 1 = Σ(a/tot) − 1：tot 本身的相对误差 εtot 使 Σp 偏 −εtot；除法的平均相对偏差 εdiv 使 Σp 偏 +εdiv
    res["rowsum_first_order"] = {
        "from_row_total_sum": -res["teacher_row_total"]["rel_signed_mean"],
        "from_norm_div": res["teacher_norm_div"]["rel_signed_mean"],
    }
    # 学生各步
    zs = o["zs"].astype(f64)
    zsel = causal & (zs > -1e30)
    res["student_exp"] = stats_rel(o["e2"], np.exp(zs), zsel)
    res["student_sum"] = stats_rel(o["s2"][..., 0], o["e2"].astype(f64).sum(-1))
    res["student_log"] = stats_rel(o["l2"][..., 0], np.log(o["s2"][..., 0].astype(f64)))
    lse64 = np.log(np.where(zsel, np.exp(zs), 0.0).sum(-1))
    res["student_lse_total"] = {"abs_signed_mean": float((o["l2"][..., 0].astype(f64) - lse64).mean()),
                                "ulp_signed_mean": float(((o["l2"][..., 0].astype(f64) - lse64) / ulp32(lse64)).mean())}
    # 一阶分解 lse 的偏差：求和的相对偏差 → lse 偏 +εsum；exp 的平均相对偏差（按 e 加权）→ +εexp；log 自己的偏差
    w = np.where(zsel, np.exp(zs), 0.0)
    eexp = np.where(zsel, (o["e2"].astype(f64) - np.exp(zs)), 0.0).sum(-1) / w.sum(-1)
    res["lse_first_order"] = {
        "from_exp": float(eexp.mean()),
        "from_sum": res["student_sum"]["rel_signed_mean"],
        "from_log_abs": float((o["l2"][..., 0].astype(f64) - np.log(o["s2"][..., 0].astype(f64))).mean()),
    }
    # KL：TPU 图里的值、FP64 真值（同一份输入）、换写法后的值
    res["kl_row_mean_tpu"] = float(o["kl"].astype(f64).mean())
    res["kl_row_mean_tpu_refined_div"] = float(o["kl_fix"].astype(f64).mean())
    res["kl_row_mean_sow"] = float(kl_sow.astype(f64).mean())
    res["probe_vs_sow_kl_rel"] = float(o["kl"].astype(f64).mean() / kl_sow.astype(f64).mean() - 1)
    res["probe_target_equals_sow_frac"] = float((o["p"] == target_sow).mean())
    return res


def fp64_truth(q, k, z, m, causal):
    f64 = np.float64
    q, k = q.astype(f64), k.astype(f64)
    sel = causal & (m > -1e30)
    a = np.zeros((S, S))
    for h in range(q.shape[1]):
        x = q[:, h] @ k[:, h].T
        x = np.where(sel, x, -np.inf)
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        a += e / e.sum(-1, keepdims=True)
    p = a / np.maximum(a.sum(-1, keepdims=True), 1e-10)
    zz = np.where(sel, z.astype(f64), -np.inf)
    zz = zz - zz.max(-1, keepdims=True)
    lq = zz - np.log(np.exp(zz).sum(-1, keepdims=True))
    ok = sel & (p > 0)
    return np.where(ok, p * (np.log(np.maximum(p, 1e-10)) - np.where(ok, lq, 0.0)), 0.0).sum(-1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--make-inputs", action="store_true")
    p.add_argument("--kd", help="debug_sow_indexer 的 diag_kd_inputs.safetensors（--make-inputs）")
    p.add_argument("--inputs", help="--make-inputs 生成的 npz（本地或 gs://）")
    p.add_argument("--out", required=True)
    p.add_argument("--cpu", action="store_true", help="在 CPU 上跑：只验证脚本，不是 TPU 的数值")
    p.add_argument("--layers", default="", help="只跑这些层号（npz 里的序号，逗号分隔）")
    a, _rest = p.parse_known_args(argv)
    if a.make_inputs:
        make_inputs(a.kd, a.out)
        return 0
    import jax
    if not a.cpu and not jax.distributed.is_initialized():
        try:
            jax.distributed.initialize()
        except Exception as e:  # noqa: BLE001
            print("jax.distributed.initialize 跳过：", e)
    if a.cpu:
        jax.config.update("jax_platforms", "cpu")
    if jax.process_index() == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dev = jax.local_devices()[0]
    d = load_inputs(a.inputs)
    names = list(d["layers"])
    idx = [int(x) for x in a.layers.split(",")] if a.layers else range(len(names))
    causal = np.tril(np.ones((S, S), bool))
    out = {}
    for i in idx:
        q, k, z, m = (jax.device_put(d[f"l{i}_{n}"], dev) for n in "qkzm")
        o = tpu_steps(q, k, z, m)
        r = analyse(o, d[f"l{i}_kl_rows_sow"], d[f"l{i}_target_sow"], causal)
        kl64 = fp64_truth(d[f"l{i}_q"], d[f"l{i}_k"], d[f"l{i}_z"], d[f"l{i}_m"], causal)
        r["kl_row_mean_fp64"] = float(kl64.mean())
        r["kl_rel_tpu_vs_fp64"] = r["kl_row_mean_tpu"] / r["kl_row_mean_fp64"] - 1
        r["kl_rel_tpu_refined_div_vs_fp64"] = r["kl_row_mean_tpu_refined_div"] / r["kl_row_mean_fp64"] - 1
        r["kl_rel_sow_vs_fp64"] = r["kl_row_mean_sow"] / r["kl_row_mean_fp64"] - 1
        out[str(names[i])] = r
        print(names[i], json.dumps({x: y for x, y in r.items() if not isinstance(y, dict)}), flush=True)
    rec = {"what": __doc__.split("\n")[0], "platform": dev.platform, "device_kind": dev.device_kind, "jax": jax.__version__,
           "inputs": a.inputs, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "layers": out, "note": "CPU 上跑的只验证脚本，不是 TPU 的数值" if a.cpu else "TPU 一侧"}
    if jax.process_index() == 0:
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
