"""索引器 KD 归一化偏差的主机侧核对（TPU `debug_sow_indexer` 写出的第 0 行输入）。

1. 教师分布每行总和在数学上恰好是头数 8（2 的幂）。FP32 里先精确求和再舍入一次时 S = 8，行和无偏；
   FP32 求和落到 8 以下时（8 以下的格距是 8 以上的一半），用 S 归一化后 Σp > 1；整行共用一个舍入后的倒数时，
   S 比 8 小奇数个格的行，1/S 恰好在两格中点稍上，总是向上舍，再加一份正偏差。逐元素 IEEE 除法没有后一份。
2. 广义 KL：Σ p log(p/q) − Σp + Σq。在归一化的 p、q 上与 KL 相同；对 p 整体放大 (1+δ)、log q 整体平移 ε 一阶不变。
   用 TPU 图里的 p 与 log q 在 FP64 里算两种写法，对 FP64 真值比。

    python models/hunyuan4/analysis/kd_norm_emulate.py <diag_kd_inputs.safetensors> <out.json>
"""

from __future__ import annotations

import json
import sys

import numpy as np

S = 4096


def teacher(q, k, m):
    a = np.zeros((S, S))
    for h in range(q.shape[1]):
        x = np.where(m, q[:, h] @ k[:, h].T, -np.inf)
        x -= x.max(-1, keepdims=True)
        e = np.exp(x)
        a += e / e.sum(-1, keepdims=True)
    return a


def rowsum_bias(a32, s32):
    pd = (a32 / s32).astype(np.float64)
    r = np.float32(1) / s32
    pr = (a32 * r).astype(np.float64)
    return {"frac_S_below_8": float((s32 < 8).mean()),
            "rowsum_minus1_ieee_div": float((pd.sum(-1) - 1).mean()),
            "rowsum_minus1_shared_reciprocal": float((pr.sum(-1) - 1).mean()),
            "reciprocal_rel_err_mean": float((r.astype(np.float64) * s32.astype(np.float64) - 1).mean())}


def main():
    from safetensors import safe_open

    f = safe_open(sys.argv[1], "np")
    out = {}
    for L in sorted({k.rsplit("/debug_kd", 1)[0] for k in f.keys()}):
        def g(n, L=L):
            return f.get_tensor(f"{L}/debug_kd_{n}/0")

        q = g("query").reshape(S, 8, 256).astype(np.float64)
        k = g("key").reshape(S, 8, 256).astype(np.float64)
        m = (g("indexer_mask").reshape(S, S) > -1e20) & np.tril(np.ones((S, S), bool))
        a = teacher(q, k, m)
        a32 = a.astype(np.float32)
        tp = g("target").reshape(S, S).astype(np.float64)
        res = {"rowsum_minus1_tpu_sow": float((tp.sum(-1) - 1).mean()), "emulate": {
            "exact_sum_then_round": rowsum_bias(a32, a32.astype(np.float64).sum(-1, keepdims=True).astype(np.float32)),
            "fp32_pairwise_sum": rowsum_bias(a32, a32.sum(-1, keepdims=True, dtype=np.float32)),
            "fp32_sequential_sum": rowsum_bias(a32, np.cumsum(a32, -1, dtype=np.float32)[:, -1:]),
            "sum_one_ulp_below_8": rowsum_bias(a32, np.full((S, 1), np.nextafter(np.float32(8), np.float32(0)))),
        }}
        z = np.where(m, g("indexer_score").reshape(S, S).astype(np.float64), -np.inf)
        z -= z.max(-1, keepdims=True)
        lq = z - np.log(np.exp(z).sum(-1, keepdims=True))
        p64 = a / a.sum(-1, keepdims=True)
        ok = m & (p64 > 0)
        kl64 = np.where(ok, p64 * (np.log(np.maximum(p64, 1e-10)) - np.where(ok, lq, 0)), 0).sum(-1).mean()
        tl = g("log_indexer").reshape(S, S).astype(np.float64)
        okt = m & (tp > 0)
        kl_t = np.where(okt, tp * (np.log(np.maximum(tp, 1e-10)) - np.where(okt, tl, 0)), 0).sum(-1)
        sum_p = np.where(m, tp, 0).sum(-1)
        sum_q = np.where(m, np.exp(np.where(m, tl, -np.inf)), 0).sum(-1)
        gen_t = kl_t - (sum_p - 1) + (sum_q - 1)
        res.update(kl_fp64=float(kl64), kl_tpu_p_logq_mcore_formula_rel=float(kl_t.mean() / kl64 - 1),
                   kl_tpu_p_logq_generalized_rel=float(gen_t.mean() / kl64 - 1))
        out[L] = res
        print(L, json.dumps(res))
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
