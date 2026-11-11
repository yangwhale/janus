"""L1 单算子的 CPU 参照：FP64 真值正确舍入到输出 dtype、GPU 舍入方式的 CPU 模拟、R-ratio、
打平规则、解析 VJP。纯 numpy，不依赖 torch（GPU 侧回放见 models/hunyuan3/frameworks/megatron/l1_ops.py）。

约定：被测输出 out 保持原始 dtype；真值 truth 是 float64；「正确舍入」= 从 FP64 一次 RNE 舍到目标格点。
FP64 真值本身离数学真值约 1e-16（相对），舍到 FP32 / BF16 时只有落在格点中点附近 1e-16 以内才会舍错，忽略不计。
"""
from __future__ import annotations

import math

import ml_dtypes
import numpy as np

from janus import metrics as M

BF16 = np.dtype(ml_dtypes.bfloat16)

# 有效位数 p（含隐含位）、最小正规数的 frexp 指数下限、最大有限值
_FMT = {
    np.dtype(np.float32): (24, -125, float(np.finfo(np.float32).max)),
    BF16: (8, -125, float(ml_dtypes.finfo(ml_dtypes.bfloat16).max)),
    np.dtype(np.float16): (11, -13, float(np.finfo(np.float16).max)),
}


def round_from_f64(x, dtype) -> np.ndarray:
    """FP64 → 目标浮点类型，一次 RNE（含次正规数与溢出）。不经过 FP32 中转，避免 BF16 的两次舍入。

    做法：x = m·2^e（m ∈ [0.5, 1)），格距 2^(e−p)；次正规区格距固定，指数截在下限。np.rint 是就近取偶。
    """
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return np.asarray(x, dtype=np.float64).copy()
    p, emin, fmax = _FMT[dtype]
    x = np.asarray(x, dtype=np.float64)
    out = np.array(x, copy=True)
    fin = np.isfinite(x) & (x != 0.0)
    xf = x[fin]
    _, e = np.frexp(xf)
    e = np.maximum(e, emin)
    q = np.ldexp(np.rint(np.ldexp(xf, p - e)), e - p)
    q[np.abs(q) > fmax] = np.inf * np.sign(q[np.abs(q) > fmax])
    out[fin] = q
    return out.astype(dtype)            # 已在格点上，这次转换是精确的


def same_bits(a, b) -> np.ndarray:
    """逐元素位模式相同；两边都是 NaN 也算相同（NaN 的载荷不比）。"""
    a, b = np.asarray(a), np.asarray(b)
    M._check_shape(a, b)
    if a.dtype != b.dtype:
        raise TypeError(f"逐位比要求同一 dtype: {a.dtype} vs {b.dtype}")
    if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_:
        return (a == b).ravel()
    from janus.dtypes import float_bits
    _, ut = float_bits(a.dtype)
    eq = a.view(ut).ravel() == b.view(ut).ravel()
    nan = np.isnan(a.astype(np.float64)).ravel() & np.isnan(b.astype(np.float64)).ravel()
    return eq | nan


def match_frac(a, b) -> float:
    s = same_bits(a, b)
    return 1.0 if s.size == 0 else float(np.count_nonzero(s) / s.size)


def r_ratio(test, ref, truth) -> dict:
    """三方比（R-ratio）：被测→真值、参照→真值、被测→参照。

    R = rel_l2(被测, 真值) / rel_l2(参照, 真值)。R ≈ 1：被测和参照离真值一样远（差别只是舍入顺序）；
    R 明显 > 1：被测多了误差源（多舍一次、低精度累加）。两边都是 0 给 1，参照为 0 而被测不为 0 给 inf。
    """
    e_t, e_r = M.rel_l2(test, truth), M.rel_l2(ref, truth)
    r = 1.0 if (e_t == 0.0 and e_r == 0.0) else (math.inf if e_r == 0.0 else e_t / e_r)
    return {"r_ratio": r, "rel_l2_test": e_t, "rel_l2_ref": e_r, "rel_l2_test_ref": M.rel_l2(test, ref),
            "cosine_test": M.cosine(test, truth), "cosine_ref": M.cosine(ref, truth)}


# ---------------------------------------------------------------- GEMM 舍入方式的 CPU 模拟

def _round_f32_rz(x: np.ndarray) -> np.ndarray:
    """FP64 → FP32 向零舍入（截断）。只处理有限值。"""
    f = x.astype(np.float32)                         # RNE
    over = np.abs(f.astype(np.float64)) > np.abs(x)
    f[over] = np.nextafter(f[over], np.float32(0))
    return f


def _exponent(a: np.ndarray) -> np.ndarray:
    """⌊log2|a|⌋（a = 1.xxx·2^e 的 e）；0 给一个很小的数，不参与取最大。"""
    _, e = np.frexp(a)
    return np.where(a == 0.0, -100000, e - 1)


def gemm_emulate(x, w, mode: str = "fp64", block: int = 0, group: int = 16, frac_bits: int = 25,
                 rounding: str = "rz", rows=None) -> np.ndarray:
    """模拟「BF16 输入、FP32 累加」的 GEMM：out[i, j] = Σ_k x[i, k]·w[j, k]，返回 FP32（未舍到输出 dtype）。

    BF16 × BF16 的乘积只有 16 位有效位，在 FP32 / FP64 里都是精确的，所以差别只在累加顺序与舍入：
    - fp64：FP64 累加后一次舍到 FP32（≈ 正确舍入的点积）；
    - seq：FP32 逐项顺序累加，每加一项 RNE 一次；
    - block：K 切成 block 段，段内顺序累加，段间再顺序相加；
    - tc：张量核的分组累加模型——每 group 个乘积与累加器一起对齐到最大指数 emax，低于 2^(emax − frac_bits)
      的部分截掉（向零），精确相加，再按 rounding（rz 截断 / rne）写回 FP32。乘积的指数取两个输入指数之和
      （不按乘积规格化后的指数），累加器取自身指数。思路见 Fasi 等 2021《Numerical behavior of NVIDIA tensor cores》；
      具体参数（group、frac_bits、rounding）要靠逐位命中率去试，B200 上 BF16→FP32 的实测见 docs/dev/design/0003-l1-op-replay.md。
    rows：只算这些行（模拟较慢时抽样用）。
    """
    x = np.asarray(x).astype(np.float64)
    w = np.asarray(w).astype(np.float64)
    if rows is not None:
        x = x[np.asarray(rows)]
    if mode == "fp64":
        return (x @ w.T).astype(np.float32)
    m, k = x.shape
    n = w.shape[0]
    out = np.empty((m, n), np.float32)
    step = max(1, (1 << 24) // max(1, n * (group if mode == "tc" else 1)))
    for i0 in range(0, m, step):
        xs = x[i0:i0 + step]
        if mode == "seq":
            acc = np.zeros((xs.shape[0], n), np.float32)
            for kk in range(k):
                acc = acc + np.outer(xs[:, kk], w[:, kk]).astype(np.float32)
        elif mode == "block":
            acc = np.zeros((xs.shape[0], n), np.float32)
            for b0 in range(0, k, block):
                part = np.zeros_like(acc)
                for kk in range(b0, min(k, b0 + block)):
                    part = part + np.outer(xs[:, kk], w[:, kk]).astype(np.float32)
                acc = acc + part
        elif mode == "tc":
            acc = np.zeros((xs.shape[0], n), np.float64)
            ex, ew = _exponent(xs), _exponent(w)
            for g0 in range(0, k, group):
                t = xs[:, None, g0:g0 + group] * w[None, :, g0:g0 + group]     # 精确乘积
                et = ex[:, None, g0:g0 + group] + ew[None, :, g0:g0 + group]    # 乘积指数 = 输入指数之和
                emax = np.maximum(et.max(axis=-1), _exponent(acc))[..., None]
                emax = np.where(emax < -10000, 0, emax)                         # 全是 0 的组
                unit = np.ldexp(1.0, emax - frac_bits)
                terms = np.concatenate([acc[..., None], t], axis=-1)
                s = np.sum(np.trunc(terms / unit) * unit, axis=-1)              # 截断对齐后精确相加（不超过 53 位）
                acc = (_round_f32_rz(s) if rounding == "rz" else s.astype(np.float32)).astype(np.float64)
        else:
            raise ValueError(f"不认识的模拟方式 {mode}")
        out[i0:i0 + step] = acc
    return out


def rmsnorm_emulate(x, w, eps: float, out_dtype) -> np.ndarray:
    """模拟「FP32 里算 RMSNorm、最后一次舍到输出 dtype」：r = rsqrt(mean(x²) + eps)，y = (x·r)·w。
    归约在 FP64 里做再舍到 FP32（相当于一个足够准的 FP32 归约）。"""
    xf = np.asarray(x).astype(np.float32)
    ms = (xf.astype(np.float64) ** 2).mean(axis=-1, keepdims=True).astype(np.float32)
    r = (1.0 / np.sqrt(ms.astype(np.float64) + eps)).astype(np.float32)
    y = (xf * r) * np.asarray(w).astype(np.float32)
    return round_from_f64(y.astype(np.float64), out_dtype)


# ---------------------------------------------------------------- FP64 公式（真值）

def rmsnorm_f64(x, w, eps: float) -> np.ndarray:
    x = np.asarray(x).astype(np.float64)
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * r * np.asarray(w).astype(np.float64)


def sigmoid_f64(x) -> np.ndarray:
    x = np.asarray(x).astype(np.float64)
    e = np.exp(-np.abs(x))                       # 只算 exp(−|x|)，不会溢出
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu_f64(x) -> np.ndarray:
    return np.asarray(x).astype(np.float64) * sigmoid_f64(x)


def softmax_f64(x) -> np.ndarray:
    x = np.asarray(x).astype(np.float64)
    with np.errstate(invalid="ignore"):
        z = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return z / z.sum(axis=-1, keepdims=True)


def router_probs_f64(logits, bias, k: int, scale: float, idx=None):
    """sigmoid 路由：s = σ(z)，按 s + bias 选 top-k（idx 给出时用它），p = s_sel / Σ s_sel · scale。返回 (p, idx)。"""
    s = sigmoid_f64(logits)
    if idx is None:
        idx = topk_stable(s + np.asarray(bias, np.float64), k)
    sel = np.take_along_axis(s, idx, axis=-1)
    return sel / (sel.sum(axis=-1, keepdims=True) + 1e-20) * scale, idx


# ---------------------------------------------------------------- 打平规则

def topk_stable(scores, k: int) -> np.ndarray:
    """CPU 稳定排序的 top-k：分数降序，打平时下标小的在前（jax.lax.top_k 默认 is_stable=True 的规则）。返回 [行, k] 下标（按名次）。"""
    s = np.asarray(scores).astype(np.float64)
    return np.argsort(-s, axis=-1, kind="stable")[..., :k]


def boundary_ties(scores, k: int) -> np.ndarray:
    """每行第 k 名与第 k+1 名分数逐位相等的行（这些行的选中集合取决于打平规则）。"""
    s = np.asarray(scores)
    top = -np.sort(-s, axis=-1)
    return np.nonzero(top[..., k - 1] == top[..., k])[0]


# ---------------------------------------------------------------- 解析 VJP（FP64）

def rmsnorm_vjp_f64(x, w, eps: float, u):
    """y = x·r·w，r = (mean(x²) + eps)^(-1/2)。返回 (∂L/∂x, ∂L/∂w)，u 是 cotangent。"""
    x, w, u = (np.asarray(a).astype(np.float64) for a in (x, w, u))
    h = x.shape[-1]
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    uw = u * w
    gx = r * uw - x * r ** 3 * (uw * x).sum(axis=-1, keepdims=True) / h
    gw = (u * x * r).reshape(-1, h).sum(axis=0)
    return gx, gw


def router_probs_vjp_f64(logits, idx, scale: float, u):
    """p = σ(z)_sel / Σ σ(z)_sel · scale（选中集合 idx 固定）。u：[行, k] 对 p 的 cotangent。返回 ∂L/∂z（[行, E]）。"""
    z = np.asarray(logits).astype(np.float64)
    u = np.asarray(u).astype(np.float64)
    s = sigmoid_f64(z)
    sel = np.take_along_axis(s, idx, axis=-1)
    tot = sel.sum(axis=-1, keepdims=True) + 1e-20
    gs = scale * (u / tot - (u * sel).sum(axis=-1, keepdims=True) / tot ** 2)
    gz = np.zeros_like(z)
    np.put_along_axis(gz, idx, gs * sel * (1.0 - sel), axis=-1)
    return gz


def directional_fd(f, x, v, u, eps: float = 1e-6) -> float:
    """数值核对：uᵀ·(f(x + εv) − f(x − εv)) / 2ε，与解析 VJP 的 ⟨g, v⟩ 比（FP64 中心差分）。"""
    return float(np.sum(np.asarray(u, np.float64) * (f(x + eps * v) - f(x - eps * v))) / (2 * eps))
