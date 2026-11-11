"""指标计算。定义见 docs/reference/metrics.md。

约定：a 是被测，b 是参照（golden）；张量一律拉直；除 ULP 外全部在 FP64 里算。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from janus.dtypes import float_bits

CHUNK = 1 << 23   # 分块计算时每块 800 万元素：FP64 临时数组约 64 MB


def _f64(x) -> np.ndarray:
    return np.asarray(x).astype(np.float64).ravel()


def _check_shape(a, b):
    if np.shape(a) != np.shape(b):
        raise ValueError(f"形状不同: {np.shape(a)} vs {np.shape(b)}")


@dataclass
class Finite:
    """两边非有限值（NaN / ±Inf）的位置是否一致。不一致时其余指标没有意义。"""
    a_nonfinite: int
    b_nonfinite: int
    mismatch: int          # 非有限值位置或取值（NaN / +Inf / -Inf）不同的元素数

    @property
    def ok(self) -> bool:
        return self.mismatch == 0


def finite_check(a, b) -> Finite:
    _check_shape(a, b)
    x, y = _f64(a), _f64(b)
    fx, fy = ~np.isfinite(x), ~np.isfinite(y)
    same = (np.isnan(x) & np.isnan(y)) | (x == y)
    mismatch = int(np.count_nonzero((fx | fy) & ~same))
    return Finite(int(fx.sum()), int(fy.sum()), mismatch)


class _Clean(np.ndarray):
    """已转成 FP64、已剔掉非有限值的一维数组（_finite_pair 的输出）。再传进各指标时不重复检查与转换。"""


def _finite_pair(a, b):
    """去掉两边同为非有限值的位置；位置不一致时抛 ValueError。"""
    if isinstance(a, _Clean) and isinstance(b, _Clean):
        return a, b
    fc = finite_check(a, b)
    if not fc.ok:
        raise ValueError(f"非有限值位置不一致: {fc.mismatch} 个元素")
    x, y = _f64(a), _f64(b)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.all():
        return x.view(_Clean), y.view(_Clean)
    return x[keep].view(_Clean), y[keep].view(_Clean)


def rel_l2(a, b) -> float:
    """‖a − b‖₂ / ‖b‖₂。参照全零时：a 也全零给 0，否则给 inf。"""
    x, y = _finite_pair(a, b)
    num, den = np.linalg.norm(x - y), np.linalg.norm(y)
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return float(num / den)


def cosine(a, b) -> float:
    """a·b / (‖a‖₂‖b‖₂)。两边都全零给 1；只有一边全零给 0。"""
    x, y = _finite_pair(a, b)
    na, nb = np.linalg.norm(x), np.linalg.norm(y)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(x, y) / (na * nb))


def mean_shift(a, b) -> float:
    """|mean(a) − mean(b)| / std(b)。参照标准差为 0 时：均值相同给 0，否则给 inf。"""
    x, y = _finite_pair(a, b)
    if y.size == 0:
        return 0.0
    d, s = abs(x.mean() - y.mean()), y.std()
    if s == 0.0:
        return 0.0 if d == 0.0 else math.inf
    return float(d / s)


def max_abs(a, b):
    """(max|aᵢ − bᵢ|, 坐标)。坐标按原始形状给出；空张量给 (0.0, None)。"""
    _check_shape(a, b)
    x, y = _finite_pair(a, b)
    if x.size == 0:
        return 0.0, None
    diff = np.abs(_f64(a) - _f64(b))
    diff[~np.isfinite(diff)] = -1.0          # 同为非有限值的位置不参与
    i = int(np.argmax(diff))
    return float(diff[i]), tuple(int(k) for k in np.unravel_index(i, np.shape(a)))


def out_of_tol_ratio(a, b, atol: float, rtol: float) -> float:
    """|aᵢ − bᵢ| > atol + rtol·|bᵢ| 的元素比例。"""
    x, y = _finite_pair(a, b)
    if x.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(x - y) > atol + rtol * np.abs(y)) / x.size)


# 参考档（docs/reference/judging.md「两档门槛」）要的「被测对参照」统计。超差元素的容差是推断：
# rtol = 2⁻⁶（BF16 在 1 附近的间距是 2⁻⁷，取两格）；atol = 2⁻⁶ × rms(参照)，防止近零元素一律算超差。
REF_RTOL = 2.0 ** -6
REF_ATOL_REL = 2.0 ** -6


def reference_stats(a, b, with_l2: bool = True) -> dict:
    """被测 a 对参照 b 的参考档统计（FP64）：
    oot_frac  = |aᵢ − bᵢ| > REF_ATOL_REL·rms(b) + REF_RTOL·|bᵢ| 的元素比例；
    mean_diff = mean|a − b| / mean|b|（参照全零时：差也全零给 0，否则 inf）；
    norm_max  = max|a − b| / max|b|（同上）；with_l2 时另给 rel_l2、cosine。"""
    x, y = _finite_pair(a, b)
    if x.size == 0:
        out = {"oot_frac": 0.0, "mean_diff": 0.0, "norm_max": 0.0}
        return {**out, "rel_l2": 0.0, "cosine": 1.0} if with_l2 else out
    d, ay = np.abs(x - y), np.abs(y)
    rms = float(np.sqrt(np.mean(y * y)))

    def _ratio(num, den):
        return 0.0 if num == 0.0 else (math.inf if den == 0.0 else float(num / den))

    out = {"oot_frac": float(np.count_nonzero(d > REF_ATOL_REL * rms + REF_RTOL * ay) / x.size),
           "mean_diff": _ratio(float(d.mean()), float(ay.mean())), "norm_max": _ratio(float(d.max()), float(ay.max()))}
    if with_l2:
        out.update(rel_l2=rel_l2(x, y), cosine=cosine(x, y))
    return out


def norm_ratio(a, b) -> float:
    """‖a‖₂ / ‖b‖₂。参照全零时：a 也全零给 1，否则 inf。"""
    x, y = _finite_pair(a, b)
    na, nb = np.linalg.norm(x), np.linalg.norm(y)
    if nb == 0.0:
        return 1.0 if na == 0.0 else math.inf
    return float(na / nb)


def std_ratio(a, b) -> float:
    """std(a) / std(b)（总体标准差）。参照标准差为 0 时：a 的也为 0 给 1，否则 inf。
    看分布宽度有没有系统性变宽 / 变窄（梯度整体缩放、漏乘系数），与 norm_ratio 互补：均值大时范数比被均值主导。"""
    x, y = _finite_pair(a, b)
    if y.size == 0:
        return 1.0
    sa, sb = float(x.std()), float(y.std())
    if sb == 0.0:
        return 1.0 if sa == 0.0 else math.inf
    return sa / sb


def zero_frac_diff(a, b) -> float:
    """|a 中零元素比例 − b 中零元素比例|（±0 都算零；两边同为非有限值的位置不计入分母）。
    抓多出来 / 少掉的稀疏：漏了 mask、多了 dropout、专家没收到 token 时梯度整片为 0。"""
    x, y = _finite_pair(a, b)
    if y.size == 0:
        return 0.0
    return abs(np.count_nonzero(x == 0.0) - np.count_nonzero(y == 0.0)) / x.size


REL_ERR_EPS_FACTOR = 1e-3


def rel_err_p99(a, b) -> float:
    """逐元素相对误差 |aᵢ − bᵢ| / max(|aᵢ|, |bᵢ|, eps) 的 99 分位，eps = 1e-3 × RMS(b)。

    分母取两边绝对值较大者，对称且不会因为参照恰好接近 0 而爆掉；eps 再兜住两边都接近 0 的元素
    （量级远小于张量典型值的元素，相对误差没有意义）。两边都为 0 的元素给 0。"""
    x, y = _finite_pair(a, b)
    if x.size == 0:
        return 0.0
    eps = REL_ERR_EPS_FACTOR * math.sqrt(float(np.dot(y, y)) / y.size)
    out = np.empty(x.size, dtype=np.float64)
    for i in range(0, x.size, CHUNK):
        xs, ys = x[i:i + CHUNK], y[i:i + CHUNK]
        den = np.maximum(np.maximum(np.abs(xs), np.abs(ys)), eps)
        num = np.abs(xs - ys)
        r = np.zeros_like(num)
        np.divide(num, den, out=r, where=den > 0.0)     # den = 0 只在两边都为 0（且 eps = 0）时出现，此时 num 也是 0
        out[i:i + CHUNK] = r
    return float(np.percentile(out, 99))


def kl_div(a, b, chunk_rows: int = 256) -> float:
    """把被测 a、参照 b 的 logits 按最后一维当作每行 softmax 分布，算 KL(参照 ‖ 被测) 的行平均：
    mean_rows Σⱼ p_b(j)·(log p_b(j) − log p_a(j))。FP64，按行 log-softmax；按 chunk_rows 行分块，避免整张 FP64 拷贝。

    方向：期望取在参照分布上——参照认为可能的 token，被测给的概率低了才罚得重（与「用参照当真值」一致）。
    非有限值：两边位置必须一致（否则抛 ValueError）；-inf（被屏蔽的 token）按概率 0 处理，NaN 不允许。"""
    _check_shape(a, b)
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 0:
        raise ValueError("kl 需要至少一维（最后一维是词表）")
    fc = finite_check(a, b)
    if not fc.ok:
        raise ValueError(f"非有限值位置不一致: {fc.mismatch} 个元素")
    v = a.shape[-1]
    a2, b2 = a.reshape(-1, v), b.reshape(-1, v)
    n = a2.shape[0]
    if n == 0 or v == 0:
        return 0.0
    total = 0.0
    for i in range(0, n, chunk_rows):
        x = a2[i:i + chunk_rows].astype(np.float64)
        y = b2[i:i + chunk_rows].astype(np.float64)
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError("kl 不接受 NaN")
        lx = x - x.max(axis=1, keepdims=True)
        lx -= np.log(np.exp(lx).sum(axis=1, keepdims=True))
        ly = y - y.max(axis=1, keepdims=True)
        ly -= np.log(np.exp(ly).sum(axis=1, keepdims=True))
        py = np.exp(ly)
        # 参照概率 > 0 而被测为 -inf（概率 0）时 KL 为 inf
        if np.any((py > 0.0) & ~np.isfinite(lx)):
            return math.inf
        with np.errstate(invalid="ignore"):
            d = np.where(py > 0.0, py * (ly - np.where(np.isfinite(lx), lx, 0.0)), 0.0)
        total += float(d.sum())
    return total / n


def normalized_dot_error(s_hat, s_ref, abs_sum):
    """点积 / 归约误差按 Σ|xᵢyᵢ| 归一：|ŝ − s| / Σ|xᵢyᵢ|。

    ŝ 是被测的归约结果（如 GPU 上一次 GEMM 的一个输出元素），s 是同一组输入的高精度参照（FP64），
    abs_sum 是同一组输入的 Σ|xᵢyᵢ|（GEMM 时即 |X| @ |Y|，FP64 算）。分母是这一次归约里舍入误差的自然尺度：
    正负抵消让 |s| 接近 0 时，按 |s| 归一的相对误差会虚高，按 Σ|xᵢyᵢ| 归一则只和累加顺序、累加精度有关。
    三个参数可以是标量或同形状数组（逐元素算）；abs_sum = 0 时：差为 0 给 0，否则 inf。"""
    sh = np.asarray(s_hat, dtype=np.float64)
    sr = np.asarray(s_ref, dtype=np.float64)
    den = np.asarray(abs_sum, dtype=np.float64)
    if np.any(den < 0):
        raise ValueError("abs_sum 不能为负")
    num = np.abs(sh - sr)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.where(num == 0.0, 0.0, np.inf))
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------- ULP

def _ordered(bits: np.ndarray, width: int) -> np.ndarray:
    """浮点位模式 → 单调整数：数值越大整数越大，+0 与 −0 都映射到 0。"""
    sign = np.uint64(1) << np.uint64(width - 1)
    u = bits.astype(np.uint64)
    mag = (u & (sign - np.uint64(1))).astype(np.int64)
    neg = (u & sign) != 0
    return np.where(neg, -mag, mag)


ULP_BUCKETS = (("0", 0, 0), ("1", 1, 1), ("2-4", 2, 4), ("5-8", 5, 8), ("9-16", 9, 16), ("17+", 17, None))


@dataclass
class UlpSummary:
    n: int
    zero_frac: float       # 逐位相同（0 ULP）的比例
    p50: float
    p99: float
    p999: float
    max: int
    cross_zero: int        # 两边符号相反且都不为 0 的元素数，单列，不计入分位数
    one_frac: float = 0.0  # 恰好差 1 ULP 的比例
    buckets: dict = field(default_factory=dict)   # 分桶占比：0 / 1 / 2-4 / 5-8 / 9-16 / 17+（分母同 n）


def ulp_distance(a, b) -> np.ndarray:
    """逐元素 ULP 距离（按位模式数格）。两边必须是同一种浮点 dtype；非有限值位置给 -1。

    跨零元素（符号相反且都不为 0）也按单调整数相减给出，但汇总时单列。
    """
    a, b = np.asarray(a), np.asarray(b)
    _check_shape(a, b)
    if a.dtype != b.dtype:
        raise TypeError(f"ULP 要求同一 dtype: {a.dtype} vs {b.dtype}")
    width, ut = float_bits(a.dtype)
    oa = _ordered(a.view(ut).ravel(), width)
    ob = _ordered(b.view(ut).ravel(), width)
    d = np.abs(oa - ob)
    nonfinite = ~(np.isfinite(a.astype(np.float64)).ravel() & np.isfinite(b.astype(np.float64)).ravel())
    d[nonfinite] = -1
    return d


def ulp_summary(a, b) -> UlpSummary:
    d = ulp_distance(a, b)
    x, y = np.asarray(a).astype(np.float64).ravel(), np.asarray(b).astype(np.float64).ravel()
    cross = (np.sign(x) * np.sign(y) < 0)
    valid = (d >= 0) & ~cross
    v = d[valid]
    if v.size == 0:
        return UlpSummary(0, 1.0, 0.0, 0.0, 0.0, 0, int(cross.sum()), 0.0,
                          {k: (1.0 if k == "0" else 0.0) for k, _, _ in ULP_BUCKETS})
    buckets = ulp_buckets(v)
    return UlpSummary(
        n=int(v.size), zero_frac=buckets["0"],
        p50=float(np.percentile(v, 50)), p99=float(np.percentile(v, 99)),
        p999=float(np.percentile(v, 99.9)), max=int(v.max()), cross_zero=int(cross.sum()),
        one_frac=buckets["1"], buckets=buckets)


def ulp_buckets(v: np.ndarray) -> dict:
    """非负 ULP 距离的分桶占比（ULP_BUCKETS），各桶之和为 1；空数组给全 0。"""
    v = np.asarray(v)
    if v.size == 0:
        return {k: 0.0 for k, _, _ in ULP_BUCKETS}
    out = {}
    for k, lo, hi in ULP_BUCKETS:
        m = (v >= lo) if hi is None else ((v >= lo) & (v <= hi))
        out[k] = float(np.count_nonzero(m) / v.size)
    return out


# ---------------------------------------------------------------- 离散

def exact_match_frac(a, b) -> float:
    """整数张量逐元素完全相同的比例。"""
    _check_shape(a, b)
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return 1.0 if a.size == 0 else float(np.count_nonzero(a == b) / a.size)


@dataclass
class TopkAgreement:
    n: int
    set_frac: float        # 选中集合相同的行比例
    order_frac: float      # 选中顺序也相同的行比例
    rows_differ: np.ndarray  # 集合不同的行号


def topk_agreement(idx_a, idx_b) -> TopkAgreement:
    """idx_*: [行, k] 的专家下标。"""
    a, b = np.asarray(idx_a), np.asarray(idx_b)
    _check_shape(a, b)
    a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    same_set = np.all(np.sort(a2, axis=1) == np.sort(b2, axis=1), axis=1)
    same_order = np.all(a2 == b2, axis=1)
    n = a2.shape[0]
    return TopkAgreement(n, float(same_set.mean()) if n else 1.0,
                         float(same_order.mean()) if n else 1.0, np.nonzero(~same_set)[0])


def topk_margin(scores, k: int, rows=None) -> np.ndarray:
    """每行第 k 名与第 k+1 名的分差（FP64）。rows 给出时只算这些行。"""
    s = np.asarray(scores).astype(np.float64)
    s = s.reshape(-1, s.shape[-1])
    if rows is not None:
        s = s[np.asarray(rows, dtype=np.int64)]
    top = -np.sort(-s, axis=1)
    return top[:, k - 1] - top[:, k]


# ---------------------------------------------------------------- loss 序列

@dataclass
class SeriesStats:
    d0: float | None       # 第 0 步的相对差；窗口不含第 0 步时为 None
    signed_mean: float
    mare: float
    max_step: float
    max_step_at: int
    window: tuple[int, int]
    denominator: str
    nonfinite_steps: list[int]
    signed_mean_se: float = 0.0  # 带符号均值的标准误，见 mean_se


def mean_se(x) -> float:
    """均值的标准误，取两种估计的较大者（偏保守）：
    ① 一阶自相关折算：n_eff = n·(1−ρ)/(1+ρ)（ρ 截到 [0, 0.99]），SE = std / √n_eff；
    ② 分批均值（n ≥ 100 时）：切成 20 段，SE = 段均值的标准差 / √20，能吃进一阶看不到的长程相关。
    少于 2 个样本返回 0。"""
    a = np.asarray(x, dtype=np.float64)
    n = a.size
    if n < 2:
        return 0.0
    sd = float(a.std(ddof=1))
    if sd == 0.0:
        return 0.0
    c = a - a.mean()
    rho = min(max(float(np.dot(c[:-1], c[1:]) / np.dot(c, c)), 0.0), 0.99)
    se = sd / math.sqrt(max(n * (1 - rho) / (1 + rho), 1.0))
    if n >= 100:
        m = n // 20
        se = max(se, float(a[: m * 20].reshape(20, m).mean(axis=1).std(ddof=1)) / math.sqrt(20))
    return se


def series_stats(test: dict[int, float], ref: dict[int, float], window: tuple[int, int],
                 denominator: str = "golden") -> SeriesStats:
    """逐步相对差 dₜ = (testₜ − refₜ) / |分母|。

    test / ref：步号 → 值（已按步号约定对齐）。window：[start, end)，统计用；第 0 步单独报。
    denominator：'golden'（参照，默认）、'test' 或 'mean'（两边均值 |t + r| / 2）。任一边非有限值的步记入 nonfinite_steps，不参与统计。
    'mean' 给逐步相对涨落大的量用（如 grad_norm，逐步 ±3%）：以一边为分母时，带符号均值有 +E[δ²] 的偏置，
    交换两边不再只差一个符号，噪声对按哪个顺序列会改变 N（docs/reference/judging.md「逐步涨落大的量用均值作分母」）。
    两边在窗口内缺步时抛 ValueError。
    """
    if denominator not in ("golden", "test", "mean"):
        raise ValueError("denominator 只能是 golden、test 或 mean")
    start, end = window
    steps = list(range(start, end))
    missing = [s for s in steps if s not in test or s not in ref]
    if missing:
        raise ValueError(f"窗口内缺步: {missing[:5]}{'…' if len(missing) > 5 else ''}")

    def rel(s):
        t, r = float(test[s]), float(ref[s])
        den = abs(r) if denominator == "golden" else abs(t) if denominator == "test" else abs(t + r) / 2
        return (t - r) / den

    nonfinite, ds = [], []
    for s in steps:
        t, r = float(test[s]), float(ref[s])
        if not (math.isfinite(t) and math.isfinite(r)):
            nonfinite.append(s)
            continue
        ds.append((s, rel(s)))
    d0 = rel(0) if (0 in test and 0 in ref and math.isfinite(test[0]) and math.isfinite(ref[0])) else None
    stat = [d for s, d in ds if s != 0]
    if not stat:
        return SeriesStats(d0, 0.0, 0.0, 0.0, -1, window, denominator, nonfinite)
    arr = np.array(stat, dtype=np.float64)
    at_list = [s for s, _ in ds if s != 0]
    i = int(np.argmax(np.abs(arr)))
    return SeriesStats(d0, float(arr.mean()), float(np.abs(arr).mean()), float(abs(arr[i])),
                       at_list[i], window, denominator, nonfinite, mean_se(arr))


# ---------------------------------------------------------------- 一遍算完（大张量用）

def pair_stats(a, b) -> dict | None:
    """分块一遍算出 rel_l2、cosine、mean_shift、max_abs（含坐标）、norm_ratio，全部在 FP64 里累加。
    结果与逐个指标函数一致（同样的定义）；两边有非有限值时返回 None，调用方退回逐个函数（它们会报位置不一致）。"""
    _check_shape(a, b)
    fa, fb = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    n = fa.size
    if n == 0:
        return None
    c = float(np.asarray(fb[: min(n, 4096)], dtype=np.float64).mean())     # 平移量，减轻方差的相消误差
    sdd = sxx = syy = sxy = 0.0
    sx = sy = syc = syc2 = 0.0
    best, best_i = -1.0, 0
    for i in range(0, n, CHUNK):
        x = fa[i:i + CHUNK].astype(np.float64)
        y = fb[i:i + CHUNK].astype(np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return None
        d = x - y
        sdd += float(np.dot(d, d))
        sxx += float(np.dot(x, x))
        syy += float(np.dot(y, y))
        sxy += float(np.dot(x, y))
        sx += float(x.sum())
        sy += float(y.sum())
        yc = y - c
        syc += float(yc.sum())
        syc2 += float(np.dot(yc, yc))
        ad = np.abs(d)
        j = int(np.argmax(ad))
        if ad[j] > best:
            best, best_i = float(ad[j]), i + j
    na, nb, num = math.sqrt(sxx), math.sqrt(syy), math.sqrt(sdd)
    out = {}
    out["rel_l2"] = (0.0 if num == 0.0 else math.inf) if nb == 0.0 else num / nb
    out["cosine"] = 1.0 if (na == 0.0 and nb == 0.0) else (0.0 if (na == 0.0 or nb == 0.0) else sxy / (na * nb))
    var = max(syc2 / n - (syc / n) ** 2, 0.0)
    dmean, s = abs(sx / n - sy / n), math.sqrt(var)
    out["mean_shift"] = (0.0 if dmean == 0.0 else math.inf) if s == 0.0 else dmean / s
    out["max_abs"] = best
    out["max_abs_at"] = tuple(int(k) for k in np.unravel_index(best_i, np.shape(a)))
    out["norm_ratio"] = (1.0 if na == 0.0 else math.inf) if nb == 0.0 else na / nb
    return out
