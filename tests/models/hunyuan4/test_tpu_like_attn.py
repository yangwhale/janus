"""HY4_TPU_LIKE_ATTNK：带 sink 的稀疏注意力照 tokamax splash 的舍入写法（tpu_like_attn.py）。只要 torch，CPU 上跑。

- FP64 输入时没有舍入，四处换不换都与 autograd 的基线（hy4_mcore.sparse_sink_attention 的 FP32 q·k 口径，照抄在
  _baseline 里）一致：验证每一处换的公式在数学上等价。
- BF16 输入时四处都不换，与基线只差 FP32 量级的求和顺序；每一处单开都改变它该改变的量，不碰不该碰的量。
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_H4 = Path(__file__).resolve().parents[3] / "models" / "hunyuan4"
sys.path.insert(0, str(_H4 / "frameworks" / "megatron"))
sys.path.insert(0, str(_H4 / "analysis"))

from splash_pv_rounding_check import emulate  # noqa: E402
from tpu_like_attn import EXTRA, PARTS, TPULikeSinkAttention, parts_from_env  # noqa: E402

S, H, D, DV, K = 48, 2, 32, 16, 12
SCALE = 1.0 / 16  # 2 的幂，同混元 4 mini


def _inputs(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, H, S, D, generator=g) * 3).to(dtype)
    k = (torch.randn(1, H, S, D, generator=g) * 3).to(dtype)
    v = torch.randn(1, H, S, DV, generator=g).to(dtype)
    topk = torch.stack([torch.randperm(S, generator=g)[:K] for _ in range(S)])
    allowed = torch.zeros(S, S, dtype=torch.bool)
    allowed.scatter_(-1, topk, True)
    allowed &= torch.ones(S, S, dtype=torch.bool).tril()
    allowed = allowed[None, None]
    sinks = torch.randn(H, generator=g).double() if dtype == torch.float64 else torch.randn(H, generator=g)
    d_out = torch.randn(1, H, S, DV, generator=g).to(dtype)
    return q, k, v, allowed, sinks, d_out


def _baseline(q, k, v, allowed, sinks):
    """hy4_mcore.sparse_sink_attention（scores_fp32=True）的同一串运算，[b, h, s, d] 布局。"""
    b, h, sq, _ = q.shape
    scores = (
        torch.matmul(
            q.float() if q.dtype != torch.float64 else q,
            (k.float() if k.dtype != torch.float64 else k).transpose(-1, -2),
        )
        * SCALE
    )
    scores = scores.masked_fill(~allowed, float("-inf"))
    sink = sinks.to(scores.dtype).view(1, h, 1, 1).expand(b, h, sq, 1)
    comb = torch.cat([scores, sink], dim=-1)
    comb = comb - comb.max(dim=-1, keepdim=True).values
    probs = torch.softmax(comb, dim=-1)[..., :-1].to(v.dtype)
    return torch.matmul(probs, v)


def _run(fn, q, k, v, allowed, sinks, d_out):
    q, k, v, sinks = (t.clone().requires_grad_(True) for t in (q, k, v, sinks))
    out = fn(q, k, v, allowed, sinks)
    out.backward(d_out)
    return out.detach(), q.grad, k.grad, v.grad, sinks.grad


def _tpu(parts, block=16):
    return lambda q, k, v, a, s: TPULikeSinkAttention.apply(q, k, v, a, s, SCALE, frozenset(parts), block)[0]


def _rel(x, y):
    x, y = x.double(), y.double()
    return float((x - y).norm() / y.norm())


def test_fp64_every_part_equals_baseline():
    args = _inputs(torch.float64)
    ref = _run(_baseline, *args)
    for parts in [(), *[(p,) for p in PARTS + EXTRA], PARTS]:
        got = _run(_tpu(parts), *args)
        for name, a, b in zip(("out", "dq", "dk", "dv", "dsinks"), got, ref):
            assert _rel(a, b) < 1e-10, (parts, name, _rel(a, b))


def test_bf16_no_parts_matches_baseline():
    args = _inputs(torch.bfloat16)
    ref = _run(_baseline, *args)
    got = _run(_tpu(()), *args)
    for name, a, b in zip(("out", "dq", "dk", "dv", "dsinks"), got, ref):
        assert _rel(a, b) < 5e-3, (name, _rel(a, b))  # 只差 FP32 求和顺序，经 BF16 输出放大到一两个 ulp


def test_bf16_each_part_changes_what_it_should():
    args = _inputs(torch.bfloat16)
    base = _run(_tpu(()), *args)
    out0, dq0, dk0, dv0, ds0 = base
    for p in ("dp", "di", "ds"):
        out, dq, dk, dv, dsk = _run(_tpu((p,)), *args)
        assert torch.equal(out, out0), p  # 反向的三处不碰前向
        assert torch.equal(dv, dv0), p  # dv 两边一样，不在开关里
        assert not torch.equal(dq, dq0) and not torch.equal(dk, dk0), p
        if p != "dp":
            assert torch.equal(dsk, ds0), p  # sink 梯度照 GPU 写法
    for p in ("pv", "pv32"):
        out, dq, dk, dv, dsk = _run(_tpu((p,)), *args)
        assert not torch.equal(out, out0), p
        assert torch.equal(dv, dv0), p  # 前向那一处不碰反向的写法（反向用的是 FP32 概率）


def test_pv_matches_splash_emulation_checked_on_tpu():
    """pv 与 analysis/splash_pv_rounding_check.py 的 block 写法（对 TPU 实跑的 splash 输出逐元素相同 99.8% 以上）一致。"""
    g = torch.Generator().manual_seed(1)
    q = (torch.randn(1, H, S, D, generator=g) * 3 * SCALE).bfloat16()  # 那边的 q 已乘缩放
    k = (torch.randn(1, H, S, D, generator=g) * 3).bfloat16()
    v = torch.randn(1, H, S, DV, generator=g).bfloat16()
    sinks = torch.randn(H, generator=g)
    causal = torch.ones(S, S, dtype=torch.bool).tril()[None, None]
    out, _ = TPULikeSinkAttention.apply(q, k, v, causal, sinks, 1.0, frozenset({"pv"}), 16)
    ref = emulate(q[0], k[0], v[0], sinks, block=16)["block"]
    assert float((out[0].float() == ref).float().mean()) > 0.99


def test_all_parts_close_to_fp64():
    args = _inputs(torch.bfloat16)
    ref = _run(_baseline, *(t.double() if t.dtype != torch.bool else t for t in args))
    got = _run(_tpu(PARTS), *args)
    for name, a, b in zip(("out", "dq", "dk", "dv", "dsinks"), got, ref):
        assert _rel(a, b) < 2e-2, (name, _rel(a, b))


def test_max_logit_excludes_sink():
    q, k, v, allowed, sinks, _ = _inputs(torch.bfloat16)
    _, m = TPULikeSinkAttention.apply(q, k, v, allowed, sinks + 100, SCALE, frozenset(PARTS))
    scores = (q.float() @ k.float().transpose(-1, -2) * SCALE).masked_fill(~allowed, float("-inf"))
    assert torch.equal(m, scores.amax(dim=(0, 2, 3)))


def test_parts_from_env(monkeypatch):
    monkeypatch.delenv("HY4_TPU_LIKE_ATTNK", raising=False)
    assert parts_from_env() == frozenset()
    monkeypatch.setenv("HY4_TPU_LIKE_ATTNK", "1")
    monkeypatch.delenv("HY4_TPU_LIKE_ATTNK_PARTS", raising=False)
    assert parts_from_env() == frozenset(PARTS)
    monkeypatch.setenv("HY4_TPU_LIKE_ATTNK_PARTS", "dp, di")
    assert parts_from_env() == frozenset({"dp", "di"})
    monkeypatch.setenv("HY4_TPU_LIKE_ATTNK_PARTS", "pv32")
    assert parts_from_env() == frozenset({"pv32"})
    for bad in ("dq", "pv,pv32"):
        monkeypatch.setenv("HY4_TPU_LIKE_ATTNK_PARTS", bad)
        with pytest.raises(ValueError):
            parts_from_env()
