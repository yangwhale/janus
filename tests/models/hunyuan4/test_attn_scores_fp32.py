"""hy4_attn_scores_fp32：稀疏注意力的 q·k 在 FP32 里算（不先舍成 BF16）。要 torch 与 Megatron-Core，没装时跳过。"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "frameworks" / "megatron"))


def _inputs(seed=0, s=64, h=2, d=32, dv=16, k=16):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(s, 1, h, d, generator=g) * 3).to(torch.bfloat16)
    kk = (torch.randn(s, 1, h, d, generator=g) * 3).to(torch.bfloat16)
    v = torch.randn(s, 1, h, dv, generator=g).to(torch.bfloat16)
    topk = torch.stack([torch.randperm(s, generator=g)[:k] for _ in range(s)])[None]
    sinks = torch.zeros(h)
    return q, kk, v, topk, sinks


def test_fp32_scores_closer_to_fp64_reference():
    from hy4_mcore import sparse_sink_attention

    q, k, v, topk, sinks = _inputs()
    scale = q.shape[-1] ** -0.5
    ref = sparse_sink_attention(q.double(), k.double(), v.double(), topk, sinks.double(), scale, scores_fp32=False)
    bf = sparse_sink_attention(q, k, v, topk, sinks, scale, scores_fp32=False).double()
    fp = sparse_sink_attention(q, k, v, topk, sinks, scale, scores_fp32=True).double()
    err = lambda x: float((x - ref).norm() / ref.norm())  # noqa: E731
    assert not torch.equal(bf, fp)  # 开关确实改变了结果
    assert err(fp) < err(bf)


def test_flag_reaches_attention():
    import inspect

    import hy4_mcore

    src = inspect.getsource(hy4_mcore.Hy4DSAttention.forward)
    assert "hy4_attn_scores_fp32" in src
