"""GPU 抓数回调的可选观测点：不依赖 GPU / Megatron 的部分（GEMM 清单的 torch 分支、参数检查）。没有 torch 时跳过。"""
import pytest
from helpers import ROOT

torch = pytest.importorskip("torch")


def test_gemm_inventory_records_input_dtypes():
    from janus.dump.megatron import _GemmInventory
    inv = _GemmInventory()
    inv.start()
    try:
        a32, b32 = torch.randn(4, 8), torch.randn(8, 3)
        a16, b16 = a32.to(torch.bfloat16), b32.to(torch.bfloat16)
        torch.mm(a32, b32)
        torch.mm(a16, b16)
        torch.mm(a16, b16)
        a32 + a32                                   # 非 GEMM 不记
    finally:
        inv.stop()
    calls = {(c["op"], c["a_dtype"], c["b_dtype"]): c["count"] for c in inv.summary()}
    assert calls[("torch.mm", "float32", "float32")] == 1
    assert calls[("torch.mm", "bfloat16", "bfloat16")] == 2
    assert all(op.startswith(("torch.", "te.")) for op, _, _ in calls)
    assert not inv.active


def test_unknown_observe_group_rejected():
    from janus.dump.megatron import JanusDump
    JanusDump("/tmp/unused", {}, observe="attn,moe,expw,gemm")
    with pytest.raises(ValueError, match="未知观测点"):
        JanusDump("/tmp/unused", {}, observe="attn,nope")


def test_pairs_include_fp32_params():
    """本身就是 FP32 的参数（--router-true-fp32 的路由权重）在 fp32_from_fp32_groups 里：参数即主权重，也要进抓数。"""
    from types import SimpleNamespace

    from janus.dump.megatron import JanusDump
    d = JanusDump("/tmp/unused", {})
    w16, m16 = torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2)
    r32 = torch.zeros(3)
    d._params = [("a.weight", w16), ("router.weight", r32)]
    d._opt = SimpleNamespace(float16_groups=[[w16]], fp32_from_float16_groups=[[m16]], fp32_from_fp32_groups=[[r32]],
                             optimizer="inner")
    got = {n: (p, mp) for n, p, mp, _ in d._pairs()}
    assert got["a.weight"][0] is w16 and got["a.weight"][1] is m16
    assert got["router.weight"][0] is r32 and got["router.weight"][1] is r32


def test_router_true_fp32_gemm_is_fp32_both_ways():
    """路由真 FP32 的 autograd 函数：前向、反向都是 FP32 × FP32；dgrad 舍回输入 dtype，wgrad 保持 FP32。"""
    import importlib.util
    path = ROOT / "models/hunyuan3/frameworks/megatron/router_fp32.py"
    spec = importlib.util.spec_from_file_location("router_fp32", path)
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 16, 64, generator=g).to(torch.bfloat16).requires_grad_()
    w = (torch.randn(8, 64, generator=g) * 0.02).requires_grad_()
    out = rf.router_gating_linear_true_fp32(x, w, None, torch.float32)
    assert out.dtype == torch.float32 and out.shape == (2, 16, 8)
    ref = x.detach().double().reshape(-1, 64) @ w.detach().double().t()
    assert float((out.reshape(-1, 8).double() - ref).norm() / ref.norm()) < 1e-6
    cot = torch.randn(out.shape, generator=g)
    out.backward(cot)
    assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.float32
    ref_w = cot.double().reshape(-1, 8).t() @ x.detach().double().reshape(-1, 64)
    assert float((w.grad.double() - ref_w).norm() / ref_w.norm()) < 1e-6
    with pytest.raises(RuntimeError, match="moe_router_dtype=fp32"):
        rf.router_gating_linear_true_fp32(x, w, None, torch.bfloat16)


def test_global_expert_name_and_kinds():
    from janus.dump.megatron import JanusDump, global_expert_name

    n = "decoder.layers.3.mlp.experts.linear_fc1.weight2"
    assert global_expert_name(n, 0, 4) == n                                  # rank 0：本地号 = 全局号
    assert global_expert_name(n, 3, 4) == "decoder.layers.3.mlp.experts.linear_fc1.weight14"
    assert global_expert_name("mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2.weight63", 1, 64).endswith("weight127")
    with pytest.raises(ValueError):
        global_expert_name("decoder.layers.3.mlp.router.weight", 1, 4)
    d = JanusDump("/tmp/x", {}, full=True, gather_experts=True, full_kinds="main_param,grad")
    assert d.skip is None and d._want_kind("main_param") and not d._want_kind("exp_avg")
    assert JanusDump("/tmp/x", {})._want_kind("exp_avg")
    with pytest.raises(ValueError):
        JanusDump("/tmp/x", {}, full_kinds="main_params")
