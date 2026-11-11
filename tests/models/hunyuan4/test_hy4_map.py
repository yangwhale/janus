"""混元 4 的 HF ↔ MaxText 映射工具（models/hunyuan4/model/hy4_map.py、make_tiny_hf.hf_shapes）：不依赖 MaxText 的部分。

分支映射本身（mapping_and_hooks）要 MaxText，在 frameworks/maxtext/cpu_smoke.py 里整条链路核。
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/hunyuan4/model"
sys.path.insert(0, str(MODEL))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, MODEL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hy4_map = _load("hy4_map")
make_tiny_hf = _load("make_tiny_hf")

# tencent/Hy4-preview config.json（revision 705d81ee）截 4 层后与形状有关的项
CFG = {
    "hidden_size": 6144, "vocab_size": 120832, "num_attention_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512,
    "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256, "index_n_heads": 32, "index_head_dim": 128,
    "n_routed_experts": 256, "n_shared_experts": 1, "moe_intermediate_size": 2048, "intermediate_size": 18432, "hc_mult": 4,
    "num_hidden_layers": 4, "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "indexer_types": ["full", "full", "shared", "shared"], "num_nextn_predict_layers": 1,
}


@pytest.mark.parametrize("stage", sorted(set(hy4_map.STAGES) - {"mini", "mid"}))
def test_stage_param_counts_match_checkpoint_headers(stage):
    layers, mtp = hy4_map.STAGES[stage]
    shapes = make_tiny_hf.hf_shapes(hy4_map.stage_hf_config(CFG, layers), layers, mtp)
    assert sum(int(np.prod(s)) for s in shapes.values()) == hy4_map.STAGE_PARAMS[stage]


# hy4-mini（MaxText hf_model_configs.hy4_mini_dict 的维度）：不是截层，是另一个尺寸的随机初始化模型。
MINI = dict(CFG, hidden_size=1024, num_attention_heads=8, q_lora_rank=512, n_routed_experts=16, moe_intermediate_size=256,
            intermediate_size=3072, num_hidden_layers=7, mlp_layer_types=["dense"] + ["sparse"] * 6,
            indexer_types=["full", "full", "shared", "shared", "shared", "full", "shared"])


def test_mini_param_count():
    shapes = make_tiny_hf.hf_shapes(MINI, 7, 1)
    assert sum(int(np.prod(s)) for s in shapes.values()) == hy4_map.STAGE_PARAMS["mini"] == 427_783_489
    assert hy4_map.STAGES["mini"] == (7, 1)


# hy4-mid（MaxText hf_model_configs.hy4_mid_dict 的维度）：hy4-mini 的结构放到生产的宽度与注意力，专家宽 512。
MID = dict(MINI, hidden_size=6144, num_attention_heads=64, q_lora_rank=2048, intermediate_size=18432,
           moe_intermediate_size=512)


def test_mid_param_count():
    shapes = make_tiny_hf.hf_shapes(MID, 7, 1)
    assert sum(int(np.prod(s)) for s in shapes.values()) == hy4_map.STAGE_PARAMS["mid"] == 5_189_678_849
    assert hy4_map.STAGES["mid"] == (7, 1)


def test_stage_tensor_names_selects_layers_and_mtp():
    names = list(make_tiny_hf.hf_shapes(CFG, 4, 1))
    a = hy4_map.stage_tensor_names(names, 1, 0)
    assert len(a) == 31 and not any("layers.1." in n or "mtp_layers" in n for n in a)
    d = hy4_map.stage_tensor_names(names, 2, 1)
    assert any(n.startswith("model.mtp_layers.0.") for n in d) and not any("layers.2." in n for n in d)
    assert hy4_map.stage_tensor_names(names, 4, 1) == sorted(names)


def test_stage_hf_config_truncates_per_layer_lists():
    c = hy4_map.stage_hf_config(CFG, 2)
    assert c["num_hidden_layers"] == 2 and c["indexer_types"] == ["full", "full"] and CFG["num_hidden_layers"] == 4
    with pytest.raises(ValueError):
        hy4_map.stage_hf_config(CFG, 5)


def test_hf_dtype_table():
    fp32 = [n for n in make_tiny_hf.hf_shapes(CFG, 4, 1) if hy4_map.hf_dtype(n) == "float32"]
    # 发布的检查点：34 个 FP32 张量（hc_head 3 个；每层 iHC 6 个 + sink 1 个；三个 MoE 层的路由偏置）；MTP 层的偏置与 sink 是 BF16
    assert len(fp32) == 34
    assert hy4_map.hf_dtype("model.mtp_layers.0.mlp.gate.e_score_correction_bias") == "bfloat16"
    assert hy4_map.hf_dtype("model.mtp_layers.0.self_attn.learnable_sink_param") == "bfloat16"
    assert hy4_map.hf_dtype("model.layers.0.self_attn.learnable_sink_param") == "float32"


def test_gpu_names():
    g = hy4_map.gpu_name
    assert g("model.layers.0.hc_attn_layer.hc_pre.hc_fn") == "hf.model.layers.0.attn_hc.fn"
    assert g("model.layers.0.hc_mlp_layer.hc_pre.hc_scale") == "hf.model.layers.0.ffn_hc.scale"
    assert g("model.hc_head.hc_head_base") == "hf.model.hc_head.hc_base"
    assert g("model.layers.0.self_attn.linear_gate.weight") == "hf.model.layers.0.self_attn.gate_proj.weight"
    assert g("model.layers.0.self_attn.learnable_sink_param") == "hf.model.layers.0.self_attn.sinks"
    assert g("lm_head.weight") == "hf.lm_head.weight"


def test_gpu_rank_pieces_follow_fsdp2_chunks():
    x = np.arange(21 * 3).reshape(21, 3)
    pieces = hy4_map.gpu_rank_pieces("hf.model.layers.0.mlp.up_proj.weight", x)
    assert [p.shape[0] for p in pieces] == [3, 3, 3, 3, 3, 3, 3, 0]      # ceil(21 / 8) = 3 行一片
    assert np.array_equal(np.concatenate(pieces), x)
    rep = hy4_map.gpu_rank_pieces("hf.model.layers.0.attn_hc.fn", x)
    assert all(p is x for p in rep)
    for n in ("hf.model.hc_head.hc_fn", "hf.model.layers.0.self_attn.sinks",
              "hf.model.layers.0.self_attn.indexer.weights_proj.weight", "hf.model.layers.0.self_attn.indexer.k_norm.bias"):
        assert all(p is x for p in hy4_map.gpu_rank_pieces(n, x)), n
    assert hy4_map.gpu_rank_pieces("hf.model.layers.0.self_attn.indexer.wk.weight", x)[0].shape[0] == 3


def _forward_rope(x, rope):
    """正向去交错（独立于 hy4_map 再写一遍）：最后 rope 个位置 (2i, 2i+1) → [偶, 奇]。"""
    head, tail = x[..., :-rope], x[..., -rope:]
    return np.concatenate([head, tail[..., 0::2], tail[..., 1::2]], axis=-1)


def test_undo_rope_inverts_forward_and_is_bitwise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 4, 256)).astype(np.float32)
    y = _forward_rope(x, 64)
    assert not np.array_equal(y, x)
    assert np.array_equal(hy4_map.undo_rope(y, 64, True), x)
    assert np.array_equal(hy4_map.undo_rope(y, 64, False), y)
    perm = hy4_map.deinterleave_perm(256, 64)
    assert np.array_equal(x[..., perm], y)


@pytest.mark.parametrize("key,hf_shape,target", [
    ("params-decoder-dense_layers-self_attention-wq_b-kernel", (64 * 256, 2048), (2048, 64, 256)),
    ("params-decoder-dense_layers-self_attention-out-kernel", (6144, 64 * 256), (64, 256, 6144)),
    ("params-decoder-dense_layers-self_attention-wkv_a-kernel", (576, 6144), (6144, 576)),
    ("params-decoder-dense_layers-mlp-wo-kernel", (6144, 18432), (18432, 6144)),
])
def test_kernel_inverse(key, hf_shape, target):
    rng = np.random.default_rng(1)
    small = tuple(max(1, d // 64) for d in hf_shape)                         # 缩小维度，保持结构
    tgt = tuple(max(1, d // 64) for d in target)
    if key.endswith("out-kernel"):
        small, tgt = (tgt[2], tgt[0] * tgt[1]), tgt
    elif key.endswith("wq_b-kernel"):
        tgt = (tgt[0], tgt[1], 8)
        small = (tgt[1] * 8, tgt[0])
    x = rng.standard_normal(small).astype(np.float32)
    rope = 4 if hy4_map.hf_kind(key) == "kernel_rope" else 0
    y = x.T.reshape(tgt)                                                     # 分支 HOOK_FN 的 kernel 变换
    if rope:
        y = _forward_rope(y, rope)
    back = hy4_map.piece_to_hf(key, y, rope, bool(rope), x.shape)
    assert np.array_equal(back, x)


def test_assemble_hc_and_experts():
    rng = np.random.default_rng(2)
    n, D = 4, 3
    fn = rng.standard_normal((2 * n, n * D)).astype(np.float32)
    pieces = {"params-decoder-dense_layers-mhc_attention-pre_alpha": fn[:n].T.reshape(n * D, n),
              "params-decoder-dense_layers-mhc_attention-post_alpha": fn[n:].T.reshape(n * D, n)}
    assert np.array_equal(hy4_map.assemble_hf(pieces, fn.shape, 64, True), fn)
    base = rng.standard_normal(2 * n).astype(np.float32)
    pieces = {"params-decoder-dense_layers-mhc_mlp-pre_beta": base[:n], "params-decoder-dense_layers-mhc_mlp-post_beta": base[n:]}
    assert np.array_equal(hy4_map.assemble_hf(pieces, base.shape, 64, True), base)
    E, inter, d = 3, 2, 5
    gu = rng.standard_normal((E, 2 * inter, d)).astype(np.float32)
    k = "params-decoder-moe_layers_remainder-layers_0-DeepSeekMoeBlock_0-MoeBlock_0-"
    pieces = {k + "wi_0": np.transpose(gu[:, :inter], (0, 2, 1)), k + "wi_1": np.transpose(gu[:, inter:], (0, 2, 1))}
    assert np.array_equal(hy4_map.assemble_hf(pieces, gu.shape, 64, True), gu)


def test_hf_kind_covers_every_branch_key_suffix():
    # 分支 HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN 的 per_layer 表里每种键都归到一种反变换
    for suffix, kind in [("self_attention-indexer-k_norm-bias", "rope_only"), ("self_attention-indexer-wk-kernel", "kernel_rope"),
                         ("mhc_attention-pre_alpha_scale", "hc_pre_alpha_scale"), ("DeepSeekMoeBlock_0-MoeBlock_0-wo", "experts_out"),
                         ("DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel", "kernel"), ("self_attention-sinks", "identity"),
                         ("pre_self_attention_layer_norm-scale", "identity")]:
        assert hy4_map.hf_kind("params-decoder-dense_layers-" + suffix) == kind, suffix
    assert hy4_map.hf_kind("params-decoder-hc_head-hc_fn") == "kernel"
    assert re.match(r"^hf\.", hy4_map.gpu_name("model.norm.weight"))
