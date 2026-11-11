"""models/hunyuan3/model/name_map.yaml 与 janus/dump/name_map.py：覆盖、形状、来回逐位不变。"""

import ml_dtypes
import numpy as np
import pytest
from helpers import ROOT  # noqa: E402  仓库根

from janus.dump.name_map import NameMap, NameMapError, apply_ops, invert_ops, leaf_index

NM_PATH = ROOT / "models" / "hunyuan3" / "model" / "name_map.yaml"
SMALL = dict(hidden=32, ffn=48, moe_ffn=8, experts=16, topk=4, heads=8, kv_heads=2, head_dim=4, vocab=64)


def small():
    nm = NameMap.load(NM_PATH)
    return NameMap({**nm.spec, "dims": {**nm.dims, **SMALL},
                    "megatron": {**nm.spec["megatron"], "local_experts": SMALL["experts"] // nm.ep}})


def expected_hf_names(layers=4, first_dense=1, experts=192):
    """按混元 3 HF 结构列出全部张量名（截 4 层、去 MTP）。"""
    out = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(layers):
        p = f"model.layers.{i}."
        out |= {p + x for x in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
                                 "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                 "self_attn.q_norm.weight", "self_attn.k_norm.weight")}
        if i < first_dense:
            out |= {p + f"mlp.{x}_proj.weight" for x in ("gate", "up", "down")}
        else:
            out |= {p + "mlp.router.gate.weight", p + "mlp.expert_bias"}
            out |= {p + f"mlp.shared_mlp.{x}_proj.weight" for x in ("gate", "up", "down")}
            out |= {p + f"mlp.experts.{e}.{x}_proj.weight" for e in range(experts) for x in ("gate", "up", "down")}
    return out


def test_covers_all_hf_tensors():
    nm = NameMap.load(NM_PATH)
    want = expected_hf_names()
    assert len(want) == 1781                      # 与 GPU 侧 HF 目录 tensor_sha256.json 的条目数一致
    assert nm.hf_names() == want
    assert len(nm.entries) == 3 + 4 * 6 + 2 + 3 * (4 + 2 * 192)   # 非层内 3；每层注意力 5 + MLP 前归一化 1；稠密 MLP 2；MoE 每层 4 + 专家 2×192


def test_leaf_shapes_match_maxtext_abstract_model():
    """形状来自 CPU 上 MaxText 抽象模型（hy3-4layers.yml，scan_layers=True，param_scan_axis=1）。"""
    ls = NameMap.load(NM_PATH).leaf_shapes()
    assert len(ls) == 30
    assert ls["params/token_embedder/embedding"] == (120832, 4096)
    assert ls["params/decoder/logits_dense/kernel"] == (4096, 120832)
    assert ls["params/decoder/dense_layers/self_attention/query/kernel"] == (4096, 1, 64, 128)
    assert ls["params/decoder/moe_layers/self_attention/out/kernel"] == (64, 3, 128, 4096)
    assert ls["params/decoder/moe_layers/Hunyuan3MoeBlock_0/MoeBlock_0/wi_0"] == (192, 3, 4096, 1536)
    assert ls["MoEBiasVar/decoder/moe_layers/Hunyuan3MoeBlock_0/MoeBlock_0/gate/bias"] == (3, 192)
    assert ls["params/decoder/moe_layers/pre_self_attention_layer_norm/scale"] == (4096, 3)


def test_megatron_expert_names():
    nm = NameMap.load(NM_PATH)
    e0 = nm.by_canonical("layers.1.mlp.experts.0.fc1.weight")
    e25 = nm.by_canonical("layers.2.mlp.experts.25.fc2.weight")
    assert nm.megatron_name(e0) == "decoder.layers.1.mlp.experts.linear_fc1.weight0" and nm.expert_owner(e0) == 0
    assert nm.megatron_name(e25) == "decoder.layers.2.mlp.experts.linear_fc2.weight1" and nm.expert_owner(e25) == 1
    with pytest.raises(NameMapError):
        nm.megatron_name(e25, rank=0)


def test_qkv_by_group_layout():
    nm = small()
    ent = nm.by_canonical("layers.0.attn.qkv.weight")
    H, G, hd, D = SMALL["heads"], SMALL["kv_heads"], SMALL["head_dim"], SMALL["hidden"]
    q = np.arange(H * hd * D, dtype=np.float32).reshape(H * hd, D)
    k = -np.arange(G * hd * D, dtype=np.float32).reshape(G * hd, D) - 1
    v = k - 1000
    m = nm.to_megatron(ent, [q, k, v])
    per = (H // G + 2) * hd
    assert m.shape == ((H + 2 * G) * hd, D)
    for g in range(G):                            # 每组：该组 H/G 个 q 头、1 个 k 头、1 个 v 头
        blk = m[g * per:(g + 1) * per]
        assert np.array_equal(blk[: H // G * hd], q[g * H // G * hd:(g + 1) * H // G * hd])
        assert np.array_equal(blk[H // G * hd: H // G * hd + hd], k[g * hd:(g + 1) * hd])
        assert np.array_equal(blk[-hd:], v[g * hd:(g + 1) * hd])
    back = nm.from_megatron(ent, m, nm.hf_shapes(ent))
    assert all(np.array_equal(a, b) for a, b in zip(back, (q, k, v)))


def test_roundtrip_hf_maxtext_megatron_bitwise():
    nm = small()
    rng = np.random.default_rng(0)
    leaves = {p: np.full(s, np.nan, np.float32) for p, s in nm.leaf_shapes().items()}
    hf = {}
    for ent in nm.entries:
        for h, shp in zip(ent.hf, nm.hf_shapes(ent)):
            hf[h] = rng.standard_normal(shp).astype(np.float32 if h.endswith("expert_bias") else ml_dtypes.bfloat16)
        for pc in ent.pieces:
            leaves[pc.path][leaf_index(pc.index, leaves[pc.path].ndim)] = apply_ops(hf[pc.hf].astype(np.float32), pc.ops)
    assert not any(np.isnan(v).any() for v in leaves.values())          # 每个元素都被写到
    for ent in nm.entries:
        shapes = nm.hf_shapes(ent)
        for pc in ent.pieces:
            got = nm.hf_from_leaf(leaves[pc.path], pc, shapes[ent.hf.index(pc.hf)]).astype(hf[pc.hf].dtype)
            assert got.tobytes() == hf[pc.hf].tobytes(), pc.hf
        m = nm.to_megatron(ent, [hf[h] for h in ent.hf])
        assert all(a.tobytes() == hf[h].tobytes() for a, h in zip(nm.from_megatron(ent, m, shapes), ent.hf))


def test_invert_ops():
    x = np.arange(2 * 3 * 4).reshape(6, 4)
    ops = [("T", (1, 0)), ("reshape", (4, 2, 3))]
    assert np.array_equal(invert_ops(apply_ops(x, ops), ops, x.shape), x)


def test_wrong_qkv_order_is_visible():
    """故意改坏：q / k 来源顺序写反，拼出的 Megatron qkv 就不同（核对能抓到映射写错）。"""
    nm = small()
    ent = nm.by_canonical("layers.0.attn.qkv.weight")
    q, k, v = (np.random.default_rng(i).standard_normal(s).astype(np.float32) for i, s in enumerate(nm.hf_shapes(ent)))
    good = nm.to_megatron(ent, [q, k, v])
    assert not np.array_equal(good, nm.to_megatron(ent, [q, v, k]))
