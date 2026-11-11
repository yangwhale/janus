"""混元 4 参数量公式：用公开 config 的取值算，对上 checkpoint 头核过的精确值与官方口径。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("hy4_sizing", ROOT / "models/hunyuan4/analysis/sizing.py")
sizing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sizing)

# Hugging Face tencent/Hy4-preview config.json（revision 705d81ee）里与参数量有关的项
FULL = ["full", "full", "shared", "shared", "shared"] + (["full", "shared", "shared", "shared"] * 18) + ["full"]
CFG = {
    "hidden_size": 6144,
    "vocab_size": 120832,
    "num_attention_heads": 64,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "intermediate_size": 18432,
    "hc_mult": 4,
    "num_hidden_layers": 78,
    "indexer_types": FULL,
    "mlp_layer_types": ["dense"] + ["sparse"] * 77,
}


def test_indexer_pattern_matches_config():
    assert len(FULL) == 78
    assert [i for i, t in enumerate(FULL) if t == "full"][:4] == [0, 1, 5, 9]


def test_totals_match_checkpoint_and_model_card():
    p = sizing.module_params(CFG)
    backbone = sizing.outer(p) + sum(sizing.layer_params(CFG, p, i) for i in range(78))
    assert backbone == 769_907_408_797  # 模型卡 770B
    assert sizing.mtp_params(CFG, p) == 10_053_583_936  # 模型卡 10B
    active = sizing.outer(p) + sum(sizing.layer_params(CFG, p, i, active=True) for i in range(78))
    assert active == 49_057_546_141  # 模型卡 49B
    assert sizing.mtp_params(CFG, p, active=True) == 691_897_408  # 模型卡 0.7B


def test_stage_sizes():
    p = sizing.module_params(CFG)
    got = {
        n: sizing.truncated(CFG, p, k, m)["params"]
        for n, k, m in [("A", 1, False), ("B", 2, False), ("C", 3, False), ("D2", 2, True)]
    }
    assert got == {"A": 2_100_089_689, "B": 12_078_550_957, "C": 22_047_640_321, "D2": 22_132_134_893}
