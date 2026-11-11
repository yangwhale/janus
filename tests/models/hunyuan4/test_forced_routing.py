"""build_forced_routing.py：GPU 抓到的路由选择 → MaxText 路由回放文件（不要 MaxText）。"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from janus.store import read_safetensors_header, read_safetensors_tensor, write_safetensors

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "build_forced_routing", ROOT / "models/hunyuan4/frameworks/maxtext/build_forced_routing.py"
)
bfr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfr)

S, E, K = 6, 8, 2


def _row(path, seed, tokens, break_map=False):
    rng = np.random.default_rng(seed)
    t = {"input.tokens": tokens.reshape(1, S).astype(np.int64)}
    for name in ("layers.1", "layers.2", "mtp_layers.0"):
        raw = rng.uniform(0.1, 0.9, (S, E)).astype(np.float32)
        bias = rng.uniform(-0.01, 0.01, E).astype(np.float32)
        sel = np.argsort(-(raw + bias), axis=1)[:, :K]
        rm = np.zeros((S, E), bool)
        np.put_along_axis(rm, sel, True, axis=1)
        p = np.where(rm, raw, 0)
        p = (p / p.sum(1, keepdims=True) * 2.827).astype(np.float32)
        if break_map and name == "layers.2":
            rm[0] = False
            rm[0, np.argsort(raw[0] + bias)[:K]] = True  # 换成分数最低的 K 个，与 probs 的非零位置不一致
        t[f"hf.model.{name}.mlp.gate.routing_map"] = rm
        t[f"hf.model.{name}.mlp.gate.probs"] = p
        t[f"hf.model.{name}.mlp.gate.expert_bias"] = bias
    for name in ("layers.0", "layers.1", "layers.2", "mtp_layers.0"):
        top = np.full((1, S, 3), -1, np.int64)
        for q in range(S):
            sel = rng.permutation(q + 1)[:3]
            top[0, q, : len(sel)] = sel
        t[f"hf.model.{name}.self_attn.topk"] = top
    write_safetensors(path, t)
    return t


def test_build_slots_order_and_tokens(tmp_path):
    rows = [_row(tmp_path / f"step_1_row_{i:02d}.safetensors", i, np.arange(S) + 100 * i) for i in range(2)]
    files = sorted(tmp_path.glob("step_*_row_*.safetensors"))
    out, info = bfr.build(files, num_decoder_layers=3)
    ids = out["forced_routed_experts"]
    assert ids.shape == (2, S, 4, K) and ids.dtype == np.int32
    assert info["main_moe_layers"] == [1, 2] and info["mtp_layers"] == [0] and info["top_k"] == K
    assert np.all(ids[:, :, 0] == -1)  # 稠密层槽位
    for r, t in enumerate(rows):
        for slot, name in ((1, "layers.1"), (2, "layers.2"), (3, "mtp_layers.0")):
            p = t[f"hf.model.{name}.mlp.gate.probs"]
            rm = t[f"hf.model.{name}.mlp.gate.routing_map"]
            got = ids[r, :, slot]
            # 选中的集合 = routing_map，顺序按 GPU 权重降序
            assert np.array_equal(np.sort(got, 1), np.sort(np.nonzero(rm)[1].reshape(S, K), 1))
            w = np.take_along_axis(p, got, 1)
            assert np.all(w[:, 0] >= w[:, 1])
            np.testing.assert_array_equal(out["gpu_weights"][r, :, slot], w)
        np.testing.assert_array_equal(out["tokens"][r], t["input.tokens"].reshape(-1))


def test_build_indexer_topk(tmp_path):
    rows = [_row(tmp_path / f"step_1_row_{i:02d}.safetensors", i, np.arange(S) + 100 * i) for i in range(2)]
    out, info = bfr.build(sorted(tmp_path.glob("*.safetensors")), num_decoder_layers=3, indexer=True)
    fit = out["forced_indexer_topk"]
    assert fit.shape == (2, S, 4, 3) and fit.dtype == np.int16 and info["indexer_topk"] == 3
    for r, t in enumerate(rows):
        for slot, name in ((0, "layers.0"), (1, "layers.1"), (2, "layers.2"), (3, "mtp_layers.0")):
            np.testing.assert_array_equal(fit[r, :, slot], t[f"hf.model.{name}.self_attn.topk"][0])
    assert "forced_indexer_topk" not in bfr.build(sorted(tmp_path.glob("*.safetensors")), num_decoder_layers=3)[0]


def test_build_rejects_map_not_topk(tmp_path):
    _row(tmp_path / "step_1_row_00.safetensors", 0, np.arange(S), break_map=True)
    with pytest.raises(SystemExit, match="非零位置不一致"):
        bfr.build(sorted(tmp_path.glob("*.safetensors")), num_decoder_layers=3)


def test_main_writes_metadata(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    _row(d / "step_1_row_00.safetensors", 0, np.arange(S))
    out = tmp_path / "fr.safetensors"
    bfr.main(["--choices", str(d), "--out", str(out), "--num-decoder-layers", "3"])
    header, _ = read_safetensors_header(out)
    info = json.loads(header["__metadata__"]["info"])
    assert info["slots"] == 4 and info["sources"][0]["file"] == "step_1_row_00.safetensors"
    assert read_safetensors_tensor(out, "forced_routed_experts")[0].shape == (1, S, 4, K)
