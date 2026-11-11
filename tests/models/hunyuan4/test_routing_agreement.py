"""routing_agreement.py：路由一致率（TPU 自选对 GPU 两两的翻转率）的计算与判定；故意改坏的 TPU 选择要被判不过。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("safetensors")
from safetensors.numpy import save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("routing_agreement", ROOT / "models/hunyuan4/analysis/routing_agreement.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

S, E, K, ROWS = 32, 8, 2, 2


def _sel(rng, flip_tokens=()):
    raw = rng.uniform(0, 1, (S, E))
    idx = np.argsort(-raw, axis=1)[:, :K]
    for t in flip_tokens:  # 换掉第 t 个 token 的第 2 名
        rest = [e for e in range(E) if e not in idx[t]]
        idx[t, 1] = rest[0]
    return idx


def _gpu_dir(path: Path, tokens, idxs, topks):
    path.mkdir()
    for r in range(ROWS):
        rm = np.zeros((S, E), bool)
        np.put_along_axis(rm, idxs[r], True, axis=1)
        save_file(
            {
                "input.tokens": tokens[r].reshape(1, S).astype(np.int64),
                "hf.model.layers.1.mlp.gate.routing_map": rm,
                "hf.model.layers.0.self_attn.topk": topks[r].reshape(1, S, -1).astype(np.int64),
            },
            str(path / f"step_1_row_{r:02d}.safetensors"),
        )


def _fp64(tmp_path, tokens, base, topk, flips=()):
    """FP64 参照：选择同 GPU a（再翻 flips 个 token），分数让选中的专家排在前 K；每行都存 DSA top-k。"""
    t = {}
    for r in range(ROWS):
        idx = _sel(np.random.default_rng(10 + r), flips)
        sc = np.zeros((S, E))
        np.put_along_axis(sc, idx, 1.0, axis=1)
        t[f"r{r}.route.1.idx"] = idx.astype(np.int16)
        t[f"r{r}.route.1.scores"] = sc
        t[f"r{r}.dsa.0.topk"] = topk[r].reshape(1, S, -1).astype(np.int16)
    (tmp_path / "fp64").mkdir()
    save_file(t, str(tmp_path / "fp64" / "choices.safetensors"))
    np.save(tmp_path / "tokens.npy", tokens.astype(np.int64))


def _run(tmp_path, tpu_flips, gpu_flips_b=(3,), fp64_flips=None):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1000, (ROWS, S))
    base = [_sel(np.random.default_rng(10 + r)) for r in range(ROWS)]
    topk = [np.where(np.arange(S)[None, :4] <= np.arange(S)[:, None], np.arange(S)[None, :4], -1) for _ in range(ROWS)]
    _gpu_dir(tmp_path / "a", tokens, base, topk)
    _gpu_dir(tmp_path / "b", tokens, [_sel(np.random.default_rng(10 + r), gpu_flips_b) for r in range(ROWS)], topk)
    # TPU：行序倒过来（按 token 对上），翻 tpu_flips 个 token
    t_idx = np.stack([_sel(np.random.default_rng(10 + r), tpu_flips) for r in range(ROWS)])[::-1]
    fre = np.full((ROWS, S, 1, K), -1, np.int32)
    fre = np.concatenate([fre, t_idx[:, :, None, :].astype(np.int32)], axis=2)  # 槽位 0 稠密，槽位 1 是 MoE
    fit = np.stack(topk)[::-1][:, :, None, :].repeat(2, axis=2).astype(np.int16)
    save_file(
        {"forced_routed_experts": fre, "forced_indexer_topk": fit, "tokens": tokens[::-1].copy().astype(np.int32)},
        str(tmp_path / "tpu.safetensors"),
    )
    if fp64_flips is not None:
        _fp64(tmp_path, tokens, base, topk, fp64_flips)
    out = tmp_path / "out.json"
    argv = ["x", "--gpu", f"a={tmp_path / 'a'}", "--gpu", f"b={tmp_path / 'b'}", "--tpu", f"t={tmp_path / 'tpu.safetensors'}",
            "--num-decoder-layers", "2", "--num-experts", str(E), "--topk", str(K), "--moe-layers", "1",
            "--dsa-layers", "0", "--out", str(out), "--gbs", str(ROWS)]  # fmt: skip
    if fp64_flips is not None:
        argv += ["--fp64", str(tmp_path / "fp64"), "--tokens", str(tmp_path / "tokens.npy")]
    old = sys.argv
    sys.argv = argv
    try:
        ra.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text())


def test_within_gpu_spread_passes(tmp_path):
    d = _run(tmp_path, tpu_flips=(3,))  # 和 GPU b 翻同一个 token：对 a 1 / S、对 b 0
    r = d["routing"]["1"]
    assert r["flip_rate"]["a~b"] == pytest.approx(1 / S)  # 每行翻 1 个 token
    assert r["tpu"]["t"]["status"] == "pass"
    assert d["dsa"]["0"]["tpu"]["t"]["status"] == "pass"  # 全相同，尺子 0 且 TPU 0
    assert d["overall_gpu_pairs"]["t"] == "pass"
    assert d["overall"]["t"]["status"] is None  # 没给 FP64：正式口径不判


def test_twice_ruler_is_inconclusive(tmp_path):
    d = _run(tmp_path, tpu_flips=(5,))  # 对 b 差 2 个 token = 2 倍尺子
    assert d["routing"]["1"]["tpu"]["t"]["status"] == "inconclusive"


def test_broken_tpu_is_caught(tmp_path):
    d = _run(tmp_path, tpu_flips=tuple(range(0, 12)))  # 翻 12 个 token，远超 GPU 的 1 个
    assert d["routing"]["1"]["tpu"]["t"]["status"] == "fail"
    assert d["overall_gpu_pairs"]["t"] == "fail"


def test_zero_ruler_any_flip_fails(tmp_path):
    d = _run(tmp_path, tpu_flips=(5,), gpu_flips_b=())
    assert d["routing"]["1"]["ruler_gpu_max"] == 0
    assert d["routing"]["1"]["tpu"]["t"]["status"] == "fail"


def test_judge_bands():
    assert ra.judge([0.01], 0.01) == ("pass", 1.0)
    assert ra.judge([0.015], 0.01)[0] == "inconclusive"
    assert ra.judge([0.03], 0.01)[0] == "fail"
    assert ra.judge([0.0], 0.0) == ("pass", None)


def test_judge_truth_bands():
    assert ra.judge_truth(0.012, [0.01, 0.008]) == ("pass", pytest.approx(1.2))
    assert ra.judge_truth(0.015, [0.01, 0.008])[0] == "inconclusive"
    assert ra.judge_truth(0.021, [0.01, 0.008])[0] == "fail"
    assert ra.judge_truth(0.0, [0.0]) == ("pass", None)
    assert ra.judge_truth(0.001, [0.0]) == ("fail", None)


def test_truth_view_judges(tmp_path):
    # FP64 = GPU a 再翻 token 7：GPU a 对 FP64 1 / S、GPU b 2 / S；TPU 翻 token 3（同 b）→ 2 / S，R = 1 过
    d = _run(tmp_path, tpu_flips=(3,), fp64_flips=(7,))
    tr = d["routing"]["1"]["tpu"]["t"]["truth"]
    assert tr["status"] == "pass" and tr["R"] == pytest.approx(1.0)
    assert d["dsa"]["0"]["tpu"]["t"]["truth"]["status"] == "pass"
    assert d["overall"]["t"]["status"] == "pass"


def test_truth_view_catches_broken_tpu(tmp_path):
    d = _run(tmp_path, tpu_flips=tuple(range(8, 20)), fp64_flips=(7,))
    assert d["routing"]["1"]["tpu"]["t"]["truth"]["status"] == "fail"
    assert d["overall"]["t"]["status"] == "fail"


def test_bad_queries_catches_zero_filled_block():
    rng = np.random.default_rng(0)
    seq, k = 64, 8
    t = np.stack([np.sort(rng.choice(q + 1, size=min(k, q + 1), replace=False)).tolist() + [-1] * (k - min(k, q + 1))
                  for q in range(seq)]).astype(np.int16)  # fmt: skip
    assert len(ra.bad_queries(t)) == 0
    t[50:60] = 0  # 取回时被零填充的一段
    assert ra.bad_queries(t).tolist() == list(range(50, 60))
