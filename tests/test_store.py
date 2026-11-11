import json

import ml_dtypes
import numpy as np
import pytest

from janus.store import Run, RunWriter, StoreError, read_safetensors_tensor, write_safetensors

META = {
    "schema_version": 1, "run_name": "t-golden-1", "side": "golden-1", "framework": "megatron",
    "model": "toy", "code": {"repo": "x", "commit": "0"}, "image": "img@sha256:0", "versions": {},
    "hardware": {"accelerator": "cpu", "count": 1, "topology": "-"},
    "parallelism": {"dp": 1, "fsdp": 1, "tp": 1, "ep": 1, "cp": 1, "pp": 1},
    "config": {"path": "c.yaml", "sha256": "0"}, "checkpoint": {"path": "-", "manifest_sha256": "0"},
    "data": {"manifest": "-", "manifest_sha256": "0", "seed": 0},
    "steps": {"first": 0, "last": 2, "convention": "golden_step = step + 1"},
    "determinism": {}, "dtypes": {"compute": "bf16"}, "created_at": "2026-11-03T23:00:00+08:00",
}


def make_run(tmp_path, done=True):
    w = RunWriter(tmp_path / "r", META)
    for s, v in enumerate([2.5, 2.25, float("nan")]):
        w.log_metrics(s, lm_loss=v, lr=1e-4 * s)
    bf = np.arange(6, dtype=np.float32).reshape(2, 3).astype(ml_dtypes.bfloat16)
    w.write_tensors(0, "layers.0", {"layers.0.attn.out": bf,
                                     "layers.0.mlp.router.topk_idx": np.array([[1, 2]], np.int32)},
                    kind="activation", point="forward")
    if done:
        w.done()
    return tmp_path / "r", bf


def test_roundtrip_keeps_bf16_and_metrics(tmp_path):
    p, bf = make_run(tmp_path)
    r = Run.open(p)
    t = r.tensor(0, "layers.0.attn.out")
    assert t.dtype == np.dtype(ml_dtypes.bfloat16)
    assert np.array_equal(t.view(np.uint16), bf.view(np.uint16))
    s = r.series("lm_loss")
    assert s[1] == 2.25 and np.isnan(s[2])
    assert r.names(0, "layers.0.mlp.*") == ["layers.0.mlp.router.topk_idx"]
    assert r.to_golden_step(3) == 4


def test_missing_done_is_error(tmp_path):
    p, _ = make_run(tmp_path, done=False)
    with pytest.raises(StoreError):
        Run.open(p)


def test_sha_mismatch_is_error(tmp_path):
    p, _ = make_run(tmp_path)
    f = p / "step_0" / "layers.0.safetensors"
    raw = bytearray(f.read_bytes())
    n = int.from_bytes(raw[:8], "little")
    raw[8 + n] ^= 0xFF                   # 改坏第一个张量（按名字排序为 attn.out）的第一个字节
    f.write_bytes(bytes(raw))
    r = Run.open(p)
    with pytest.raises(StoreError):
        r.tensor(0, "layers.0.attn.out")


def test_meta_missing_field_rejected(tmp_path):
    bad = dict(META)
    bad.pop("image")
    with pytest.raises(StoreError):
        RunWriter(tmp_path / "x", bad)


def test_no_overwrite_after_done(tmp_path):
    p, _ = make_run(tmp_path)
    with pytest.raises(StoreError):
        RunWriter(p, META)


def test_safetensors_header_alignment(tmp_path):
    info = write_safetensors(tmp_path / "a.safetensors", {"x": np.ones(3, np.float16)})
    arr, raw = read_safetensors_tensor(tmp_path / "a.safetensors", "x")
    assert arr.dtype == np.float16 and len(raw) == 6 and info["x"]["dtype"] == "F16"
    n = int.from_bytes((tmp_path / "a.safetensors").read_bytes()[:8], "little")
    assert n % 8 == 0


def test_duplicate_name_in_index_is_error(tmp_path):
    p, _ = make_run(tmp_path)
    f = p / "step_0" / "index.json"
    idx = json.loads(f.read_text())
    idx["tensors"].append(dict(idx["tensors"][0]))           # 同名张量出现两次
    f.write_text(json.dumps(idx))
    r = Run.open(p)
    with pytest.raises(StoreError, match="重名"):
        r.tensor(0, "layers.0.attn.out")
    with pytest.raises(StoreError, match="重名"):
        r.names(0)


def test_writer_rejects_duplicate_name_across_scopes(tmp_path):
    w = RunWriter(tmp_path / "w", META)
    w.write_tensors(0, "a", {"x": np.zeros(2, np.float32)}, kind="other", point="forward")
    with pytest.raises(StoreError, match="重名"):
        w.write_tensors(0, "b", {"x": np.ones(2, np.float32)}, kind="other", point="forward")
    w.write_tensors(1, "b", {"x": np.ones(2, np.float32)}, kind="other", point="forward")   # 别的步可以同名
