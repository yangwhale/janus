"""janus/dump/maxtext.py 里不依赖 jax 的部分，与 models/hunyuan3/frameworks/maxtext/replay.py。"""
import hashlib
import json
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from helpers import (
    ROOT,  # noqa: E402  仓库根
    tpu_meta,
    tpu_state,
)

from janus.dump import maxtext as jd
from janus.dump.name_map import NameMap, apply_ops, leaf_index
from janus.store import Run, StoreError

sys.path.insert(0, str(ROOT / "models" / "hunyuan3" / "frameworks" / "maxtext"))
import replay  # noqa: E402

SMALL = dict(hidden=32, ffn=48, moe_ffn=8, experts=16, topk=4, heads=8, kv_heads=2, head_dim=4, vocab=64)


def small():
    nm = NameMap.load(ROOT / "models" / "hunyuan3" / "model" / "name_map.yaml")
    return NameMap({**nm.spec, "dims": {**nm.dims, **SMALL},
                    "megatron": {**nm.spec["megatron"], "local_experts": SMALL["experts"] // nm.ep}})


def fake_state(nm, seed=0):
    """按 name_map 把随机 HF 权重摆成训练状态的扁平字典（参数、偏置、Adam 动量）。"""
    rng = np.random.default_rng(seed)
    flat, hf = {}, {}
    leaves = {p: np.zeros(s, np.float32) for p, s in nm.leaf_shapes().items()}
    for ent in nm.entries:
        for h, shp in zip(ent.hf, nm.hf_shapes(ent)):
            hf[h] = rng.standard_normal(shp).astype(np.float32 if h.endswith("expert_bias") else ml_dtypes.bfloat16)
        for pc in ent.pieces:
            leaves[pc.path][leaf_index(pc.index, leaves[pc.path].ndim)] = apply_ops(hf[pc.hf].astype(np.float32), pc.ops)
    for p, v in leaves.items():
        flat[jd.state_key(p)] = v
        if p.startswith("params/"):
            flat[jd.state_key(p, "mu")] = v * 0.5
            flat[jd.state_key(p, "nu")] = v * v
    return flat, hf


def test_state_key():
    assert jd.state_key("params/decoder/x/kernel") == "params/model/decoder/x/kernel"
    assert jd.state_key("MoEBiasVar/decoder/g/bias") == "MoEBiasVar/model/decoder/g/bias"
    assert jd.state_key("params/decoder/x/kernel", "nu") == "OptVariable/optimizer/opt_state/0/nu/decoder/x/kernel"


def test_param_sha_matches_megatron_layout(tmp_path):
    """TPU 写的参数清单 = HF 权重按 Megatron 布局（EP8 持有关系）算的 sha256：和 GPU 的 params_sha 同名同值。"""
    nm = small()
    flat, hf = fake_state(nm)
    meta = {k: {} for k in ("code", "versions", "hardware", "parallelism", "config", "checkpoint", "data", "steps",
                            "determinism", "dtypes")}
    meta.update(schema_version=1, run_name="t", side="tpu", framework="maxtext", model="toy", image="x", created_at="-")
    d = jd.MaxTextDump(tmp_path / "t", meta, nm, dump_steps={1}, full=True)
    d.write_param_sha(flat, "init")
    for r in range(nm.ep):
        got = json.loads((tmp_path / "t" / "params_sha" / f"rank{r}.init.json").read_text())
        want = {}
        for ent in nm.entries:
            if nm.expert_owner(ent) not in (None, r):
                continue
            a = nm.to_megatron(ent, [hf[h] for h in ent.hf])
            want[("buffer." if ent.buffer else "") + nm.megatron_name(ent)] = hashlib.sha256(a.tobytes()).hexdigest()
        assert {k: v["sha256"] for k, v in got.items()} == want
    # 全量抓数：更新前后的主权重、动量、偏置，名字是 Megatron 名，专家只留 0、1
    d.capture_before(1, flat)
    d.capture_after(1, flat, None)
    d.log_metrics(1, {"lm_loss": 1.0})
    d.flush(1)
    d.on_train_end()
    run = Run.open(tmp_path / "t")
    names = run.names(1)
    assert "main_param.decoder.layers.1.mlp.experts.linear_fc1.weight1" in names
    assert "main_param.decoder.layers.1.mlp.experts.linear_fc1.weight2" not in names
    assert "exp_avg_sq.decoder.layers.0.self_attention.linear_qkv.weight" in names
    assert "buffer_after.decoder.layers.3.mlp.router.expert_bias" in names
    assert not any(n.startswith("exp_avg.") and "expert_bias" in n for n in names)
    qkv = run.tensor(1, "param.decoder.layers.0.self_attention.linear_qkv.weight")
    ent = nm.by_canonical("layers.0.attn.qkv.weight")
    assert qkv.tobytes() == nm.to_megatron(ent, [hf[h] for h in ent.hf]).tobytes()


def test_route_from_scores_ties_and_weights():
    scores = np.array([[[0.5, 0.2, 0.2, 0.6]]], np.float32)
    bias = np.array([0.0, 0.1, 0.1, -0.5], np.float32)
    idx, probs = jd.route_from_scores(scores + bias, scores, k=2, scale=2.0)
    assert idx.tolist() == [[[0, 1]]]            # 加偏置后 0.5、0.3、0.3、0.1：第 2 名并列时取小下标；权重用不加偏置的分数
    w = np.array([0.5, 0.2]) / 0.7 * 2.0
    assert np.allclose(probs[0, 0], [w[0], w[1], 0, 0])


def test_assemble_activations_layout():
    dims = dict(SMALL, layers=4, first_dense=1, topk=4, routed_scaling=1.0)
    B, S, H, G, hd = 2, 5, 8, 2, 4
    rng = np.random.default_rng(0)
    q = rng.standard_normal((1, B, S, H, hd)).astype(np.float32)          # 稠密栈 1 层
    k = rng.standard_normal((1, B, S, G, hd)).astype(np.float32)
    v = rng.standard_normal((1, B, S, G, hd)).astype(np.float32)
    lin = rng.standard_normal((3, B, S, 32)).astype(np.float32)           # MoE 栈 3 层
    act = {"decoder/dense_layers/self_attention/query/janus__out": q, "decoder/dense_layers/self_attention/key/janus__out": k,
           "decoder/dense_layers/self_attention/value/janus__out": v, "decoder/moe_layers/janus__in": lin,
           "token_embedder/janus__out": lin[0]}
    out = jd.assemble_activations(act, dims, rows=B)
    assert out["embed.out"].shape == (S, 1, 32) and np.array_equal(out["embed.out"][:, 0], lin[0, 0])
    assert np.array_equal(out["layers.3.in"][:, 0], lin[2, 0])          # MoE 栈第 j 层 = 第 1 + j 层
    qkv = out["layers.0.attn.qkv.out"]
    assert qkv.shape == (S, 1, (H + 2 * G) * hd)
    per = (H // G + 2) * hd
    assert np.array_equal(qkv[:, 0, per: per + (H // G) * hd], q[0, 0, :, H // G:, :].reshape(S, -1))
    assert np.array_equal(qkv[:, 0, per - hd: per], v[0, 0, :, 0, :])


def _fake_golden(d: Path, steps=2, ranks=8, micro=2, seq=6):
    (d / "data" / "batches").mkdir(parents=True)
    for r in range(ranks):
        off = 0
        with open(d / "data" / "batches" / f"rank{r}.bin", "wb") as fb, open(d / "data" / "batches" / f"rank{r}.jsonl", "w") as fi:
            for s in range(1, steps + 1):
                for m in range(micro):
                    f = {"tokens": np.full((1, seq), 1000 * s + 10 * r + m, np.int64), "labels": np.full((1, seq), r, np.int64),
                         "loss_mask": np.ones((1, seq), np.float32), "position_ids": np.arange(seq, dtype=np.int64)[None]}
                    rec = {"step": s, "micro": m, "fields": {}}
                    for k, a in f.items():
                        rec["fields"][k] = {"offset": off, "dtype": str(a.dtype), "shape": list(a.shape),
                                            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                        fb.write(a.tobytes())
                        off += a.nbytes
                    fi.write(json.dumps(rec) + "\n")


def test_replay_row_order_and_fingerprints(tmp_path):
    _fake_golden(tmp_path)
    gb = replay.GoldenBatches(tmp_path)
    assert (gb.world, gb.micro, gb.steps) == (8, 2, [1, 2])
    g = gb.global_batch(2)
    assert g["tokens"][:, 0].tolist() == [2000 + 10 * r + m for r in range(8) for m in range(2)]   # 第 r·M+m 行 = rank r 的 micro m
    mt = replay.to_maxtext(g)
    assert mt["inputs"].dtype == np.int32 and mt["targets_segmentation"].sum() == g["loss_mask"].sum()
    back = replay.from_maxtext(mt)
    assert all(back[k].tobytes() == g[k].tobytes() for k in replay.FIELDS)
    fp = replay.fingerprints(back, 2, 2)
    assert fp[3][1]["tokens"] == gb.index[(2, 3, 1)]["tokens"]["sha256"]                         # 设备 d ↔ rank d
    assert fp[3][1]["nonpad"] == 6


def test_param_sha_catches_one_flipped_bit(tmp_path):
    """故意改坏：主权重里翻一位，参数清单里恰好那一个参数（及其所在 rank）的 sha256 变了。"""
    nm = small()
    flat, _ = fake_state(nm)
    meta = {k: {} for k in ("code", "versions", "hardware", "parallelism", "config", "checkpoint", "data", "steps",
                            "determinism", "dtypes")}
    meta.update(schema_version=1, run_name="t", side="tpu", framework="maxtext", model="toy", image="x", created_at="-")
    good = jd.MaxTextDump(tmp_path / "a", meta, nm)
    good.write_param_sha(flat, "init")
    key = jd.state_key("params/decoder/moe_layers/Hunyuan3MoeBlock_0/MoeBlock_0/wo")
    bad_leaf = flat[key].copy()
    bad_leaf.view(np.uint32)[5, 1, 0, 0] ^= 1 << 16          # 专家 5、第 2 层（栈内 1）、BF16 能看见的那一位
    bad = jd.MaxTextDump(tmp_path / "b", {**meta, "run_name": "b"}, nm)
    bad.write_param_sha({**flat, key: bad_leaf}, "init")
    owner = 5 // nm.local_experts
    for r in range(nm.ep):
        a = json.loads((tmp_path / "a" / "params_sha" / f"rank{r}.init.json").read_text())
        b = json.loads((tmp_path / "b" / "params_sha" / f"rank{r}.init.json").read_text())
        differ = [n for n in a if a[n]["sha256"] != b[n]["sha256"]]
        want = [f"decoder.layers.2.mlp.experts.linear_fc2.weight{5 - owner * nm.local_experts}"] if r == owner else []
        assert differ == want, (r, differ)


def test_state_manifest_sees_change_below_half_bf16_ulp():
    """FP32 主权重上一个元素改 1 个 FP32 ulp（远小于半个 BF16 格）：BF16 清单看不见，FP32 状态清单看得见。"""
    a = tpu_state()
    b = {k: np.array(v, copy=True) for k, v in a.items()}
    w = b["params/model/decoder/w"]
    w.reshape(-1)[5] = np.nextafter(w.reshape(-1)[5], np.float32(np.inf))
    assert jd.sha256(jd.to_bf16(a["params/model/decoder/w"])) == jd.sha256(jd.to_bf16(w))   # BF16 清单：看不见
    ma, mb = jd.state_manifest(a), jd.state_manifest(b)
    assert ma["params/model/decoder/w"]["sha256"] != mb["params/model/decoder/w"]["sha256"]
    assert {k: e for k, e in ma.items() if k != "params/model/decoder/w"} == \
           {k: e for k, e in mb.items() if k != "params/model/decoder/w"}
    kinds = {k: e["kind"] for k, e in ma.items()}
    assert kinds == {"params/model/decoder/w": "param", "MoEBiasVar/model/decoder/gate/bias": "bias",
                     "OptVariable/optimizer/opt_state/0/mu/decoder/w": "mu",
                     "OptVariable/optimizer/opt_state/0/nu/decoder/w": "nu",
                     "OptArray/optimizer/opt_state/0/count": "count", "OptState/optimizer/step": "count"}
    assert ma["params/model/decoder/w"]["dtype"] == "float32" and ma["OptState/optimizer/step"]["dtype"] == "uint32"
    assert jd.manifest_digest(ma, "nu") == jd.manifest_digest(mb, "nu")
    assert jd.manifest_digest(ma, "param") != jd.manifest_digest(mb, "param")


def test_sha256_matches_tobytes_for_all_dtypes():
    for a in (np.arange(7, dtype=np.float32), np.ones((2, 3), ml_dtypes.bfloat16), np.array(5, np.int32),
              np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))):
        assert jd.sha256(a) == hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_write_state_sha_and_read_back(tmp_path):
    from janus.dump.name_map import NameMap
    nm = NameMap.load(ROOT / "models" / "hunyuan3" / "model" / "name_map.yaml")
    d = jd.MaxTextDump(tmp_path / "r", tpu_meta("r", steps={"first": 1, "last": 1}), nm, dump_steps=())
    man = d.write_state_sha(tpu_state(), "init")
    d.log_metrics(1, {"lm_loss": 1.0})
    d.on_train_end()
    run = Run.open(tmp_path / "r")
    assert run.state_manifest("init") == {"state": man}
    assert run.manifest("state", "init") == {"state": man}
    with pytest.raises(StoreError):
        run.state_manifest("final")


def test_full_only_filter_and_probe_tensors(tmp_path):
    """--full-only（逗号分隔按「或」）只留选中的参数，buffer 照抓；独立梯度探针 gp_* 随本步写出、按层分文件、kind 是 grad。"""
    nm = small()
    flat, _ = fake_state(nm)
    d = jd.MaxTextDump(tmp_path / "t", tpu_meta("t", steps={"first": 1, "last": 1}), nm, dump_steps={1}, full=True,
                       only=r"linear_proj,final_layernorm")
    d.capture_before(1, flat)
    d.capture_after(1, flat, None)
    g = np.ones((4, 4), np.float32)
    d.add_tensors({"gp_f32.decoder.layers.2.self_attention.linear_proj.weight": g, "gp_f32.output_layer.colsum": g[0]})
    d.log_metrics(1, {"lm_loss": 1.0})
    d.flush(1)
    d.on_train_end()
    run = Run.open(tmp_path / "t")
    names = set(run.names(1))
    assert "main_param.decoder.layers.0.self_attention.linear_proj.weight" in names
    assert "exp_avg.decoder.final_layernorm.weight" in names
    assert "main_param.decoder.layers.0.self_attention.linear_qkv.weight" not in names
    assert not any("experts" in n for n in names)
    assert "buffer_after.decoder.layers.3.mlp.router.expert_bias" in names
    assert run.tensor(1, "gp_f32.decoder.layers.2.self_attention.linear_proj.weight").tobytes() == g.tobytes()
    ent = next(e for e in run.index(1) if e.name == "gp_f32.output_layer.colsum")
    assert ent.kind == "grad"


def test_param_io_export_import_roundtrip(tmp_path):
    """TPU 侧整套参数导出（Megatron 布局，按 GPU rank 分文件）再导入回 MaxText 叶子：逐位还原；sha256 核对能抓到改坏的文件。"""
    import param_io
    nm = small()
    flat, _ = fake_state(nm)
    d = jd.MaxTextDump(tmp_path / "t", tpu_meta("t", steps={"first": 1, "last": 1}), nm, dump_steps=())
    man = param_io.export(d, nm, flat, tmp_path / "exp", 7, {"framework": "maxtext"}, 0)
    assert sorted(man["ranks"]) == [str(r) for r in range(nm.ep)]
    assert all("experts" in k for k in man["ranks"]["3"])
    leaves, info = param_io.load_leaves(str(tmp_path / "exp"), nm, nm.leaf_shapes())
    assert info["step"] == 7
    for p, v in leaves.items():
        assert v.tobytes() == np.asarray(flat[jd.state_key(p)], np.float32).tobytes(), p
    assert param_io.verify(d, nm, flat, man)["mismatch"] == 0
    flat2 = dict(flat)
    k = jd.state_key("params/decoder/decoder_norm/scale")
    flat2[k] = flat[k] + 1e-3
    assert param_io.verify(d, nm, flat2, man)["mismatch"] == 1


def test_inject_args_keyword_inputs():
    """逐算子回放的注入：iHC / mHC 包着的分支用关键字传输入（MLP inputs=，注意力 inputs_q= / inputs_kv=），也要换掉。
    以前只换位置参数与 x=，关键字传的输入原样放过，回放的「单算子」其实吃的是本侧上游的输入（混元 4 稠密 MLP R 1.56 的来源）。"""
    pytest.importorskip("jax")

    class M:
        _janus_path = "decoder/dense_layers/mlp"

    new = np.full((1, 2, 3), 7.0, np.float32)
    prev = jd._STATE.inject
    jd._STATE.inject = {M._janus_path: new}
    try:
        old = np.zeros((1, 2, 3), ml_dtypes.bfloat16)
        args, kw = jd._inject_args(M(), "mlp", (), {"inputs": old, "deterministic": True})
        assert args == () and kw["deterministic"] is True
        assert np.asarray(kw["inputs"]).dtype == old.dtype and float(np.asarray(kw["inputs"], np.float32).max()) == 7.0
        args, kw = jd._inject_args(M(), "attn", (), {"inputs_q": old, "inputs_kv": old})
        assert float(np.asarray(kw["inputs_q"], np.float32).min()) == 7.0 and float(np.asarray(kw["inputs_kv"], np.float32).min()) == 7.0
        args, kw = jd._inject_args(M(), "mlp", (old,), {})            # 位置参数照旧
        assert float(np.asarray(args[0], np.float32).min()) == 7.0
        with pytest.raises(ValueError):
            jd._inject_args(M(), "mlp", (), {"deterministic": True})   # 换不了就报错，不再静默放过
    finally:
        jd._STATE.inject = prev


def test_inject_output_replaces_first_output_only():
    """逐算子回放的输出注入（inject_out）：只换打了标记、在表里的模块的输出；元组只换第一个，其余原样；dtype 与形状随原输出。"""
    pytest.importorskip("jax")

    class M:
        _janus_path = "decoder/dense_layers/self_attention"

    class Other:
        _janus_path = "decoder/dense_layers/mlp"

    new = np.full((1, 2, 3), 5.0, np.float32)
    prev = jd._STATE.inject_out
    jd._STATE.inject_out = {M._janus_path: new}
    try:
        old = np.zeros((1, 2, 3), ml_dtypes.bfloat16)
        out = jd._inject_output(M(), (old, "kv", None))
        assert np.asarray(out[0]).dtype == old.dtype and float(np.asarray(out[0], np.float32).min()) == 5.0
        assert out[1:] == ("kv", None)
        out = jd._inject_output(M(), old)
        assert float(np.asarray(out, np.float32).max()) == 5.0
        assert jd._inject_output(Other(), old) is old                   # 不在表里的模块原样
    finally:
        jd._STATE.inject_out = prev
