import numpy as np
import pytest
from helpers import HLO_SAMPLE

from janus.checks import REGISTRY, Threshold
from janus.checks.base import ERROR, FAIL, PASS
from janus.checks.extra import hlo_fingerprint


def test_all_rfc_types_registered():
    rfc = ["bitwise_equal", "tensor_tolerance", "ulp_distribution", "discrete_agreement", "loss_series", "multi_golden",
           "scalar_series", "optimizer_step", "router_stats", "noise_verdict", "determinism", "coverage",
           "poison_selftest", "config_diff", "data_fingerprint", "checkpoint_equal", "replica_consistency",
           "overlap_replay", "artifact_check", "health_signals", "dtype_manifest", "fake_fp32", "op_replay",
           "run_complete"]
    assert sorted(rfc) == sorted(REGISTRY)


def test_optimizer_step_requires_positive_lr():
    p = [("w", np.ones(4, np.float32), np.ones(4, np.float32))]
    th = Threshold("numeric", {"rel_l2": 1e-6, "max_abs": 1e-6}, source="推断：示例")
    assert REGISTRY["optimizer_step"]("o", "L4", p, th, lr=0.0).status == ERROR
    assert REGISTRY["optimizer_step"]("o", "L4", p, th, lr=1e-5).status == PASS


def test_router_stats():
    ref = {"router.bias_std.1": {s: 0.01 + 1e-4 * s for s in range(20)}}
    test = {"router.bias_std.1": {s: v * 1.5 for s, v in ref["router.bias_std.1"].items()}}
    th = Threshold("numeric", {"signed_mean": 1e-3}, source="推断：示例")
    assert REGISTRY["router_stats"]("r", "L4", test, ref, th, params={"window": "0:20"}).status == FAIL
    assert REGISTRY["router_stats"]("r", "L4", ref, ref, th, params={"window": "0:20"}).status == PASS
    assert REGISTRY["router_stats"]("r", "L4", {}, ref, th, params={"window": "0:20"}).status == ERROR


def test_config_diff():
    gpu = {"model": {"hidden_dropout": 0.1, "eps": 1e-5}, "opt": {"beta2": 0.95}}
    tpu = {"dropout_rate": 0.0, "normalization_layer_epsilon": 1e-5, "adam_b2": 0.95}
    mapping = [{"test": "dropout_rate", "ref": "model.hidden_dropout", "must_equal": True},
               {"test": "normalization_layer_epsilon", "ref": "model.eps", "must_equal": True, "transform": "float"},
               {"test": "adam_b2", "ref": "opt.beta2", "must_equal": True}]
    r = REGISTRY["config_diff"]("c", "L0", tpu, gpu, mapping)
    assert r.status == FAIL and r.metrics["must_equal_differ"] == 1
    mapping.append({"test": "nope", "ref": "opt.beta2", "must_equal": True})
    assert REGISTRY["config_diff"]("c", "L0", tpu, gpu, mapping).status == ERROR


def test_data_fingerprint():
    t = {"1.0": np.arange(8), "2.0": np.arange(8) + 1}
    assert REGISTRY["data_fingerprint"]("d", "L0", t, dict(t), first_loss=(2.5, 2.5)).status == PASS
    shifted = {"1.0": np.roll(t["1.0"], 1), "2.0": t["2.0"]}
    assert REGISTRY["data_fingerprint"]("d", "L0", shifted, t).status == FAIL
    assert REGISTRY["data_fingerprint"]("d", "L0", t, t, first_loss=(2.5, 2.5000001)).status == FAIL


def test_checkpoint_equal(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "w.bin").write_bytes(b"\x00\x01")
    assert REGISTRY["checkpoint_equal"]("k", "L0", a, b).status == PASS
    (b / "w.bin").write_bytes(b"\x00\x02")
    assert REGISTRY["checkpoint_equal"]("k", "L0", a, b).status == FAIL
    (b / "extra.bin").write_bytes(b"")
    assert REGISTRY["checkpoint_equal"]("k", "L0", a, b).metrics["only_one_side"] == 1


def test_replica_and_overlap_and_artifact():
    assert REGISTRY["replica_consistency"]("r", "L2", {"r0": np.ones(3), "r1": np.ones(3)}).status == PASS
    assert REGISTRY["replica_consistency"]("r", "L2", {"r0": np.ones(3), "r1": np.zeros(3)}).status == FAIL
    before = {10: {"loss": 2.0, "num_zeros": 0.0}, 11: {"loss": 1.9, "num_zeros": 0.0}}
    after = {10: {"loss": 2.0, "num_zeros": 0.0}, 11: {"loss": 1.9, "num_zeros": 384.0}}
    r = REGISTRY["overlap_replay"]("o", "L4", before, after)
    assert r.status == FAIL and r.metrics["first_differ_step"] == 11
    hlo = "dot(f8e4m3 a, f8e4m3 b)\ndot(bf16 c, bf16 d)\n"
    rules = [{"name": "fp8_dots", "regex": r"dot\(f8", "min": 1}]
    assert REGISTRY["artifact_check"]("a", "L1", hlo, rules).status == PASS
    assert REGISTRY["artifact_check"]("a", "L1", "dot(bf16 c)", rules).status == FAIL


def test_health_signals():
    ok = {"raw_grad_norm": {s: 1.0 + 0.01 * (s % 3) for s in range(50)}}
    assert REGISTRY["health_signals"]("h", "L4", ok).status == PASS
    spike = {"raw_grad_norm": {**ok["raw_grad_norm"], 40: 50.0}}
    assert REGISTRY["health_signals"]("h", "L4", spike).status == FAIL
    nan = {"raw_grad_norm": {**ok["raw_grad_norm"], 30: float("nan")}}
    assert REGISTRY["health_signals"]("h", "L4", nan).status == FAIL


def test_determinism():
    a = {"lm_loss": {1: 2.0, 2: 1.9}}
    assert REGISTRY["determinism"]("d", "L4", a, {"lm_loss": {1: 2.0, 2: 1.9}}).status == PASS
    assert REGISTRY["determinism"]("d", "L4", a, {"lm_loss": {1: 2.0, 2: 1.9000001}}).status == FAIL


# ---------------------------------------------------------------- config_diff 等于给定值

def test_config_diff_equals_value():
    t = {"model_config": {"hidden_dropout": 0.0, "attention_dropout": 0.1}}
    r_ = {"model_config": {"hidden_dropout": 0.0, "attention_dropout": 0.1}}
    m = [{"test": "model_config.hidden_dropout", "equals": 0.0},
         {"test": "model_config.attention_dropout", "equals": 0.0}]
    r = REGISTRY["config_diff"]("c", "L0", t, r_, m)
    assert r.status == FAIL and "attention_dropout" in r.message      # 两边相同但不是 0，照样红
    assert r.metrics["equals_rules"] == 2
    ok = REGISTRY["config_diff"]("c", "L0", t, r_, m[:1])
    assert ok.status == PASS
    r_bad_ref = {"model_config": {"hidden_dropout": 0.1}}
    assert REGISTRY["config_diff"]("c", "L0", t, r_bad_ref, m[:1]).status == FAIL   # 参照一边不是 0 也红
    assert REGISTRY["config_diff"]("c", "L0", t, {}, m[:1]).status == ERROR          # 缺键 error


# ---------------------------------------------------------------- data_fingerprint 缺字段

def test_data_fingerprint_missing_field_is_error_and_counts_units():
    t = {"1.r0.m0.tokens": "a", "1.r0.m0.labels": "b"}
    r = REGISTRY["data_fingerprint"]("d", "L0", t, dict(t), missing=["x:1.r0.m0.loss_mask"], fields=["tokens", "labels", "loss_mask"])
    assert r.status == ERROR and "缺失" in r.message
    r = REGISTRY["data_fingerprint"]("d", "L0", t, dict(t), fields=["tokens", "labels"])
    assert r.status == PASS and r.metrics["units_per_field"] == {"tokens": 1, "labels": 1}


# ---------------------------------------------------------------- dtype 清单

DTYPE_RULES = [{"pattern": "buffer.*expert_bias", "dtype": "float32"},
               {"pattern": "*", "exclude": "buffer.*", "dtype": "bfloat16"}]


def test_dtype_manifest_rules():
    good = {"final/rank0": {"buffer.decoder.layers.1.mlp.router.expert_bias": {"dtype": "float32"},
                            "decoder.layers.0.self_attention.linear_qkv.weight": {"dtype": "bfloat16"}}}
    r = REGISTRY["dtype_manifest"]("t", "L0", good, DTYPE_RULES)
    assert r.status == PASS and r.metrics["entries"] == 2
    bad = {"final/rank0": {"buffer.decoder.layers.1.mlp.router.expert_bias": {"dtype": "torch.bfloat16"},
                           "w": {"dtype": "BF16"}}}
    r = REGISTRY["dtype_manifest"]("t", "L0", bad, DTYPE_RULES)
    assert r.status == FAIL and r.metrics["wrong_dtype"] == 1 and "expert_bias" in r.message
    other_buf = {"final/rank0": {"buffer.something_else": {"dtype": "float32"}}}
    assert REGISTRY["dtype_manifest"]("t", "L0", other_buf, DTYPE_RULES).status == ERROR   # 没有规则覆盖
    assert REGISTRY["dtype_manifest"]("t", "L0", {}, DTYPE_RULES).status == ERROR


# ---------------------------------------------------------------- 假 FP32

def test_fake_fp32_catches_bf16_rounded_tensor():
    import ml_dtypes
    rng = np.random.default_rng(0)
    real = rng.standard_normal((256, 64)).astype(np.float32)
    fake = real.astype(ml_dtypes.bfloat16).astype(np.float32)      # BF16 舍入后存成 FP32
    r = REGISTRY["fake_fp32"]("f", "L3", [("main_param.w", fake)])
    assert r.status == FAIL and r.metrics["low16_zero_frac_max"] == 1.0
    r = REGISTRY["fake_fp32"]("f", "L3", [("main_param.w", real)])
    assert r.status == PASS and r.metrics["low16_zero_frac_max"] < 1e-3
    # 大量真 0（没收到 token 的嵌入行梯度 / 动量）不算假 FP32
    sparse = real.copy()
    sparse[:250] = 0.0
    assert REGISTRY["fake_fp32"]("f", "L3", [("exp_avg.emb", sparse)]).status == PASS
    # 只看 float32；全是别的 dtype → error；改门槛必须写 source
    assert REGISTRY["fake_fp32"]("f", "L3", [("p", fake.astype(ml_dtypes.bfloat16))]).status == ERROR
    assert REGISTRY["fake_fp32"]("f", "L3", [("w", real)], params={"max_frac": 0.5}).status == ERROR


# ---------------------------------------------------------------- tensor_tolerance 新指标、ULP 分桶

def test_tensor_tolerance_new_metrics_and_aggregation():
    rng = np.random.default_rng(0)
    b = rng.standard_normal((32, 16)).astype(np.float32)
    pairs = [("g0", b * np.float32(1.02), b), ("g1", b * np.float32(0.97), b)]
    th = Threshold("numeric", {"std_ratio": 0.05, "zero_frac_diff": 0.0, "rel_err_p99": 0.05, "kl": 1e-2},
                   source="推断：示例")
    r = REGISTRY["tensor_tolerance"]("t", "L3", pairs, ["std_ratio", "zero_frac_diff", "rel_err_p99", "kl"], th)
    assert r.metrics["std_ratio_dev_max"] == pytest.approx(0.03, rel=1e-3)      # |0.97 − 1|，偏小也算
    assert r.metrics["zero_frac_diff_max"] == 0.0
    assert r.metrics["rel_err_p99_max"] == pytest.approx(0.03, rel=1e-2)
    assert r.metrics["kl_max"] > 0 and r.status == PASS
    z = b.copy()
    z[:4] = 0.0
    th0 = Threshold("numeric", {"zero_frac_diff": 0.0}, source="推断：示例")
    r = REGISTRY["tensor_tolerance"]("t", "L3", [("g", z, b)], ["zero_frac_diff"], th0)
    assert r.status == FAIL and r.metrics["zero_frac_diff_max"] == pytest.approx(4 * 16 / b.size)


def test_ulp_distribution_reports_buckets():
    b = np.ones(4, np.float32)
    a = (b.view(np.uint32) + np.array([0, 1, 1, 30], np.uint32)).view(np.float32)
    r = REGISTRY["ulp_distribution"]("u", "L3", [("x", a, b), ("y", b, b)])
    assert r.status == PASS
    assert r.details[0]["one_frac"] == 0.5 and r.details[0]["buckets"]["17+"] == 0.25
    assert r.metrics["one_frac"] == pytest.approx(2 / 8) and r.metrics["buckets"]["0"] == pytest.approx(5 / 8)


def test_health_signals_without_data_is_error_not_pass():
    """字段拼错或这次运行没记：以前判「健康」，现在判 error。"""
    fn = REGISTRY["health_signals"]
    assert fn("H", "L4", {"lm_loss": {1: 3.0, 2: 2.9}}).status == PASS
    r = fn("H", "L4", {"lm_loss": {1: 3.0}, "lm_loss_typo": {}})
    assert r.status == ERROR and "lm_loss_typo" in r.message
    assert fn("H", "L4", {}).status == ERROR


def test_determinism_with_empty_test_field_is_error_not_pass():
    fn = REGISTRY["determinism"]
    ref = {"lm_loss": {1: 3.0, 2: 2.9}}
    assert fn("D", "L0", {"lm_loss": {1: 3.0, 2: 2.9}}, ref).status == PASS
    assert fn("D", "L0", {"lm_loss": {1: 3.0}}, ref).status == PASS             # 只比被测有的步：短运行对长基线，设计如此
    r = fn("D", "L0", {"lm_loss": {}}, ref)                                     # 被测一步都没有：以前什么都不比就判 pass
    assert r.status == ERROR and "lm_loss" in r.message


def test_run_complete():
    fn = REGISTRY["run_complete"]
    meta = {"steps": {"first": 1, "last": 5}}
    assert fn("r", "regress", [1, 2, 3, 4, 5], meta, params={"steps": 5}).status == PASS
    r = fn("r", "regress", [1, 2, 3, 2, 3, 4, 5], meta, params={"steps": 5})
    assert r.status == FAIL and r.metrics["duplicate"] == 2 and "重复步" in r.message
    assert fn("r", "regress", [1, 2, 4, 5], meta, params={"steps": 5}).status == FAIL
    assert fn("r", "regress", [1, 2, 3], {"steps": {"first": 1, "last": 3}}, params={"steps": 5}).status == FAIL
    assert fn("r", "regress", [1, 2, 3, 4, 5], {**meta, "restarts": 1}, params={"steps": 5}).status == FAIL
    assert fn("r", "regress", [1], meta, params={}).status == ERROR


def test_artifact_check_match_ref_and_fingerprint():
    fn = REGISTRY["artifact_check"]
    rules = [{"name": "custom", "regex": 'custom_call_target="tpu_custom_call"', "min": 1, "match_ref": True},
             {"name": "fp8", "regex": r"\bf8e[45]m[23]\w*\[", "max": 0}]
    # 只改了元数据（源码行号、op 名、stack_frame_id）：计数相同、指纹相同
    moved = HLO_SAMPLE.replace("source_line=3", "source_line=9").replace("stack_frame_id=7", "stack_frame_id=8")
    assert hlo_fingerprint(HLO_SAMPLE) == hlo_fingerprint(moved)
    r = fn("a", "regress", moved, rules, ref_text=HLO_SAMPLE, fingerprint=True)
    assert r.status == PASS and r.metrics["fingerprint_equal"]
    # 少了一个 custom call：计数对不上 LKG、指纹也变
    fewer = HLO_SAMPLE.replace('%splash_mha_fwd.1 = bf16[2]{0} custom-call(%b), custom_call_target="tpu_custom_call"\n', "")
    r = fn("a", "regress", fewer, rules, ref_text=HLO_SAMPLE, fingerprint=True)
    assert r.status == FAIL and "custom=1（参照 2）" in r.message and r.metrics["lines_only_ref"] == 1
    assert fn("a", "regress", HLO_SAMPLE.replace("bf16[2]", "f8e4m3fn[2]"), rules, ref_text=HLO_SAMPLE).status == FAIL
    assert fn("a", "regress", HLO_SAMPLE, rules).status == ERROR             # match_ref 没给参照
    # ignore_lines：只差一行 GPU 选项时指纹相同，差别在别的行仍报
    da, db = 'xla_dump_to: "/data/xla"\nxla_gpu_x: READ\n', 'xla_dump_to: "/data/xla"\n'
    assert fn("d", "regress", db, [], ref_text=da, fingerprint=True).status == FAIL
    r = fn("d", "regress", db, [], ref_text=da, fingerprint=True, ignore_lines=["^xla_gpu_"])
    assert r.status == PASS and r.metrics["ignored_lines"] == {"test": 0, "ref": 1}
    assert fn("d", "regress", "xla_dump_to: \"/x\"\n", [], ref_text=da, fingerprint=True, ignore_lines=["^xla_gpu_"]).status == FAIL
    assert fn("a", "regress", HLO_SAMPLE, [{"name": "c", "regex": "custom-call", "min": 1}]).status == PASS


def test_config_diff_dict_with_ignore():
    fn = REGISTRY["config_diff"]
    a = {"cfg": {"run_name": "a", "out_dir": "/a", "lr": 1e-5, "remat": "custom"}}
    b = {"cfg": {"run_name": "b", "out_dir": "/b", "lr": 1e-5, "remat": "custom"}}
    m = [{"test": "cfg", "ref": "cfg", "must_equal": True, "ignore": ["run_name", "*dir*"]}]
    assert fn("c", "regress", a, b, m).status == PASS
    b["cfg"]["remat"] = "full"
    r = fn("c", "regress", a, b, m)
    assert r.status == FAIL and r.details[0]["differ"] == [{"key": "remat", "test": "custom", "ref": "full"}]
