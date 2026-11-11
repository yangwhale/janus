"""测试共用的造数据函数：小运行目录、最小检查项目录。只用 numpy 与 janus 自己的 RunWriter，不读真实运行。

conftest.py 把 tests/ 加进 sys.path，测试里 `from helpers import …`。
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from janus.replay import adamw_step
from janus.store import RunWriter

ROOT = Path(__file__).resolve().parents[1]  # 仓库根


BASE_META = {
    "schema_version": 1, "framework": "megatron", "model": "toy", "code": {"repo": "x", "commit": "0"},
    "image": "img@sha256:0", "versions": {}, "hardware": {"accelerator": "cpu", "count": 1, "topology": "-"},
    "parallelism": {"dp": 1, "fsdp": 1, "tp": 1, "ep": 1, "cp": 1, "pp": 1},
    "config": {"path": "c", "sha256": "0"}, "checkpoint": {"path": "-", "manifest_sha256": "0"},
    "data": {"manifest": "-", "manifest_sha256": "0", "seed": 0},
    "steps": {"first": 1, "last": 50, "convention": "golden_step = step"},
    "determinism": {}, "dtypes": {}, "created_at": "2026-11-03T23:00:00+08:00",
}


def make(root, name, side, seed, noise=1e-4):
    rng = np.random.default_rng(seed)
    w = RunWriter(root / name, {**BASE_META, "run_name": name, "side": side})
    for s in range(1, 51):
        w.log_metrics(s, lm_loss=(3.0 - s / 50) * (1 + (noise * rng.standard_normal() if s > 1 else 0)), lr=1e-5)
    act = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    act = act * (1 + noise * rng.standard_normal(act.shape)).astype(np.float32)
    w.write_tensors(1, "layers.0", {"layers.0.out": act}, kind="activation", point="forward")
    w.done()


def catalog(root, extra_checks=None):
    return {
        "schema_version": 1, "model": "toy", "stage": "t",
        "runs": {"test": "g2", "golden": ["g1"], "root": str(root),
                 "noise": [["g1", "g2"], ["g1", "g3"], ["g2", "g3"]]},
        "checks": [
            {"id": "L4.loss", "type": "multi_golden", "level": "L4",
             "select": {"kind": "metrics", "field": "lm_loss", "window": "1:51"}, "threshold": "calibrated"},
            {"id": "L4.lr", "type": "scalar_series", "level": "L4",
             "select": {"kind": "metrics", "field": "lr", "window": "1:51"}},
            {"id": "L2.act", "type": "tensor_tolerance", "level": "L2", "metrics": ["rel_l2", "cosine"],
             "select": {"kind": "tensor", "step": 1, "name": "layers.*"}, "threshold": "calibrated"},
            {"id": "L2.cov", "type": "coverage", "level": "L2", "select": {"kind": "tensor", "step": 1, "name": "*"}},
        ] + (extra_checks or []),
    }



# ---- 带输入指纹、参数清单、优化器状态的完整小运行（test_run_full.py、test_audit_fixes.py 用）

P = "decoder.layers.0.self_attention.linear_proj.weight"


def make_full(root, name, *, wd=0.033, lr=1e-3, token_shift=False, bad_update=False, extra_meta=None):
    rng = np.random.default_rng(0)
    meta = {**BASE_META, "run_name": name, "side": f"golden-{name}",
            "optimizer_config": {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8, "clip_grad": 1.0, "weight_decay": 0.1},
            "model_config": {"hidden_dropout": 0.0}, **(extra_meta or {})}
    w = RunWriter(root / name, meta)
    for s in range(1, 11):
        w.log_metrics(s, lm_loss=3.0 - s / 10, lr=lr, wd=wd, grad_norm=0.5, param_norm=10.0,
                      **{"router.bias_std.1": 0.01 * s})
    p = rng.standard_normal((16, 8)).astype(np.float32)
    g = (rng.standard_normal((16, 8)) * 0.1).astype(np.float32)
    new, m, v = adamw_step(p, g, lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, wd=wd, step=1)
    if bad_update:
        new = p + (new - p) * 1.5
    w.write_tensors(1, "opt", {f"main_param.{P}": p, f"grad.{P}": g, f"main_param_after.{P}": new.astype(np.float32),
                               f"exp_avg.{P}": m.astype(np.float32), f"exp_avg_sq.{P}": v.astype(np.float32)},
                    kind="param", point="before_update")
    w.write_tensors(1, "layers.0", {"layers.0.attn.out": np.ones((4, 4), np.float32)}, kind="activation", point="forward")
    w.done()
    d = root / name
    (d / "data").mkdir()
    for r in range(2):
        with open(d / "data" / f"rank{r}.jsonl", "w") as f:
            for s in range(1, 11):
                tok = f"tok-{s}-{r}" + ("x" if token_shift and r == 0 and s >= 3 else "")
                f.write(json.dumps({"step": s, "micro": 0, "tokens": hashlib.sha256(tok.encode()).hexdigest(),
                                    "labels": "l", "loss_mask": "m", "position_ids": "p", "nonpad": 100}) + "\n")
    (d / "params_sha").mkdir()
    for r in range(2):
        for tag in ("init", "final"):
            (d / "params_sha" / f"rank{r}.{tag}.json").write_text(json.dumps({P: {"sha256": "aa", "shape": [16, 8], "dtype": "float32"}}))
    (d / "modules.txt").write_text("decoder.layers.1.mlp.router\tTopKRouter\n")


def full_checks():
    return [
        {"id": "L0.cfg", "type": "config_diff", "level": "L0", "select": {"kind": "config"},
         "params": {"mapping": [{"test": "model_config", "ref": "model_config", "must_equal": True}]}},
        {"id": "L0.data", "type": "data_fingerprint", "level": "L0", "select": {"kind": "tokens", "steps": "1:11"}},
        {"id": "L0.w", "type": "checkpoint_equal", "level": "L0", "select": {"kind": "checkpoint", "which": "init"}},
        {"id": "L0.struct", "type": "artifact_check", "level": "L0", "select": {"kind": "artifact", "file": "modules.txt"},
         "params": {"rules": [{"name": "router", "regex": "TopKRouter", "min": 1, "max": 1}]}},
        {"id": "L1.opt", "type": "optimizer_step", "level": "L1", "select": {"kind": "tensor", "step": 1, "name": "*"},
         "threshold": {"rel_l2": 1e-3}, "source": "测试：更新量 1e-3 量级、参数 1 量级，FP32 存储舍入约 6e-5"},
        {"id": "L2.rep", "type": "replica_consistency", "level": "L2", "select": {"kind": "checkpoint", "which": "final"}},
        {"id": "L4.bias", "type": "router_stats", "level": "L4", "select": {"kind": "metrics", "prefix": "router.bias_std.", "window": "1:11"},
         "threshold": "exact"},
        {"id": "L4.health", "type": "health_signals", "level": "L4", "select": {"kind": "metrics", "fields": ["lm_loss", "grad_norm"]}},
        {"id": "L4.det", "type": "determinism", "level": "L4", "select": {"kind": "metrics", "fields": ["lm_loss"]}},
        {"id": "L4.overlap", "type": "overlap_replay", "level": "L4", "select": {"kind": "metrics"}},
        {"id": "L2.layer{i}", "foreach": {"i": [0]}, "type": "bitwise_equal", "level": "L2",
         "select": {"kind": "tensor", "step": 1, "name": "layers.{i}.*"}},
    ]


# ---- TPU 侧（MaxText）的小运行：FP32 状态、meta、一段 HLO（日常回归与 dump 测试共用）

def tpu_state(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    return {
        "params/model/decoder/w": w,
        "MoEBiasVar/model/decoder/gate/bias": rng.standard_normal((3, 16)).astype(np.float32),
        "OptVariable/optimizer/opt_state/0/mu/decoder/w": (w * 0.1).astype(np.float32),
        "OptVariable/optimizer/opt_state/0/nu/decoder/w": (w * w).astype(np.float32),
        "OptArray/optimizer/opt_state/0/count": np.array(3, np.int32),
        "OptState/optimizer/step": np.array(3, np.uint32),
    }


def tpu_meta(name, **over):
    m = {k: {} for k in ("code", "versions", "hardware", "parallelism", "config", "checkpoint", "data", "steps",
                         "determinism", "dtypes")}
    m.update(schema_version=1, run_name=name, side="tpu", framework="maxtext", model="hunyuan3-4l", image="img@sha256:x",
             created_at="2026-11-04T12:00:00+08:00")
    m.update(over)
    return m


HLO_SAMPLE = '''HloModule jit_train_step
  %a = f32[2]{0} add(%x, %y), metadata={op_name="jit(train_step)/add" source_file="/x.py" source_line=3}, stack_frame_id=7
  %b = bf16[2]{0} custom-call(%a), custom_call_target="tpu_custom_call", metadata={op_name="k"}
  %splash_mha_fwd.1 = bf16[2]{0} custom-call(%b), custom_call_target="tpu_custom_call"
'''

