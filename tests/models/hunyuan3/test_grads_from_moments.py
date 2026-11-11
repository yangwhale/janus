"""grads_from_moments.py：从 Adam 一阶动量反推梯度。合成一个两步的运行目录（已知梯度、裁剪、FP32 动量），
确认反推的梯度对真值吻合到 FP32 舍入量级、状态类张量原样带过来；埋错的裁剪系数会被看出来（工具先自证）。"""
import importlib.util
import json

import numpy as np
import pytest
from helpers import ROOT

from janus.store import Run, RunWriter

SPEC = importlib.util.spec_from_file_location("grads_from_moments", ROOT / "models/hunyuan3/analysis/grads_from_moments.py")
gfm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gfm)

B1, CLIP = 0.9, 1.0
META = {"schema_version": 1, "run_name": "src", "side": "tpu", "framework": "maxtext", "model": "toy", "code": [], "image": "x",
        "versions": {}, "hardware": {}, "parallelism": {}, "config": {}, "checkpoint": {}, "data": {}, "steps": {"first": 1, "last": 2},
        "determinism": {}, "dtypes": {}, "created_at": "now",
        "optimizer_config": {"adam_beta1": B1, "adam_beta2": 0.95, "adam_eps": 1e-8, "clip_grad": CLIP}}


def make(tmp, norms=(17.5, 0.5)):
    """两步：第 1 步范数大于裁剪阈值（要裁），第 2 步小于（不裁）。返回 (运行目录, 真梯度 {步: {名: g}})。"""
    rng = np.random.default_rng(0)
    w = RunWriter(tmp / "src", META)
    true, m = {}, {}
    names = ["decoder.layers.0.self_attention.linear_proj.weight", "decoder.final_layernorm.weight"]
    for s, gn in zip((1, 2), norms):
        true[s] = {n: rng.standard_normal((8, 4) if "proj" in n else (8,)).astype(np.float32) for n in names}
        c = 1.0 if gn < CLIP else CLIP / gn
        for n in names:
            m[n] = (B1 * m.get(n, np.zeros_like(true[s][n])) + (1 - B1) * (true[s][n] * np.float32(c))).astype(np.float32)
        w.log_metrics(s, grad_norm=gn, lm_loss=1.0)
        w.write_tensors(s, "exp_avg.other", {f"exp_avg.{n}": m[n] for n in names}, kind="opt_state", point="after_update")
        w.write_tensors(s, "main_param.other", {f"main_param.{n}": np.ones_like(true[s][n]) for n in names},
                        kind="param", point="before_update")
        w.write_tensors(s, "grad.other", {f"grad.{n}": np.zeros_like(true[s][n]) for n in names}, kind="grad", point="backward")
    w.done()
    return tmp / "src", true


def test_recover_two_steps(tmp_path):
    src, true = make(tmp_path)
    assert gfm.main(["--src", str(src), "--out", str(tmp_path / "gm")]) == 0
    r = Run.open(tmp_path / "gm")
    for s in (1, 2):
        for n, g in true[s].items():
            got = r.tensor(s, f"grad.{n}").astype(np.float64)
            # 第 2 步要减掉 β1·m_1，误差按 |m| / |0.1·g| 放大；合成数据里同量级，FP32 舍入几个 ULP
            assert np.allclose(got, g, rtol=1e-5, atol=1e-6), (s, n)
        # 状态类张量原样带过来，加钩子抓的 grad.* 不带
        assert r.tensor(s, f"main_param.{n}").tobytes() == Run.open(src).tensor(s, f"main_param.{n}").tobytes()
    meta = json.loads((tmp_path / "gm" / "meta.json").read_text())
    assert meta["derived"]["steps"]["1"]["clip_coeff"] == pytest.approx(CLIP / 17.5)
    assert meta["derived"]["steps"]["2"]["clip_coeff"] == 1.0


def test_wrong_clip_is_visible(tmp_path):
    """记录的 grad_norm 错了（裁剪系数错），反推的梯度整体差一个比例——比较时会是 O(1) 的范数比偏差，而不是悄悄吻合。"""
    src, true = make(tmp_path)
    lines = (src / "metrics.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["grad_norm"] = 35.0
    (src / "metrics.jsonl").write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
    gfm.main(["--src", str(src), "--out", str(tmp_path / "gm"), "--steps", "1"])
    got = Run.open(tmp_path / "gm").tensor(1, "grad.decoder.final_layernorm.weight")
    ratio = np.linalg.norm(got) / np.linalg.norm(true[1]["decoder.final_layernorm.weight"])
    assert ratio == pytest.approx(2.0, rel=1e-5)


def test_needs_previous_step(tmp_path):
    src, _ = make(tmp_path)
    with pytest.raises(SystemExit):
        gfm.main(["--src", str(src), "--out", str(tmp_path / "gm"), "--steps", "3"])
