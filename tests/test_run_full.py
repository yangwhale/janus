"""运行器接上的其余类型：用小运行目录（含输入指纹、参数清单、优化器状态）把整条链跑一遍，并验证埋错会被抓。"""
import json

import numpy as np
import yaml
from helpers import BASE_META, P, full_checks, make_full

from janus.catalog import expand
from janus.replay import adamw_step
from janus.run import main
from janus.store import RunWriter


def run(tmp_path, test, golden="a"):
    cat = {"schema_version": 1, "model": "toy", "stage": "t", "runs": {"test": test, "golden": [golden], "root": str(tmp_path)},
           "checks": full_checks()}
    p = tmp_path / f"cat-{test}.yaml"
    p.write_text(yaml.safe_dump(cat, allow_unicode=True))
    out = tmp_path / f"out-{test}"
    rc = main(["run", str(p), "--out", str(out), "--html"])
    res = json.loads((out / "results.json").read_text())
    return rc, {c["id"]: c for c in res["checks"]}, out


def test_full_chain_clean_and_faults(tmp_path):
    make_full(tmp_path, "a")
    make_full(tmp_path, "b")
    make_full(tmp_path, "shift", token_shift=True)
    make_full(tmp_path, "upd", bad_update=True)
    make_full(tmp_path, "drop", extra_meta={"model_config": {"hidden_dropout": 0.1}})
    rc, st, out = run(tmp_path, "b")
    assert rc == 0, {k: (v["status"], v.get("message")) for k, v in st.items() if v["status"] != "pass"}
    assert (out / "report.html").exists() and "四级门槛" in (out / "report.html").read_text()
    assert st["L1.opt"]["metrics"]["weight_decay"] == 0.033          # 取每步记录的实际 wd，不是配置里的 0.1
    _, st, _ = run(tmp_path, "shift")
    assert st["L0.data"]["status"] == "fail"
    _, st, _ = run(tmp_path, "upd")
    assert st["L1.opt"]["status"] == "fail"
    _, st, _ = run(tmp_path, "drop")
    assert st["L0.cfg"]["status"] == "fail"


def test_foreach_expand():
    out = expand([{"id": "x{i}.{j}", "foreach": {"i": [0, 1], "j": ["a"]}, "select": {"name": "layers.{i}.*"}}])
    assert [c["id"] for c in out] == ["x0.a", "x1.a"] and out[1]["select"]["name"] == "layers.1.*"


def _run_checks(tmp_path, test, chks, golden="a", tag="x"):
    cat = {"schema_version": 1, "model": "toy", "stage": "t", "runs": {"test": test, "golden": [golden], "root": str(tmp_path)},
           "checks": chks}
    p = tmp_path / f"cat-{tag}-{test}.yaml"
    p.write_text(yaml.safe_dump(cat, allow_unicode=True))
    out = tmp_path / f"out-{tag}-{test}"
    main(["run", str(p), "--out", str(out)])
    return {c["id"]: c for c in json.loads((out / "results.json").read_text())["checks"]}


def test_step_convention_off_by_one_drill(tmp_path):
    """演练：被测 meta 的步号约定故意写错一格，首步检查与偏移探测都必须红；约定正确时不红。"""
    make_full(tmp_path, "a")
    make_full(tmp_path, "b")
    wrong = {**BASE_META["steps"], "convention": "golden_step = step + 1"}
    make_full(tmp_path, "w", extra_meta={"steps": wrong})
    chks = [{"id": "L4.loss.first", "type": "loss_series", "level": "L4",
             "select": {"kind": "metrics", "field": "lm_loss", "window": "2:11"}, "threshold": "exact",
             "params": {"first_step_tol": 0.0, "first_step_source": "测试：同配置首步必须逐位相同"}},
            {"id": "L4.loss.probe", "type": "loss_series", "level": "L4",
             "select": {"kind": "metrics", "field": "lm_loss", "window": "2:11"}, "threshold": "exact",
             "params": {"auto_offset": [-2, 2]}},
            {"id": "L4.lr", "type": "scalar_series", "level": "L4", "select": {"kind": "metrics", "field": "lr", "window": "2:11"}}]
    st = _run_checks(tmp_path, "b", chks, tag="conv")
    assert all(c["status"] == "pass" for c in st.values()), st
    st = _run_checks(tmp_path, "w", chks, tag="conv")
    assert st["L4.loss.first"]["status"] == "fail" and "首步差" in st["L4.loss.first"]["message"]
    assert st["L4.loss.probe"]["status"] == "fail"
    assert "golden_step = step +0" in st["L4.loss.probe"]["message"]        # 报出探测到的真实偏移


def test_new_types_from_catalog(tmp_path):
    import ml_dtypes
    make_full(tmp_path, "a")
    make_full(tmp_path, "b")
    chks = [{"id": "L0.dtype", "type": "dtype_manifest", "level": "L0", "select": {"kind": "checkpoint", "which": ["init", "final"]},
             "params": {"rules": [{"pattern": "buffer.*expert_bias", "dtype": "float32"},
                                  {"pattern": "*", "exclude": "buffer.*", "dtype": "float32"}]}},
            {"id": "L3.fake", "type": "fake_fp32", "level": "L3", "select": {"kind": "tensor", "step": 1, "name": "main_param_after.*"}},
            {"id": "L3.fake.before", "type": "fake_fp32", "level": "L3", "select": {"kind": "tensor", "step": 1, "name": "main_param.*"}},
            {"id": "L0.data", "type": "data_fingerprint", "level": "L0", "select": {"kind": "tokens", "steps": "1:11"}}]
    st = _run_checks(tmp_path, "b", chks, tag="new")
    assert st["L0.dtype"]["status"] == "pass" and st["L0.dtype"]["metrics"]["entries"] == 4
    assert st["L3.fake"]["status"] == "pass"
    assert st["L3.fake.before"]["status"] == "pass"
    assert st["L0.data"]["metrics"]["units_per_field"] == {"tokens": 20, "labels": 20, "loss_mask": 20, "position_ids": 20}
    # 被测把主权重存成了 BF16 精度：假 FP32 必须红
    from janus.store import Run
    r = Run.open(tmp_path / "b")
    w = RunWriter(tmp_path / "fake", {**r.meta, "run_name": "fake"})
    for s in range(1, 11):
        w.log_metrics(s, **{k: v for k, v in r.metrics()[s].items() if k != "step"})
    w.write_tensors(1, "opt", {f"main_param.{P}": r.tensor(1, f"main_param.{P}").astype(ml_dtypes.bfloat16).astype(np.float32)},
                    kind="param", point="before_update")
    w.done()
    st = _run_checks(tmp_path, "fake", chks[2:3], tag="new")
    assert st["L3.fake.before"]["status"] == "fail"
    # 指纹少一个字段：error，不是跳过
    f = tmp_path / "b" / "data" / "rank1.jsonl"
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    lines[3].pop("position_ids")
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    st = _run_checks(tmp_path, "b", chks[3:], tag="miss")
    assert st["L0.data"]["status"] == "error" and "position_ids" in st["L0.data"]["message"]


def test_optimizer_multi_step_and_rename(tmp_path):
    """连续两步重放（第 2 步用第 1 步的动量）；ref_rename 在同一运行里比两组张量。"""
    rng = np.random.default_rng(1)
    w = RunWriter(tmp_path / "m", {**BASE_META, "run_name": "m", "side": "golden-m",
                                   "optimizer_config": {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8, "clip_grad": 0.0}})
    p = rng.standard_normal((8, 4)).astype(np.float32)
    m = v = None
    for s in (1, 2):
        w.log_metrics(s, lm_loss=1.0, lr=1e-3, wd=0.0, grad_norm=0.1)
        g = (rng.standard_normal((8, 4)) * 0.1).astype(np.float32)
        new, m, v = adamw_step(p, g, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0, step=s, m0=m, v0=v)
        w.write_tensors(s, "opt", {f"main_param.{P}": p, f"grad.{P}": g, f"main_param_after.{P}": new.astype(np.float32),
                                   f"exp_avg.{P}": m.astype(np.float32), f"exp_avg_sq.{P}": v.astype(np.float32),
                                   "x.out": p, "x.out.recompute": p}, kind="param", point="before_update")
        p = new.astype(np.float32)
    w.done()
    cat = {"schema_version": 1, "model": "toy", "stage": "t", "runs": {"test": "m", "golden": ["m"], "root": str(tmp_path)},
           "checks": [{"id": "opt2", "type": "optimizer_step", "level": "L1", "select": {"kind": "tensor", "steps": "1:3", "name": "*"},
                       "threshold": {"rel_l2": 1e-3}, "source": "测试"},
                      {"id": "remat", "type": "bitwise_equal", "level": "L2",
                       "select": {"kind": "tensor", "step": 1, "name": "*.recompute", "ref_rename": {"from": ".recompute", "to": ""}}}]}
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(cat, allow_unicode=True))
    assert main(["run", str(cp), "--out", str(tmp_path / "o")]) == 0

    # 多步汇总不能把 inconclusive 抹成 pass（以前只认 fail / error，其余一律 pass）
    from janus.checks import REGISTRY
    from janus.checks.base import CheckResult

    def stub(cid, level, *a, **k):
        return CheckResult(cid, "optimizer_step", level, True, status="inconclusive", metrics={"rel_l2_max": 1e-4})

    orig = REGISTRY["optimizer_step"]
    REGISTRY["optimizer_step"] = stub
    try:
        main(["run", str(cp), "--out", str(tmp_path / "o2")])
    finally:
        REGISTRY["optimizer_step"] = orig
    st = {c["id"]: c["status"] for c in json.loads((tmp_path / "o2" / "results.json").read_text())["checks"]}
    assert st["opt2"] == "inconclusive"


def test_unexpected_exception_in_one_check_is_error_and_results_still_written(tmp_path, monkeypatch):
    """一条检查里出了意料之外的异常：记成这条的 error（退出码 2），其余照跑，results.json 照写。"""
    from janus.checks import REGISTRY

    make_full(tmp_path, "a")
    make_full(tmp_path, "b")
    orig = REGISTRY["config_diff"]
    monkeypatch.setitem(REGISTRY, "config_diff", lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("故意的")))
    cat = {"schema_version": 1, "model": "toy", "stage": "t", "runs": {"test": "b", "golden": ["a"], "root": str(tmp_path)},
           "checks": full_checks()}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cat, allow_unicode=True))
    try:
        rc = main(["run", str(p), "--out", str(tmp_path / "o")])
    finally:
        REGISTRY["config_diff"] = orig
    st = {c["id"]: c for c in json.loads((tmp_path / "o" / "results.json").read_text())["checks"]}
    assert rc == 2 and st["L0.cfg"]["status"] == "error" and "ZeroDivisionError: 故意的" in st["L0.cfg"]["message"]
    assert sum(1 for c in st.values() if c["status"] == "pass") >= 3


def test_render_only_without_results_is_exit_2(tmp_path, capsys):
    make_full(tmp_path, "a")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 1, "model": "toy", "stage": "t",
                                 "runs": {"test": "a", "golden": ["a"], "root": str(tmp_path)}, "checks": full_checks()[:1]}))
    assert main(["run", str(p), "--out", str(tmp_path / "nothing"), "--render-only"]) == 2
    assert "render-only" in capsys.readouterr().err



def test_out_of_tol_noise_uses_same_tolerance_as_test(tmp_path):
    """out_of_tol_ratio：噪声参照与被测用同一组 atol / rtol（以前噪声那边固定用 0，两把尺子）。"""
    from janus.run import Runner

    class Cat:
        runs = {"root": str(tmp_path), "noise": []}

    r = Runner.__new__(Runner)
    assert r._tol({"params": {"atol": 1e-3, "rtol": 2e-2}}) == (1e-3, 2e-2)
    assert r._tol({}) == (0.0, 0.0)
    base = np.linspace(-1, 1, 64).astype(np.float32)
    for name, eps in (("n1", 0.0), ("n2", 5e-4)):
        w = RunWriter(tmp_path / name, {**BASE_META, "run_name": name, "side": "golden-1"})
        w.write_tensors(1, "x", {"x.out": (base + eps).astype(np.float32)}, kind="activation", point="forward")
        w.done()
    r.root, r._runs, r._rows = tmp_path, {}, {}
    loose = r._row("n1", "n2", 1, "x.out", ["out_of_tol_ratio"], (1e-3, 0.0))["out_of_tol_ratio"]
    strict = r._row("n1", "n2", 1, "x.out", ["out_of_tol_ratio"], (0.0, 0.0))["out_of_tol_ratio"]
    assert loose == 0.0 and strict == 1.0                   # 容差进了计算，也进了缓存键（两次不串）
