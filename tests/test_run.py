import hashlib
import json

import numpy as np
import pytest
import yaml
from helpers import BASE_META, catalog, make, tpu_meta

from janus.catalog import CatalogError, validate
from janus.checks.base import FAIL, INCONCLUSIVE
from janus.run import main
from janus.store import RunWriter


def test_validate_rules():
    good = catalog("/x")
    validate(good)
    bad = catalog("/x", [{"id": "L4.loss", "type": "nope", "level": "L9", "select": {"kind": "?"}, "threshold": {"a": 1}}])
    try:
        validate(bad)
        raise AssertionError("应当报错")
    except CatalogError as e:
        msg = str(e)
        for frag in ["id 重复", "未知类型", "level 不合法", "select.kind", "source"]:
            assert frag in msg


def test_run_end_to_end(tmp_path):
    for i in (1, 2, 3):
        make(tmp_path, f"g{i}", f"golden-{i}", seed=i)
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(catalog(tmp_path), allow_unicode=True))
    code = main(["run", str(cp), "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    assert code in (0, 1)
    assert {c["id"] for c in res["checks"]} == {"L4.loss", "L4.lr", "L2.act", "L2.cov"}
    assert all(c["status"] != "error" for c in res["checks"]), res["checks"]
    assert (tmp_path / "out" / "report.md").read_text().startswith("# janus 报告")
    # 目录指纹：内容指纹（content-v1）+ 原文件 sha256（只留作历史）
    from janus.catalog import HASH_ALGO, content_sha256

    assert res["schema_version"] == 2
    assert res["catalog"]["hash_algo"] == HASH_ALGO
    assert res["catalog"]["content_sha256"] == content_sha256(cp)
    assert res["catalog"]["raw_sha256"] == hashlib.sha256(cp.read_bytes()).hexdigest()
    assert "sha256" not in res["catalog"]
    assert f"内容指纹 {content_sha256(cp)[:12]}" in (tmp_path / "out" / "report.md").read_text()


def test_run_bad_catalog_exit_2(tmp_path):
    cp = tmp_path / "bad.yaml"
    cp.write_text("schema_version: 2\n")
    assert main(["run", str(cp), "--out", str(tmp_path / "o")]) == 2


def test_run_missing_run_is_error_exit_2(tmp_path):
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(catalog(tmp_path), allow_unicode=True))
    assert main(["run", str(cp), "--out", str(tmp_path / "o")]) == 2


def test_tensor_noise_parallel_matches_serial(tmp_path, monkeypatch):
    """噪声参照对并行计算与串行结果一致。"""
    for i, n in enumerate(["g1", "g2", "g3"]):
        make(tmp_path, n, f"golden-{i + 1}", seed=i + 1)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(catalog(tmp_path), allow_unicode=True))
    outs = {}
    for w in ("1", "3"):
        monkeypatch.setenv("JANUS_WORKERS", w)
        main(["run", str(p), "--out", str(tmp_path / f"o{w}")])
        res = json.loads((tmp_path / f"o{w}" / "results.json").read_text())
        outs[w] = {c["id"]: c.get("threshold") for c in res["checks"]}
    assert outs["1"]["L2.act"] == outs["3"]["L2.act"]


def test_coverage_exclude(tmp_path):
    """coverage 从目录调用时 select.exclude 两边都剔：参照独有的分发缓冲区不算缺。"""
    for name, extra in (("g1", True), ("g2", False)):
        w = RunWriter(tmp_path / name, {**BASE_META, "run_name": name, "side": "golden-1"})
        w.log_metrics(1, lm_loss=1.0)
        t = {"layers.1.mlp.out": np.zeros(2, np.float32)}
        if extra:
            t["layers.1.mlp.experts.out"] = np.zeros(3, np.float32)
        w.write_tensors(1, "layers.1", t, kind="activation", point="forward")
        w.done()
    cat = {"schema_version": 1, "model": "toy", "stage": "t", "runs": {"test": "g2", "golden": ["g1"], "root": str(tmp_path)},
           "checks": [{"id": "cov", "type": "coverage", "level": "L3", "select": {"kind": "tensor", "step": 1, "name": "*"}},
                      {"id": "cov.ex", "type": "coverage", "level": "L3",
                       "select": {"kind": "tensor", "step": 1, "name": "*", "exclude": "*.experts.out"}}]}
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(cat, allow_unicode=True))
    main(["run", str(cp), "--out", str(tmp_path / "out")])
    st = {c["id"]: c["status"] for c in json.loads((tmp_path / "out" / "results.json").read_text())["checks"]}
    assert st == {"cov": "error", "cov.ex": "pass"}


def test_checkpoint_equal_and_dtype_run_overrides(tmp_path):
    """checkpoint_equal 的 select.test_run / ref_run 与 dtype_manifest 的 select.run：一条检查改比两个指定的运行。"""
    def man_run(name, sha_b, dtype_b):
        w = RunWriter(tmp_path / name, {**BASE_META, "run_name": name, "side": "golden-1"})
        w.done()
        (tmp_path / name / "params_sha").mkdir()
        (tmp_path / name / "params_sha" / "rank0.init.json").write_text(json.dumps(
            {"a.weight": {"sha256": "1", "dtype": "bfloat16"}, "m.expert_bias": {"sha256": sha_b, "dtype": dtype_b}}))
    man_run("ckpt", "2", "float32")
    man_run("hf", "2", "float32")
    man_run("ckpt-bad", "3", "bfloat16")
    make(tmp_path, "t", "tpu", seed=1)                       # 目录的被测与 golden 是别的运行
    rules = [{"pattern": "*.expert_bias", "dtype": "float32"}, {"pattern": "*", "dtype": "bfloat16"}]
    checks = []
    for tag, ck in (("ok", "ckpt"), ("bad", "ckpt-bad")):
        checks += [{"id": f"cmp.{tag}", "type": "checkpoint_equal", "level": "L0",
                    "select": {"kind": "checkpoint", "which": "init", "manifest": True, "test_run": ck, "ref_run": "hf"}},
                   {"id": f"dt.{tag}", "type": "dtype_manifest", "level": "L0",
                    "select": {"kind": "checkpoint", "which": "init", "run": ck}, "params": {"rules": rules}}]
    cat = {"schema_version": 1, "model": "toy", "stage": "t", "runs": {"test": "t", "golden": ["t"], "root": str(tmp_path)},
           "checks": checks}
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(cat, allow_unicode=True))
    main(["run", str(cp), "--out", str(tmp_path / "out")])
    st = {c["id"]: c["status"] for c in json.loads((tmp_path / "out" / "results.json").read_text())["checks"]}
    assert st == {"cmp.ok": "pass", "dt.ok": "pass", "cmp.bad": "fail", "dt.bad": "fail"}


def test_select_exclude_accepts_list(tmp_path):
    """select.exclude 可以是一个通配，也可以是通配列表（被测比参照多出好几类观测张量时用）。"""
    from janus.run import Runner
    w = RunWriter(tmp_path / "r", {**BASE_META, "run_name": "r", "side": "golden-1"})
    z = np.zeros(2, np.float32)
    w.write_tensors(1, "layers.0", {"layers.0.out": z, "layers.0.attn.core.q": z, "layers.0.mlp.moe.permuted": z},
                    kind="activation", point="forward")
    w.done()
    from janus.store import Run
    run = Run.open(tmp_path / "r")
    assert Runner._names(run, {"step": 1, "name": "*", "exclude": "*.attn.core.q"}) == ["layers.0.mlp.moe.permuted", "layers.0.out"]
    assert Runner._names(run, {"step": 1, "name": "*", "exclude": ["*.attn.core.q", "*.moe.*"]}) == ["layers.0.out"]


def test_results_record_janus_version_and_commit(tmp_path):
    """results.json 记下 janus 的版本与提交号（git 工作区里装的才有提交号；有未提交改动时带 +dirty）。"""
    import re

    import janus
    from janus.run import janus_commit

    for i in (1, 2, 3):
        make(tmp_path, f"g{i}", f"golden-{i}", seed=i)
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(catalog(tmp_path), allow_unicode=True))
    main(["run", str(cp), "--out", str(tmp_path / "out")])
    meta = json.loads((tmp_path / "out" / "results.json").read_text())["janus"]
    assert meta["version"] == janus.__version__
    assert meta["commit"] == janus_commit()
    assert meta["commit"] is None or re.fullmatch(r"[0-9a-f]{40}(\+dirty)?", meta["commit"])


def test_pending_threshold_is_inconclusive_but_nonfinite_still_fails(tmp_path):
    base = {"schema_version": 1, "model": "m", "stage": "s", "runs": {"root": str(tmp_path), "test": "t", "lkg": "l"}}
    ok = {**base, "checks": [{"id": "x", "type": "loss_series", "level": "regress", "compare": "lkg",
                               "select": {"kind": "metrics", "field": "lm_loss", "window": "1:4"}, "threshold": "pending"}]}
    validate(ok)
    bad = {**base, "checks": [{**ok["checks"][0], "type": "tensor_tolerance"}]}
    with pytest.raises(CatalogError, match="pending"):
        validate(bad)
    for name, vals in (("l", [3.0, 2.0, 1.0]), ("t", [3.0, 2.0 + 1e-6, 1.0])):
        w = RunWriter(tmp_path / name, tpu_meta(name, steps={"first": 1, "last": 3}))
        for s, v in enumerate(vals, 1):
            w.log_metrics(s, lm_loss=v)
        w.done()
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(ok))
    out = tmp_path / "out"
    main(["run", str(tmp_path / "c.yaml"), "--out", str(out)])
    res = json.loads((out / "results.json").read_text())["checks"][0]
    assert res["status"] == INCONCLUSIVE and res["threshold"]["kind"] == "pending" and res["metrics"]["first_differ_step"] == 2
    w = RunWriter(tmp_path / "t2", tpu_meta("t2", steps={"first": 1, "last": 3}))
    for s, v in enumerate([3.0, float("nan"), 1.0], 1):
        w.log_metrics(s, lm_loss=v)
    w.done()
    main(["run", str(tmp_path / "c.yaml"), "--out", str(out), "--test", "t2"])
    assert json.loads((out / "results.json").read_text())["checks"][0]["status"] == FAIL


def test_multi_golden_noise_from_golden_ignores_runs_noise(tmp_path):
    """目录里 runs.noise 装的是另一种噪声（回归的 TPU 执行噪声）时，noise_from: golden 的 multi_golden 仍用 golden 两两组合。"""
    for i in (1, 2, 3):
        make(tmp_path, f"g{i}", f"golden-{i}", seed=i)
        make(tmp_path, f"n{i}", "tpu", seed=10 + i, noise=1e-7)
    make(tmp_path, "t", "tpu", seed=20)
    loss = {"id": "L4.loss", "type": "multi_golden", "level": "L4", "threshold": "calibrated",
            "select": {"kind": "metrics", "field": "lm_loss", "window": "1:51"}}
    cat = {"schema_version": 1, "model": "toy", "stage": "t",
           "runs": {"test": "t", "golden": ["g1", "g2", "g3"], "root": str(tmp_path),
                    "noise": [["n1", "n2"], ["n1", "n3"], ["n2", "n3"]]},
           "checks": [loss, {**loss, "id": "L4.loss_g", "params": {"noise_from": "golden"}}]}
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(cat, allow_unicode=True))
    main(["run", str(cp), "--out", str(tmp_path / "out")])
    res = {c["id"]: c for c in json.loads((tmp_path / "out" / "results.json").read_text())["checks"]}
    pairs = {k: [r["pair"] for r in v["metrics"]["noise_loo"]] for k, v in res.items()}
    assert pairs["L4.loss"] == ["n1~n2", "n1~n3", "n2~n3"]
    assert pairs["L4.loss_g"] == ["g1~g2", "g1~g3", "g2~g3"]
    assert res["L4.loss"]["status"] != res["L4.loss_g"]["status"] or res["L4.loss"]["threshold"] != res["L4.loss_g"]["threshold"]
