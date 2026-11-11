"""混元 3 的日常回归：回归目录（T0 / T1）端到端（含故意改坏一位的自证）、regress.py 的改判与报告、HLO 收集。
日常回归用到的 janus 本身的能力（FP32 状态清单、run_complete、artifact_check、待标定门槛、目录 include / vars、
config_diff 字典比）在 tests/dump/、tests/checks/、tests/test_run.py、tests/test_catalog.py 里按模块测。"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from helpers import HLO_SAMPLE, ROOT, tpu_meta, tpu_state  # noqa: E402

from janus.catalog import load
from janus.checks.base import FAIL, PASS
from janus.dump import maxtext as jd
from janus.run import main as janus_main
from janus.store import RunWriter

REG = ROOT / "models" / "hunyuan3" / "regression"
sys.path.insert(0, str(ROOT / "models" / "hunyuan3" / "frameworks" / "maxtext"))


# ------------------------------------------------------------------------------------------------ FP32 状态清单


# ------------------------------------------------------------------------------------------------ 新检查与扩展


# ------------------------------------------------------------------------------------------------ 混元 3 的回归目录端到端
R06 = {"dropout_rate": 0.0, "enable_dropout": False, "use_random_routing": False, "load_balance_loss_weight": 0.0,
       "z_loss_multiplier": 0.0, "mtp_num_layers": 0, "adam_weight_decay": 0.033, "adamw_mask": ["norm/scale"],
       "learning_rate": 1e-5, "gradient_clipping_threshold": 1.0, "skip_step_on_spikes": False,
       "routed_bias_update_rate": 0.001, "matmul_precision": "default", "float32_gate_logits": True,
       "capacity_factor": -1.0, "remat_policy": "custom", "moe_routing_maps": "device", "weight_dtype": "float32",
       "mu_dtype": "float32", "grad_dtype": "float32", "per_device_batch_size": 2, "gradient_accumulation_steps": 1,
       "shard_optimizer_over_data": False, "use_tokamax_splash": False, "routed_gate_bf16_rounding": True,
       "rmsnorm_single_rounding": True, "mlp_activation_round_per_op": True, "dense_fp32_accumulation": True}
HLO_TPU = HLO_SAMPLE + '  %g = bf16[2]{0} custom-call(%b), custom_call_target="tpu_custom_call", backend_config=\\"tiling\\": {\\"tile_m\\": 512}\n'


def make_regression_run(root: Path, name: str, steps=20, poison_final=False, extra_cfg=None, loss_scale=1.0, loss_from=2):
    """按混元 3 TPU 侧的运行目录格式造一份小运行：meta 里 T0 目录要读的字段、逐步标量、输入指纹、FP32 状态清单、HLO_SAMPLE。"""
    cat = yaml.safe_load((REG / "catalogs" / "regression-common.yaml").read_text())
    pins = {m["test"]: m["equals"] for c in cat["checks"] if c["id"] == "R02.inputs"
            for m in c["params"]["mapping"] if "equals" in m}
    host = {"TPU_WORKER_ID": name[-1], "TPU_WORKER_HOSTNAMES": f"h-{name}", "XLA_FLAGS": "--xla_dump_to=/data/xla",
            "LIBTPU_INIT_ARGS": " --xla_tpu_use_enhanced_launch_barrier=true"}
    flags = {"env": host, "jax_config": {"jax_enable_x64": False}}
    meta = tpu_meta(
        name, steps={"first": 1, "last": steps, "convention": "golden_step = step"},
        code=[{"repo": "maxtext", "commit": "c0ffee"}, {"repo": "janus", "commit": "beef"}],
        versions={"jax": "0.11.1", "libtpu": "x"}, hardware={"accelerator": "TPU v5", "count": 8, "topology": "2x2x2",
                                                              "processes": 2},
        checkpoint={"path": "gs://<POC 桶>/ckpt", "manifest_sha256": pins["checkpoint.manifest_sha256"]},
        data={"manifest_sha256": pins["data.manifest_sha256"], "replay_of": {"run": "g1k-a"}},
        config={"sha256": pins["config.sha256"], "overrides": ["mtp_num_layers=0", "per_device_batch_size=2"],
                "args": {"steps": steps, "janus_run": f"/data/runs/{name}"}},
        determinism={"remat_policy": "custom"}, flags={"at_start": flags, "after_compile": flags},
        maxtext_config_full={**R06, "run_name": name, "base_output_directory": f"/tmp/{name}", **(extra_cfg or {})},
        scheduler_config={"lr_decay_style": "constant"}, restarts=0)
    w = RunWriter(root / name, meta)
    for s in range(1, steps + 1):
        w.log_metrics(s, lm_loss=10.0 / s * (loss_scale if s >= loss_from else 1.0), loss=10.0 / s, raw_grad_norm=1.0 + 1.0 / s, param_norm=100.0 + s * 1e-3,
                      lr=1e-5, wd=0.033, step_time=1.0 + 0.01 * (s % 3), **{"router.bias_mean.1": 0.0, "router.bias_std.1": 1e-3 * s})
    w.done()
    d = root / name
    (d / "data").mkdir(exist_ok=True)
    for r in range(8):
        with open(d / "data" / f"rank{r}.jsonl", "w") as f:
            for s in range(1, steps + 1):
                for m in range(2):
                    h = hashlib.sha256(f"{s}.{r}.{m}".encode()).hexdigest()
                    f.write(json.dumps({"step": s, "micro": m, "tokens": h, "labels": h, "loss_mask": h,
                                        "position_ids": h, "nonpad": 4096}) + "\n")
    (d / "state_sha").mkdir(exist_ok=True)
    init = jd.state_manifest(tpu_state(0))
    fin_state = tpu_state(1)
    if poison_final:     # FP32 主权重一个元素改 1 个 ulp：BF16 清单看不见，R15 必须报
        x = fin_state["params/model/decoder/w"].reshape(-1)
        x[3] = np.nextafter(x[3], np.float32(np.inf))
    (d / "state_sha" / "init.json").write_text(json.dumps(init))
    (d / "state_sha" / "final.json").write_text(json.dumps(jd.state_manifest(fin_state)))
    (d / "hlo").mkdir(exist_ok=True)
    (d / "hlo" / "train_step.after_optimizations.txt").write_text(HLO_TPU.replace("source_line=3", f"source_line={len(name)}"))
    (d / "hlo" / "train_step.flagfile").write_text("--xla_a=1\n--xla_dump_to=/data/xla\n")
    (d / "hlo" / "train_step.debug_options").write_text('xla_dump_to: "/data/xla"\n')
    return d


def make_noise_runs(root: Path, tier="T0"):
    """目录 runs.noise 用到的标定运行：造同名小运行，第 2 步起 loss 各差一点（各对不逐位，噪声上限约 1e-6 量级）。"""
    cat = load(REG / "catalogs" / f"regression-{tier.lower()}.yaml")
    names = list(dict.fromkeys(n for pr in cat.runs.get("noise") or [] for n in pr))
    for i, n in enumerate(names):
        make_regression_run(root, n, loss_scale=1.0 + 1e-6 * (i + 1))
    return names


def _run_t0(tmp_path, test, lkg):
    out = tmp_path / f"res-{test}"
    rc = janus_main(["run", str(REG / "catalogs" / "regression-t0.yaml"), "--root", str(tmp_path), "--test", test,
                     "--lkg", lkg, "--out", str(out)])
    return rc, out, {c["id"]: c for c in json.loads((out / "results.json").read_text())["checks"]}


def _report(tmp_path, out, test, lkg, *extra):
    od = tmp_path / f"rep-{test}"
    r = subprocess.run([sys.executable, str(REG / "regress.py"), "report", "--tier", "T0", "--janus", str(out),
                        "--root", str(tmp_path), "--test", test, "--lkg", lkg, "--out", str(od), "--platform", "v5p", *extra],
                       capture_output=True, text=True)
    return r.returncode, json.loads((od / "regression.json").read_text()), (od / "report.md").read_text()


def test_regression_t0_catalog_bitwise_pass(tmp_path):
    make_regression_run(tmp_path, "lkg-a")
    make_regression_run(tmp_path, "lkg-b")
    make_noise_runs(tmp_path)
    rc, out, res = _run_t0(tmp_path, "lkg-b", "lkg-a")
    assert rc == 0, {k: (v["status"], v.get("message")) for k, v in res.items() if v["status"] != PASS}
    assert res["R08.loss_tolerance"]["status"] == PASS                # 容差门已标定：逐位的被测在噪声上限内
    assert res["R03.data_fingerprint"]["metrics"]["units"] == 1300
    assert res["R20.artifact_counts"]["status"] == PASS
    rc, rep, md = _report(tmp_path, out, "lkg-b", "lkg-a", "--first-lkg")
    assert rc == 0 and rep["class"] == "通过" and rep["lkg_candidate"], md
    assert rep["rows"]["R23"]["status"] == "pass" and rep["rows"]["R22"]["status"] == "report"
    assert md.startswith("```\n[T0 · v5p · ")


def test_regression_t0_catalog_catches_one_ulp_in_final_state(tmp_path):
    """自证：只有训练后的 FP32 主权重差 1 个 ulp，其余全同 → R15 报 fail，报告归为数值、第一个不同是 R15。"""
    make_noise_runs(tmp_path)
    make_regression_run(tmp_path, "lkg-a")
    make_regression_run(tmp_path, "t-bad", poison_final=True)
    rc, out, res = _run_t0(tmp_path, "t-bad", "lkg-a")
    assert rc == 1 and res["R15.state_final"]["status"] == FAIL
    assert [k for k, v in res.items() if v["status"] == FAIL] == ["R15.state_final"]
    rc, rep, md = _report(tmp_path, out, "t-bad", "lkg-a")
    assert rc == 1 and rep["class"] == "数值" and rep["first_differ"] == "R15" and not rep["lkg_candidate"]
    # 声明「会改数值 · 数学级」且范围含 R15：这一行改为只报，整体通过
    rc, rep, _ = _report(tmp_path, out, "t-bad", "lkg-a", "--declare", "数学级", "--scope", "R15")
    assert rep["rows"]["R15"]["status"] == "report" and rep["class"] == "通过"
    # 声明「执行级」但范围不含 R15：仍 fail，并写明要退回
    rc, rep, _ = _report(tmp_path, out, "t-bad", "lkg-a", "--declare", "执行级", "--scope", "R12")
    assert rep["rows"]["R15"]["status"] == "fail" and "退回" in rep["rows"]["R15"]["note"]
    # 执行级且在范围内：R15 没有容差门 → inconclusive
    rc, rep, _ = _report(tmp_path, out, "t-bad", "lkg-a", "--declare", "执行级", "--scope", "R15")
    assert rep["rows"]["R15"]["status"] == "inconclusive" and rc == 3


def test_regression_t0_catalog_flags_config_change(tmp_path):
    """档位以外多开一个精度开关：R05 的完整配置比对报出那一个键。"""
    make_noise_runs(tmp_path)
    make_regression_run(tmp_path, "lkg-a")
    make_regression_run(tmp_path, "t-flag", extra_cfg={"lm_head_dh_wbar_remove": True})
    rc, out, res = _run_t0(tmp_path, "t-flag", "lkg-a")
    assert res["R05.process_flags"]["status"] == FAIL and "maxtext_config_full" in res["R05.process_flags"]["message"]
    det = [json.loads(x) for x in (out / "details" / "R05.process_flags.jsonl").read_text().splitlines()]
    cfg = next(d for d in det if d["name"].startswith("maxtext_config_full"))
    assert [x["key"] for x in cfg["differ"]] == ["lm_head_dh_wbar_remove"]


def test_regression_catalogs_load_and_cover_tier_rows():
    sys.path.insert(0, str(REG))
    import regress
    for tier in ("T0", "T1"):
        cat = load(REG / regress.tier_cfg(tier)["catalog"])
        rows = {c.get("row") for c in cat.checks if c.get("role") != "tolerance"}
        want = set(regress.ROWS[tier]) - {"R22", "R23"}
        assert rows == want, (tier, rows ^ want)
        assert len({c["id"] for c in cat.checks}) == len(cat.checks)


def test_collect_hlo(tmp_path):
    import train_hy3
    src = tmp_path / "xla"
    src.mkdir()
    for suf in ("after_optimizations.txt", "flagfile", "debug_options", "after_optimizations-buffer-assignment.txt"):
        (src / f"module_0366.jit_train_step.cl_963683785.{suf}").write_text(suf)
    (src / "module_0400.jit_hooked.cl_1.after_optimizations.txt").write_text("hooked")
    info = train_hy3.collect_hlo(src, tmp_path / "run")
    assert sorted(info["files"]) == ["train_step.after_optimizations.txt", "train_step.debug_options", "train_step.flagfile"]
    assert "error" not in info
    assert (tmp_path / "run" / "hlo" / "train_step.flagfile").read_text() == "flagfile"
    (src / "module_0500.jit_train_step.cpu_after_optimizations.txt").write_text("again")
    info = train_hy3.collect_hlo(src, tmp_path / "run2")
    assert "编译了 2 次" in info["error"] and "train_step.1.after_optimizations.txt" in info["files"]
    assert "dump 目录里没有" in train_hy3.collect_hlo(tmp_path / "none", tmp_path / "run3")["error"]


def test_regression_t0_tolerance_gate_gives_verdicts(tmp_path):
    """容差门标定后给真结论：loss 差 1e-7 量级的在噪声内 pass，差 1e-3 的超过 2 倍上限 fail；
    声明「执行级」时 regress.py 采用容差门，但标定绑定的镜像与依赖和被测不同时降为 inconclusive（参照作废）。"""
    make_regression_run(tmp_path, "lkg-a")
    make_noise_runs(tmp_path)
    # 第 1 步就不同（R08 的逐位检查不过），才轮到容差门
    make_regression_run(tmp_path, "t-small", loss_scale=1.0 + 1e-7, loss_from=1)
    make_regression_run(tmp_path, "t-big", loss_scale=1.0 + 1e-3, loss_from=1)
    _, out_s, res = _run_t0(tmp_path, "t-small", "lkg-a")
    assert res["R08.loss_tolerance"]["status"] == PASS and res["R08.loss_tolerance"]["threshold"]["kind"] == "calibrated"
    assert res["R08.loss_tolerance"]["threshold"]["noise"]["refs"] >= 3
    _, out_b, res = _run_t0(tmp_path, "t-big", "lkg-a")
    assert res["R08.loss_tolerance"]["status"] == FAIL
    _, rep, _ = _report(tmp_path, out_b, "t-big", "lkg-a", "--declare", "执行级", "--scope", "R08")
    assert rep["rows"]["R08"]["status"] == "fail"
    _, rep, _ = _report(tmp_path, out_s, "t-small", "lkg-a", "--declare", "执行级", "--scope", "R08")
    assert rep["rows"]["R08"]["status"] == "inconclusive" and "噪声参照作废" in rep["rows"]["R08"]["note"]


def test_calibration_yaml_matches_catalogs():
    """calibration.yaml 最新一条与目录一致：runs.noise 就是那一条的 pairs；绑定的档位配置 sha256 是现在的文件；
    参照运行都写了占位位置（不写桶名）；对数 ≥ 3 才能在目录里用 calibrated。"""
    sys.path.insert(0, str(REG))
    import regress
    for tier in ("T0", "T1"):
        cal = regress.load_calibration("v5p", tier)
        assert cal is not None, tier
        cat = load(REG / regress.tier_cfg(tier)["catalog"])
        assert [list(p) for p in cat.runs["noise"]] == [list(p) for p in cal["pairs"]], tier
        assert cal["n_pairs"] == len(cal["pairs"]) >= 3
        assert cal["binding"]["tier_config_sha256"] == regress.tier_cfg(tier)["_sha256"]
        assert all(v.startswith("gs://<POC 桶>/") for v in cal["runs"].values()), cal["runs"]
        assert {n for p in cal["pairs"] for n in p} <= set(cal["runs"])
        tol = [c for c in cat.checks if c.get("role") == "tolerance"]
        assert tol and all(c["threshold"] == "calibrated" for c in tol), tier
    r25 = next(c for c in load(REG / "catalogs" / "regression-t1.yaml").checks if c["id"] == "R25.loss_vs_golden")
    assert r25["params"]["noise_from"] == "golden"     # 对 GPU golden 的噪声仍是 golden 两两，不换成 TPU 执行噪声
    assert r25["params"]["baseline"] == "g1k-a"        # 只对指定的一份 golden 比，其余只组噪声带（docs/reference/judging.md）


def test_calibration_binding_includes_training_yml():
    """标定绑定训练配置 hy3-4layers.yml 的 sha256：被测的配置变了（例如打开一个会改数值的开关）参照作废；
    老条目没记这一项也按作废处理。"""
    sys.path.insert(0, str(REG))
    import regress

    cfg = regress.tier_cfg("T1")
    meta = {"image": "img@sha256:1", "versions": {"jax": "1"}, "config": {"sha256": "aaa"}}
    test = type("R", (), {"meta": meta})()
    good = {"binding": {"tier_config_sha256": cfg["_sha256"], "config_yml_sha256": "aaa",
                        "image": "img@sha256:1", "versions": {"jax": "1"}}}
    assert regress.calibration_valid(good, test, cfg) == (True, "")
    changed = {"binding": {**good["binding"], "config_yml_sha256": "bbb"}}
    ok, why = regress.calibration_valid(changed, test, cfg)
    assert not ok and "hy3-4layers.yml" in why and "作废" in why
    old = {"binding": {k: v for k, v in good["binding"].items() if k != "config_yml_sha256"}}
    ok, why = regress.calibration_valid(old, test, cfg)
    assert not ok and "没记" in why


def test_calibration_yml_binding_matches_pinned_config():
    """calibration.yaml 最新条目绑定的训练配置 sha256，就是回归目录 R02 钉死的那份（配置换了要一起重标）。"""
    sys.path.insert(0, str(REG))
    import regress

    cat = yaml.safe_load((REG / "catalogs" / "regression-common.yaml").read_text())
    pinned = next(m["equals"] for c in cat["checks"] if c["id"] == "R02.inputs"
                  for m in c["params"]["mapping"] if m["test"] == "config.sha256")
    for tier in ("T0", "T1"):
        assert regress.load_calibration("v5p", tier)["binding"]["config_yml_sha256"] == pinned, tier


def test_r02_registry_compares_catalog_content_hash(tmp_path, monkeypatch):
    """R02 登记核对：结果与 lkg.yaml 都有内容指纹时比内容指纹（只改注释、原文件 sha256 变了也 pass，改了检查才 fail）；
    旧结果（schema_version 1，只有原文件 sha256）退回比 catalog_sha256。"""
    sys.path.insert(0, str(REG))
    import regress
    cfg = {"_sha256": "tier"}
    ent = {"run": "lkg-a", "tier_config_sha256": "tier", "catalog_sha256": {"c.yaml": "raw-old"},
           "catalog_content_sha256": {"c.yaml": "content-1"}}
    (tmp_path / "lkg.yaml").write_text(yaml.safe_dump({"entries": [ent]}))
    monkeypatch.setattr(regress, "HERE", tmp_path)

    def r02(catalog):
        rows = {}
        regress.registry_check(rows, "lkg-a", cfg, {"catalog": catalog}, "数学不变", set())
        return rows["R02"]["checks"][0]

    new = {"path": "/x/c.yaml", "hash_algo": "content-v1", "content_sha256": "content-1", "raw_sha256": "raw-new"}
    assert r02(new)["status"] == PASS and r02(new)["metrics"]["catalog_hash"] == "content-v1"
    assert r02({**new, "content_sha256": "content-2"})["status"] == FAIL
    assert r02({"path": "/x/c.yaml", "sha256": "raw-old"})["status"] == PASS          # 旧结果：比原文件
    assert r02({"path": "/x/c.yaml", "sha256": "raw-new"})["status"] == FAIL
