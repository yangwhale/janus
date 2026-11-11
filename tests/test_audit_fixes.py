"""复现审计（models/hunyuan3/alignments/00-gpu-noise-and-drills/results/repro-audit.md 第四节）记录的代码问题，每条一个回归测试。"""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from helpers import ROOT, catalog, full_checks, make, make_full

from janus.checks import REGISTRY
from janus.checks.base import ERROR, FAIL, PASS
from janus.dump.runinfo import image_ref, pkg_version
from janus.run import main
from janus.store import Run, StoreError, side_error

REPO = ROOT
GPU = REPO / "models" / "hunyuan3" / "frameworks" / "megatron"


# ---------------------------------------------------------------- b：side 不合法
def test_side_error_lists_legal_prefixes():
    assert side_error("golden-1") is None and side_error("tpu") is None and side_error("gpu-perturb-2") is None
    msg = side_error("test")
    for frag in ("'test'", "golden-", "tpu-perturb-", "gpu-perturb-"):
        assert frag in msg


def test_run_open_bad_side_message(tmp_path):
    make(tmp_path, "x", "test", seed=1)
    with pytest.raises(StoreError, match="合法的写法以这几个前缀开头"):
        Run.open(tmp_path / "x")


def test_bad_side_with_html_exits_2_without_traceback(tmp_path, capsys):
    make(tmp_path, "g1", "golden-1", seed=1)
    make(tmp_path, "g2", "test", seed=2)            # 被测 side 写错
    make(tmp_path, "g3", "golden-3", seed=3)
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(catalog(tmp_path), allow_unicode=True))
    out = tmp_path / "out"
    assert main(["run", str(cp), "--out", str(out), "--html"]) == 2
    res = json.loads((out / "results.json").read_text())
    assert all(c["status"] == ERROR for c in res["checks"])
    assert "golden-" in capsys.readouterr().err
    html = (out / "report.html").read_text()
    assert "运行目录读不出来" in html


# ---------------------------------------------------------------- e：--render-only 看 --html
def test_render_only_respects_html_flag(tmp_path):
    for i in (1, 2, 3):
        make(tmp_path, f"g{i}", f"golden-{i}", seed=i)
    cp = tmp_path / "cat.yaml"
    cp.write_text(yaml.safe_dump(catalog(tmp_path), allow_unicode=True))
    out = tmp_path / "out"
    main(["run", str(cp), "--out", str(out)])
    assert not (out / "report.html").exists()
    assert main(["run", str(cp), "--out", str(out), "--render-only"]) == 0
    assert not (out / "report.html").exists()
    assert main(["run", str(cp), "--out", str(out), "--render-only", "--html"]) == 0
    assert (out / "report.html").exists()


# ---------------------------------------------------------------- d：data_fingerprint 缺整步
def test_data_fingerprint_missing_whole_steps_is_error():
    ref = {f"{s}.r0.m0.tokens": f"t{s}" for s in range(1, 11)}
    test = {k: v for k, v in ref.items() if int(k.split(".")[0]) <= 3}
    r = REGISTRY["data_fingerprint"]("d", "L0", test, ref)
    assert r.status == ERROR
    assert r.metrics["missing_steps"] == {"test": list(range(4, 11))}
    assert "被测缺整步 7 步：4–10" in r.message
    r = REGISTRY["data_fingerprint"]("d", "L0", ref, test)
    assert r.status == ERROR and "参照缺整步" in r.message
    # 同一步里少一个 rank 不算缺整步，仍按 token 不同报 fail
    partial = dict(ref)
    partial["5.r1.m0.tokens"] = "x"
    assert REGISTRY["data_fingerprint"]("d", "L0", partial, ref).status == FAIL
    assert REGISTRY["data_fingerprint"]("d", "L0", ref, dict(ref)).status == PASS


def test_data_fingerprint_missing_steps_through_runner(tmp_path):
    make_full(tmp_path, "a")
    make_full(tmp_path, "b")
    for f in (tmp_path / "b" / "data").glob("rank*.jsonl"):     # 被测只留前 3 步的记录
        f.write_text("".join(l + "\n" for l in f.read_text().splitlines() if json.loads(l)["step"] <= 3))
    cat = {"schema_version": 1, "model": "toy", "stage": "t", "runs": {"test": "b", "golden": ["a"], "root": str(tmp_path)},
           "checks": [c for c in full_checks() if c["id"] == "L0.data"]}
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(cat, allow_unicode=True))
    assert main(["run", str(cp), "--out", str(tmp_path / "o")]) == 2
    c = json.loads((tmp_path / "o" / "results.json").read_text())["checks"][0]
    assert c["status"] == ERROR and "4–10" in c["message"]


# ---------------------------------------------------------------- c：训练脚本取版本、image、side 提前校验
def test_pkg_version_submodule_of_namespace_package(tmp_path, monkeypatch):
    # 顶层是没有 __version__ 的命名空间包（同 megatron），版本在子包里
    (tmp_path / "nsjanus" / "core").mkdir(parents=True)
    (tmp_path / "nsjanus" / "core" / "__init__.py").write_text("__version__ = '0.20.0'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert getattr(__import__("nsjanus.core"), "__version__", None) is None     # 旧写法拿到的是顶层包
    assert pkg_version("nsjanus.core") == "0.20.0"
    import pytest as _pt
    assert pkg_version("no_such_module_janus", "pytest") == _pt.__version__     # import 不了时退回分发包元数据
    assert pkg_version("no_such_module_janus", "no-such-dist-janus") is None


def test_image_ref(monkeypatch):
    monkeypatch.delenv("JANUS_IMAGE", raising=False)
    img, warn = image_ref("repo:tag")
    assert img == "repo:tag" and "digest" in warn
    monkeypatch.setenv("JANUS_IMAGE", "repo@sha256:abc")
    assert image_ref("repo:tag") == ("repo@sha256:abc", None)


def test_train_script_rejects_bad_side_before_training(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    p = subprocess.run([sys.executable, str(GPU / "train_hy3.py"), "--hf", "x", "--out", "y",
                        "--janus-run", str(tmp_path / "r"), "--side", "test"], env=env, capture_output=True, text=True)
    assert p.returncode == 2, p.stderr        # 没有这道校验时会走到 import megatron（CPU 上没装，退出码 1）
    assert "golden-" in p.stderr


# ---------------------------------------------------------------- a、f：脚本路径可配、两种设置、RUN_DONE 带退出码
def _fake_bin(d: Path, name: str, body: str):
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_text("#!/bin/bash\n" + body)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def runsh(tmp_path):
    bin_ = tmp_path / "bin"
    _fake_bin(bin_, "torchrun", 'echo "PWD=$PWD"; echo "ARGS=$*"\n'
              'for k in NVTE_ALLOW_NONDETERMINISTIC_ALGO CUBLAS_WORKSPACE_CONFIG NCCL_ALGO; do echo "$k=${!k-UNSET}"; done\n'
              'exit ${FAKE_RC:-0}\n')
    (tmp_path / "env.sh").write_text("export FROM_ENV_SH=1\n")
    (tmp_path / "work").mkdir()

    def go(*args, **extra):
        env = {"PATH": f"{bin_}:{os.environ['PATH']}", "JANUS_ENV": str(tmp_path / "env.sh"),
               "JANUS_WORK": str(tmp_path / "work"), "JANUS_HF": "/hf", "JANUS_RUNS": str(tmp_path / "runs"),
               "JANUS_TLOGS": str(tmp_path / "tlogs"),
               # 调用方环境里已经设了三个变量：默认设置要把它们清掉
               "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_ALGO": "Ring", **extra}
        p = subprocess.run(["bash", str(GPU / "run.sh"), *args], env=env, capture_output=True, text=True)
        kv = dict(l.split("=", 1) for l in p.stdout.splitlines() if "=" in l and not l.startswith("RUN_DONE"))
        return p, kv
    return go


def test_run_sh_default_setting_unsets_env(runsh, tmp_path):
    p, kv = runsh("n1", "3", "--load", "/m")
    assert p.returncode == 0
    assert kv["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == kv["CUBLAS_WORKSPACE_CONFIG"] == kv["NCCL_ALGO"] == "UNSET"
    assert kv["PWD"] == str(tmp_path / "work")
    assert f"--hf /hf --out {tmp_path}/runs/n1 --steps 3 --load /m" in kv["ARGS"]
    assert p.stdout.splitlines()[-1] == "RUN_DONE n1 rc=0"


def test_run_sh_deterministic_sets_env(runsh):
    _, kv = runsh("n1", "3", "--deterministic")
    assert (kv["NVTE_ALLOW_NONDETERMINISTIC_ALGO"], kv["CUBLAS_WORKSPACE_CONFIG"], kv["NCCL_ALGO"]) == ("0", ":4096:8", "Ring")
    _, kv = runsh("n1", "3", "--deterministic", JANUS_DET_ENV="0")
    assert kv["NCCL_ALGO"] == "UNSET"
    _, kv = runsh("n1", "3", JANUS_DET_ENV="1")
    assert kv["NCCL_ALGO"] == "Ring"
    p, _ = runsh("n1", "3", JANUS_DET_ENV="yes")
    assert p.returncode == 2


def test_run_sh_propagates_exit_code(runsh):
    p, _ = runsh("n2", "3", FAKE_RC="3")
    assert p.returncode == 3
    assert p.stdout.splitlines()[-1] == "RUN_DONE n2 rc=3"


def test_run_sh_usage_and_defaults():
    p = subprocess.run(["bash", str(GPU / "run.sh")], capture_output=True, text=True)
    assert p.returncode == 2 and "用法" in p.stderr
    s = (GPU / "run.sh").read_text()
    for d in ("/data/env.sh", "/data/jw", "/data/hy3-4l", "/data/runs", "/data/tlogs"):
        assert d in s                          # 默认值保持第一阶段的布局


def test_env_sh_paths_configurable():
    sh = f"source {GPU / 'env.sh'}; echo $PYTHONPATH; echo $TRITON_CACHE_DIR"
    p = subprocess.run(["bash", "-c", sh], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
    assert p.stdout.split() == ["/data/janus-src", "/data/triton_cache"]
    p = subprocess.run(["bash", "-c", sh], env={"PATH": os.environ["PATH"], "JANUS_SRC": "/s", "JANUS_TRITON_CACHE": "/t",
                                                 "PYTHONPATH": "/old"}, capture_output=True, text=True)
    assert p.stdout.split() == ["/s:/old", "/t"]


def test_prep_data_creates_dir_and_stops_on_error(tmp_path):
    bin_ = tmp_path / "bin"
    _fake_bin(bin_, "curl", "exit 22\n")
    (tmp_path / "env.sh").write_text("")
    ds = tmp_path / "ds"
    p = subprocess.run(["bash", str(GPU / "prep_data.sh")], capture_output=True, text=True,
                       env={"PATH": f"{bin_}:{os.environ['PATH']}", "JANUS_ENV": str(tmp_path / "env.sh"), "JANUS_DS": str(ds)})
    assert ds.is_dir()
    assert p.returncode == 1 and "DL_FAIL" in p.stdout and "PREP_DONE" not in p.stdout
    s = (GPU / "prep_data.sh").read_text()
    assert "set -eo pipefail" in s
