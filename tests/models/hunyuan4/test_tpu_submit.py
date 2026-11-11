"""混元 4 TPU 侧的提交脚本 run.sh：SCRIPT 的三种取值（训练、逐算子回放、本目录下其余检查脚本）。

只用 DRYRUN=1（打印 JobSet、不碰集群），不要 kubectl。"""

import os
import subprocess

import yaml
from helpers import ROOT  # noqa: E402

MT = ROOT / "models" / "hunyuan4" / "frameworks" / "maxtext"
BASE_ENV = {
    "BUCKET": "gs://<POC 桶>",
    "GOLDEN": "gs://<POC 桶>/janus/hunyuan4/golden-t1/g-x",
    "STAGE": "mini",
    "PLATFORM": "v5p",
    "CONTEXT": "ctx",
    "NODEPOOL": "pool-a",
    "MAXTEXT_PKG": "gs://<POC 桶>/janus/pkg/maxtext.tgz",
    "MAXTEXT_PKG_SHA256": "0" * 64,
    "MAXTEXT_COMMIT": "1" * 40,
    "JANUS_PKG": "gs://<POC 桶>/janus/pkg/janus.tgz",
    "JANUS_PKG_SHA256": "2" * 64,
    "JANUS_COMMIT": "3" * 40,
    "DRYRUN": "1",
}


def dryrun(*args, **extra):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), **BASE_ENV, **extra}
    return subprocess.run(["bash", str(MT / "run.sh"), "t-x", "1", *args], env=env, capture_output=True, text=True)


def command(p):
    js = yaml.safe_load(p.stdout)
    return js["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_train_default_unchanged():
    p = dryrun("--dump-steps", "none")
    assert p.returncode == 0, p.stderr
    c = command(p)
    assert "train_hy4.py" in c and "--steps 1 --janus-run /data/runs/t-x" in c and "--dump-steps none" in c


def test_op_replay_unchanged():
    p = dryrun(SCRIPT="op_replay.py")
    assert p.returncode == 0, p.stderr
    c = command(p)
    assert "op_replay.py" in c and "--out /data/runs/t-x/op_replay.json" in c and "step_1/fwd.safetensors" in c


def test_generic_script():
    p = dryrun("--inputs", "gs://<POC 桶>/x.npz", SCRIPT="elementwise_ulp.py")
    assert p.returncode == 0, p.stderr
    c = command(p)
    assert "elementwise_ulp.py" in c and "--out /data/runs/t-x/elementwise_ulp.json" in c
    assert "--inputs gs://<POC 桶>/x.npz" in c and "mkdir -p /data/runs/t-x" not in c  # 运行目录只由进程 0 建


def test_generic_script_must_exist():
    p = dryrun(SCRIPT="nope.py")
    assert p.returncode != 0 and "nope.py" in p.stdout
    p = dryrun(SCRIPT="run.sh")
    assert p.returncode != 0
