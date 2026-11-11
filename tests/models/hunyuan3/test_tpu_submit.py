"""混元 3 TPU 侧的提交脚本 run.sh：默认提交方式（自己的节点池）与站点配置钩子（JANUS_SITE_PROFILE）。

只用 DRYRUN=1（打印 JobSet、不碰集群），不要 kubectl。"""

import os
import subprocess

import yaml
from helpers import ROOT  # noqa: E402

MT = ROOT / "models" / "hunyuan3" / "frameworks" / "maxtext"
BASE_ENV = {
    "BUCKET": "gs://<POC 桶>",
    "MAXTEXT_PKG": "gs://<POC 桶>/janus/pkg/maxtext.tgz",
    "MAXTEXT_PKG_SHA256": "0" * 64,
    "MAXTEXT_COMMIT": "1" * 40,
    "JANUS_PKG": "gs://<POC 桶>/janus/pkg/janus.tgz",
    "JANUS_PKG_SHA256": "2" * 64,
    "JANUS_COMMIT": "3" * 40,
    "DRYRUN": "1",
}


def dryrun(**extra):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), **BASE_ENV, **extra}
    return subprocess.run(
        ["bash", str(MT / "run.sh"), "t-x", "30", "--no-hooks"], env=env, capture_output=True, text=True
    )


def pod_spec(js):
    return js["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]


def test_default_nodepool():
    p = dryrun(PLATFORM="v5p", CONTEXT="ctx", NODEPOOL="pool-a")
    assert p.returncode == 0, p.stderr
    js = yaml.safe_load(p.stdout)
    assert js["metadata"] == {"name": "janus-t-x", "namespace": "default"}
    sel = pod_spec(js)["spec"]["nodeSelector"]
    assert sel["cloud.google.com/gke-nodepool"] == "pool-a"
    assert sel["cloud.google.com/gke-tpu-topology"] == "2x2x2"


def test_default_requires_nodepool():
    p = dryrun(PLATFORM="v5p", CONTEXT="ctx")
    assert p.returncode != 0 and "NODEPOOL" in p.stderr


def test_site_profile_hook():
    """示例站点配置被 source：它给的 labels / annotations、nodeSelector、tolerations 进 JobSet，默认的节点池选择不再出现。"""
    p = dryrun(PLATFORM="v7x", CONTEXT="ctx", JANUS_SITE_PROFILE=str(MT / "site-profile.example.sh"))
    assert p.returncode == 0, p.stderr
    js = yaml.safe_load(p.stdout)
    assert js["metadata"]["labels"] == {"example.com/queue": "<队列名>"}
    assert js["metadata"]["annotations"] == {"example.com/owner": "<提交人>"}
    spec = pod_spec(js)["spec"]
    assert "cloud.google.com/gke-nodepool" not in spec["nodeSelector"]
    assert spec["nodeSelector"]["example.com/capacity-pool"] == "<容量池>"
    assert {"key": "example.com/capacity-pool", "operator": "Exists", "effect": "NoSchedule"} in spec["tolerations"]


def test_site_profile_missing_file(tmp_path):
    p = dryrun(PLATFORM="v5p", CONTEXT="ctx", NODEPOOL="pool-a", JANUS_SITE_PROFILE=str(tmp_path / "nope.sh"))
    assert p.returncode == 1 and "JANUS_SITE_PROFILE" in p.stdout
