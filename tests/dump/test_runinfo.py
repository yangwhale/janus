"""dump/runinfo.py：meta.json 里的版本号与镜像（训练入口用，不依赖 torch）。"""

import numpy as np

from janus.dump import runinfo


def test_pkg_version():
    assert runinfo.pkg_version("numpy") == np.__version__
    assert runinfo.pkg_version("no_such_module_xyz") is None
    assert runinfo.pkg_version("no_such_module_xyz", dist="numpy") == np.__version__  # 退回分发包元数据


def test_image_ref(monkeypatch):
    monkeypatch.delenv("JANUS_IMAGE", raising=False)
    img, warn = runinfo.image_ref("repo/img@sha256:abc")
    assert img == "repo/img@sha256:abc" and warn is None
    monkeypatch.setenv("JANUS_IMAGE", "repo/img:latest")
    img, warn = runinfo.image_ref("repo/img@sha256:abc")
    assert img == "repo/img:latest" and "digest" in warn  # 环境变量优先；不带 digest 要提示
