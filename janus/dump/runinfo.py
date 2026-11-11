"""训练入口写 meta.json 时用的环境信息（版本号、镜像）。不依赖 torch，便于单测。"""
from __future__ import annotations

import importlib
import importlib.metadata
import os

DIGEST_MARK = "@sha256:"


def pkg_version(module: str, dist: str | None = None) -> str | None:
    """包版本：先取 import 出来的那个模块自己的 __version__，再退回已安装分发包的元数据，都拿不到返回 None。

    注意要用 importlib.import_module：__import__("a.b") 返回的是顶层包 a，不是 a.b。
    Megatron 的顶层 megatron 是命名空间包、没有 __version__，所以那样取 megatron.core 恒为 None。"""
    try:
        v = getattr(importlib.import_module(module), "__version__", None)
        if v:
            return str(v)
    except Exception:  # noqa: BLE001  装没装、能不能 import 都只影响这一个字段
        pass
    if dist:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def image_ref(default: str) -> tuple[str, str | None]:
    """meta.json 的 image：取环境变量 JANUS_IMAGE，没设用 default。
    规格要求带 @sha256: digest；不带时照样返回，另给一句提示（容器里拿不到 digest，要在起 Pod 时传进来）。"""
    img = os.environ.get("JANUS_IMAGE") or default
    if DIGEST_MARK in img:
        return img, None
    return img, (f"meta.json 的 image 是 {img!r}，不带 {DIGEST_MARK} digest（docs/reference/dump-format.md 要求带）；"
                 "起 Pod 时把 JANUS_IMAGE 设成 <仓库>@sha256:<digest>，digest 用 "
                 "kubectl get pod <名> -o jsonpath='{.status.containerStatuses[0].imageID}' 查")
