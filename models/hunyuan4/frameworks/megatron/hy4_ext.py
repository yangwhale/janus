"""混元 4 Megatron 代码的扩展点：模型变体与训练开关可以写在独立的扩展模块里，共用代码不依赖它们。

共用代码（`hy4_mcore.py`、`hy4_weights.py`、`hy4_capture.py`、`train_hy4mg.py`、`step1_ref.py`、`layer_ref.py`、
`export_hf_grads.py`、`equiv_hf.py`）在固定的几处按名字调扩展点；同目录下文件名是 `hy4_ext_*.py` 的模块在第一次用到
扩展点时按文件名顺序导入，导入时用 `register` 登记自己的实现。没有扩展模块时每个扩展点都是空操作，共用代码走自己的路。

扩展模块的约定：
- 顶层只能导入标准库与本模块（`hy4_mcore` 导入时就会触发加载，扩展模块要用的其余模块在函数里导入）；
- `config_fields` 返回要加进 `hy4_mcore.Hy4Fields` 的字段 `(名字, 类型, 默认值)`，默认值要不可变；
- 其余扩展点的参数与返回值见共用代码里调用处的注释；`first` 取第一个不是 None 的返回值，`call` 依次全调，
  `collect` 把各实现返回的列表 / 字典合起来。
"""

from __future__ import annotations

import dataclasses
import glob
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_HOOKS: dict[str, list] = {}
_STATE = {"loaded": False}


def register(name: str, fn=None):
    """登记扩展点 name 的一个实现；可当装饰器用。"""

    def put(f):
        _HOOKS.setdefault(name, []).append(f)
        return f

    return put if fn is None else put(fn)


def load() -> list[str]:
    """导入同目录下的 `hy4_ext_*.py`（只导入一次），返回导入了的模块名。"""
    if _STATE["loaded"]:
        return _STATE["modules"]
    _STATE["loaded"] = True
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    mods = []
    for path in sorted(glob.glob(os.path.join(HERE, "hy4_ext_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        importlib.import_module(name)
        mods.append(name)
    _STATE["modules"] = mods
    return mods


def hooks(name: str) -> list:
    load()
    return list(_HOOKS.get(name, ()))


def call(name: str, *args, **kwargs) -> list:
    """依次调 name 的全部实现，返回各自的返回值。"""
    return [f(*args, **kwargs) for f in hooks(name)]


def first(name: str, *args, **kwargs):
    """依次调 name 的实现，返回第一个不是 None 的返回值；都是 None（或没有实现）返回 None。"""
    for f in hooks(name):
        r = f(*args, **kwargs)
        if r is not None:
            return r
    return None


def collect(name: str, *args, **kwargs):
    """各实现返回列表（或元组）就按顺序接起来，返回字典就按顺序合并；没有实现返回空列表。"""
    out = None
    for r in call(name, *args, **kwargs):
        if r is None:
            continue
        if isinstance(r, dict):
            out = {**(out or {}), **r}
        else:
            out = [*(out or []), *r]
    return [] if out is None else out


def extend_dataclass(cls):
    """cls（dataclass，字段都有默认值）加上扩展登记的字段（`config_fields`），返回同名的子类；没有扩展字段时原样返回。"""
    extra = collect("config_fields")
    if not extra:
        return cls
    fields = [(n, t, dataclasses.field(default=d)) for n, t, d in extra]
    return dataclasses.make_dataclass(
        cls.__name__,
        fields,
        bases=(cls,),
        namespace={"__doc__": cls.__doc__, "__module__": cls.__module__, "__qualname__": cls.__qualname__},
    )
