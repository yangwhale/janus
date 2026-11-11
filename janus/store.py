"""运行目录的读写。格式见 docs/reference/dump-format.md（schema_version 1）。

safetensors 用自己的实现读写（格式：8 字节小端头长度 + JSON 头 + 原始字节），
不依赖 torch，BF16 保持原始位模式。
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from janus.dtypes import NP_TO_ST, ST_TO_NP

SCHEMA_VERSION = 1
SIDES_PREFIX = ("golden-", "tpu", "tpu-perturb-", "gpu-perturb-")
META_REQUIRED = ("schema_version", "run_name", "side", "framework", "model", "code", "image",
                 "versions", "hardware", "parallelism", "config", "checkpoint", "data", "steps",
                 "determinism", "dtypes", "created_at")


class StoreError(Exception):
    """运行目录不合规：缺文件、缺字段、校验和不符等。比较时一律报 error。"""


def side_error(side) -> str | None:
    """side 合法返回 None，不合法返回可读的提示（列出合法前缀）。训练脚本启动时也用它提前校验。"""
    if isinstance(side, str) and side.startswith(SIDES_PREFIX):
        return None
    return (f"side 不合法: {side!r}；合法的写法以这几个前缀开头：{'、'.join(SIDES_PREFIX)}"
            f"（如 golden-1、tpu、tpu-perturb-1、gpu-perturb-1，见 docs/reference/dump-format.md）")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- safetensors

def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> dict:
    """写一个 safetensors 文件。返回 {名字: {dtype, shape, sha256}}。"""
    header, blobs, info, off = {}, [], {}, 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        if arr.dtype not in NP_TO_ST:
            raise StoreError(f"不支持的 dtype {arr.dtype}（{name}）")
        raw = arr.tobytes()
        header[name] = {"dtype": NP_TO_ST[arr.dtype], "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        info[name] = {"dtype": NP_TO_ST[arr.dtype], "shape": list(arr.shape), "sha256": sha256_bytes(raw)}
        blobs.append(raw)
        off += len(raw)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)          # 头部按 8 字节对齐
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for raw in blobs:
            f.write(raw)
    return info


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def read_safetensors_tensor(path: Path, name: str) -> tuple[np.ndarray, bytes]:
    """读一个张量，返回 (数组, 原始字节)。"""
    header, base = read_safetensors_header(path)
    if name not in header:
        raise StoreError(f"{path} 里没有张量 {name}")
    ent = header[name]
    if ent["dtype"] not in ST_TO_NP:
        raise StoreError(f"不支持的 dtype {ent['dtype']}（{name}）")
    a, b = ent["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    arr = np.frombuffer(raw, dtype=ST_TO_NP[ent["dtype"]]).reshape(ent["shape"])
    return arr, raw


# ---------------------------------------------------------------- metrics.jsonl

def _decode_num(v):
    if isinstance(v, str) and v in ("nan", "inf", "-inf"):
        return float(v)
    return v


def _encode_num(v):
    if isinstance(v, float) and not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


# ---------------------------------------------------------------- 读

@dataclass
class TensorEntry:
    name: str
    kind: str
    point: str
    file: str
    dtype: str
    shape: list
    layout: str
    sha256: str
    shard: dict | None = None


@dataclass
class Run:
    """一个运行目录。读入时校验 DONE 与 meta.json；张量按需读取并核对 sha256。"""
    path: Path
    meta: dict = field(default_factory=dict)

    @classmethod
    def open(cls, path) -> Run:
        p = Path(path)
        if not (p / "DONE").exists():
            raise StoreError(f"{p} 没有 DONE，视为未完成")
        meta_path = p / "meta.json"
        if not meta_path.exists():
            raise StoreError(f"{p} 缺 meta.json")
        meta = json.loads(meta_path.read_text())
        missing = [k for k in META_REQUIRED if k not in meta]
        if missing:
            raise StoreError(f"meta.json 缺字段: {missing}")
        if meta["schema_version"] != SCHEMA_VERSION:
            raise StoreError(f"schema_version {meta['schema_version']} 不兼容（需要 {SCHEMA_VERSION}）")
        err = side_error(meta["side"])
        if err:
            raise StoreError(f"{meta_path}: {err}")
        return cls(p, meta)

    @property
    def name(self) -> str:
        return self.meta["run_name"]

    def meta_sha256(self) -> str:
        return sha256_bytes((self.path / "meta.json").read_bytes())

    # ---- 逐步标量
    def metrics(self) -> dict[int, dict]:
        f = self.path / "metrics.jsonl"
        if not f.exists():
            raise StoreError(f"{self.path} 缺 metrics.jsonl")
        out, last = {}, None
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            rec = {k: _decode_num(v) for k, v in json.loads(line).items()}
            if "step" not in rec:
                raise StoreError(f"metrics.jsonl 第 {i} 行缺 step")
            s = int(rec["step"])
            if last is not None and s <= last:
                raise StoreError(f"metrics.jsonl 步号不递增（第 {i} 行 step={s}）")
            out[s], last = rec, s
        return out

    def raw_steps(self) -> list[int]:
        """metrics.jsonl 里按文件顺序的步号，不做递增校验（run_complete 用它数重复与缺步：重启后接着追加写的
        运行会出现重复步，metrics() 读到会直接报错，这里要能数出来）。"""
        f = self.path / "metrics.jsonl"
        if not f.exists():
            raise StoreError(f"{self.path} 缺 metrics.jsonl")
        return [int(json.loads(line)["step"]) for line in f.read_text().splitlines() if line.strip()]

    # 派生序列：记录里没有这一列时，由同一行的其它列算出来（只读已记录的量，不改原始数据）
    #   param_norm_from_groups = sqrt(Σ group_sq.*)：分组平方和（每组在设备上 FP32 求和，组间在主机上 FP64 相加）合成的参数范数。
    #   TPU 的 param_norm 是整网一次 FP32 归约，单步测量误差到 1e-7 量级（混元 3 案例 01「param_norm 的测量精度」行）
    DERIVED = {"param_norm_from_groups"}

    def series(self, field_name: str) -> dict[int, float]:
        rows = self.metrics().items()
        if field_name in self.DERIVED and not any(field_name in r for _, r in rows):
            out = {}
            for s, r in rows:
                g = [float(v) for k, v in r.items() if k.startswith("group_sq.")]
                if g:
                    out[s] = math.sqrt(math.fsum(g))
            return out
        return {s: float(r[field_name]) for s, r in rows if field_name in r}

    # ---- 输入指纹与参数清单（GPU 抓数 v2 写在 data/、params_sha/ 下，每个 rank 一份）
    def data_records(self) -> dict[str, dict]:
        """{"<步>.r<rank>.m<microbatch>": {tokens, labels, loss_mask, position_ids 的 sha256, nonpad}}。"""
        out = {}
        for f in sorted((self.path / "data").glob("rank*.jsonl")):
            r = f.stem[4:]
            for line in f.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[f"{rec['step']}.r{r}.m{rec['micro']}"] = rec
        if not out:
            raise StoreError(f"{self.path} 没有输入指纹（data/rank*.jsonl）")
        return out

    def param_manifest(self, tag: str) -> dict[str, dict[str, dict]]:
        """{rank: {参数名: {sha256, shape, dtype}}}，tag = init | final。"""
        out = {}
        for f in sorted((self.path / "params_sha").glob(f"rank*.{tag}.json")):
            out[f.name.split(".")[0][4:]] = json.loads(f.read_text())
        if not out:
            raise StoreError(f"{self.path} 没有参数清单（params_sha/rank*.{tag}.json）")
        return out

    def state_manifest(self, tag: str) -> dict[str, dict[str, dict]]:
        """训练状态的 FP32 全量清单 state_sha/<tag>.json（TPU 侧，janus/dump/maxtext.py 的 write_state_sha），
        返回和 param_manifest 同样的形状 {分组: {状态路径: {sha256, shape, dtype, kind}}}，分组只有一个 "state"。"""
        f = self.path / "state_sha" / f"{tag}.json"
        if not f.exists():
            raise StoreError(f"{self.path} 没有状态清单（state_sha/{tag}.json）")
        return {"state": json.loads(f.read_text())}

    def manifest(self, source: str, tag: str) -> dict[str, dict[str, dict]]:
        """source = params（params_sha，按 rank 分组，BF16 参数）| state（state_sha，FP32 全量状态）。"""
        if source == "params":
            return self.param_manifest(tag)
        if source == "state":
            return self.state_manifest(tag)
        raise StoreError(f"清单来源只能是 params 或 state：{source}")

    # ---- 张量
    def index(self, step: int) -> list[TensorEntry]:
        ic = getattr(self, "_icache", None)
        if ic is not None and step in ic:
            return ic[step]
        out = self._read_index(step)
        if ic is not None:
            ic[step] = out
        return out

    def _read_index(self, step: int) -> list[TensorEntry]:
        f = self.path / f"step_{step}" / "index.json"
        if not f.exists():
            raise StoreError(f"{self.path} 第 {step} 步没有 index.json")
        idx = json.loads(f.read_text())
        ents = [TensorEntry(**t) for t in idx["tensors"]]
        seen, dup = set(), set()
        for e in ents:
            (dup if e.name in seen else seen).add(e.name)
        if dup:
            # 同名张量出现两次：不知道该信哪一份（常见于钩子重复触发、两个 scope 写了同一个名字），不静默取第一个
            raise StoreError(f"{f} 里张量重名 {len(dup)} 个：{sorted(dup)[:5]}")
        return ents

    def names(self, step: int, pattern: str = "*") -> list[str]:
        return sorted(t.name for t in self.index(step) if fnmatch.fnmatchcase(t.name, pattern))

    def enable_cache(self):
        """同一次 janus run 里反复读同一张量时只读、只校验一次（数组常驻内存）。"""
        self._cache, self._icache = {}, {}

    def tensor(self, step: int, name: str) -> np.ndarray:
        cache = getattr(self, "_cache", None)
        if cache is not None and (step, name) in cache:
            return cache[(step, name)]
        arr = self._read_tensor(step, name)
        if cache is not None:
            cache[(step, name)] = arr
        return arr

    def _read_tensor(self, step: int, name: str) -> np.ndarray:
        ents = [t for t in self.index(step) if t.name == name]
        if not ents:
            raise StoreError(f"第 {step} 步没有张量 {name}")
        e = ents[0]                            # index() 读入时已保证不重名
        if e.layout != "global":
            raise StoreError(f"{name} 是分片存的；按分片读取尚未实现")
        arr, raw = read_safetensors_tensor(self.path / f"step_{step}" / e.file, name)
        if sha256_bytes(raw) != e.sha256:
            raise StoreError(f"{name} 的 sha256 与 index.json 不符")
        if list(arr.shape) != list(e.shape) or e.dtype not in (str(arr.dtype), _st_alias(arr.dtype)):
            raise StoreError(f"{name} 的形状或 dtype 与 index.json 不符")
        return arr

    # ---- 步号换算
    def to_golden_step(self, step: int) -> int:
        """按 meta.steps.convention 把本侧步号换成 golden 步号。约定写成 'golden_step = step + k'。"""
        conv = self.meta["steps"].get("convention", "golden_step = step")
        lhs, _, rhs = conv.partition("=")
        if lhs.strip() != "golden_step":
            raise StoreError(f"不认识的步号约定: {conv}")
        rhs = rhs.replace(" ", "")
        if rhs == "step":
            return step
        if rhs.startswith("step+") or rhs.startswith("step-"):
            return step + int(rhs[4:])
        raise StoreError(f"不认识的步号约定: {conv}")


def _st_alias(dtype) -> str:
    """index.json 的 dtype 可写 numpy 名（float32、bfloat16）或 safetensors 名（F32、BF16）。"""
    return NP_TO_ST.get(np.dtype(dtype), "")


# ---------------------------------------------------------------- 写

class RunWriter:
    """按规范写运行目录。写完调 done()；之后目录只读。"""

    def __init__(self, path, meta: dict):
        self.path = Path(path)
        if (self.path / "DONE").exists():
            raise StoreError(f"{self.path} 已完成，不能覆盖")
        missing = [k for k in META_REQUIRED if k not in meta]
        if missing:
            raise StoreError(f"meta 缺字段: {missing}")
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        self._metrics = open(self.path / "metrics.jsonl", "a")
        self._index: dict[int, list] = {}

    def log_metrics(self, step: int, **values):
        def enc(v):  # 标量，或标量组成的列表（如每个专家一个值）
            return [_encode_num(float(x)) for x in v] if isinstance(v, (list, tuple)) else _encode_num(float(v))
        rec = {"step": int(step), **{k: enc(v) for k, v in values.items()}}
        # repr 精度：json 对 float 用最短往返表示，不截断
        self._metrics.write(json.dumps(rec) + "\n")
        self._metrics.flush()

    def write_tensors(self, step: int, scope: str, tensors: dict[str, np.ndarray],
                      kind: str, point: str):
        d = self.path / f"step_{step}"
        d.mkdir(exist_ok=True)
        fname = f"{scope}.safetensors"
        if (d / fname).exists():
            raise StoreError(f"{d / fname} 已存在")
        have = {e["name"] for e in self._index.get(step, [])}
        dup = sorted(set(tensors) & have)
        if dup:
            raise StoreError(f"第 {step} 步张量重名（已写在别的 scope 里）：{dup[:5]}")
        info = write_safetensors(d / fname, tensors)
        for name, i in info.items():
            self._index.setdefault(step, []).append(
                {"name": name, "kind": kind, "point": point, "file": fname, "dtype": i["dtype"],
                 "shape": i["shape"], "layout": "global", "sha256": i["sha256"]})

    def done(self):
        self._metrics.close()
        for step, ents in self._index.items():
            (self.path / f"step_{step}" / "index.json").write_text(json.dumps(
                {"schema_version": SCHEMA_VERSION, "step": step,
                 "tensors": sorted(ents, key=lambda e: e["name"])}, indent=1))
        (self.path / "DONE").write_text("")
        os.sync() if hasattr(os, "sync") else None
