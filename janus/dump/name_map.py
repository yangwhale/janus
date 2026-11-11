"""模型的名字与布局映射（models/<模型>/model/name_map.yaml）的解释器。只依赖 numpy，可单测。

一张映射表把同一个参数在三处的名字和布局连起来：HF 检查点、Megatron（GPU）、MaxText（TPU）。用途：
- HF → MaxText：转换检查点（hf_to_maxtext_pieces）；
- MaxText → HF：反向取回每个 HF 张量，逐张量对 sha256（maxtext_to_hf）；
- HF / MaxText → Megatron：拼成 Megatron 布局（qkv 按 KV 组交错、gate|up 拼接、专家按 EP rank 分本地号），
  TPU 抓数用它把状态写成和 golden 同名、同布局的张量。

全部变换只做转置、改形状、切片、拼接，不做任何算术，所以来回逐位不变。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


class NameMapError(Exception):
    pass


@dataclass
class Piece:
    """一个 HF 张量落到 MaxText 叶子的一片。"""
    hf: str
    path: str                      # 例如 params/decoder/moe_layers/self_attention/query/kernel
    index: tuple                   # ((轴, 下标), …)，在完整叶子上取这一片
    ops: list                      # [("T", perm) | ("reshape", shape)]

    @property
    def collection(self) -> str:
        return self.path.split("/", 1)[0]


@dataclass
class ParamEntry:
    canonical: str
    megatron: str                  # 专家参数是持有它的 EP rank 上的本地名（weight<k>，k = e - 24·r）
    hf: list
    to_megatron: str
    pieces: list
    layer: int | None = None
    expert: int | None = None
    buffer: bool = False
    status: str = ""
    note: str = ""
    megatron_template: str = field(default="", repr=False)


def _fmt(tpl: str, **kw) -> str:
    return tpl.format(**kw) if tpl else tpl


def _eval_index(expr, env: dict) -> int:
    if isinstance(expr, int):
        return expr
    s = str(expr).format(**env)
    if not re.fullmatch(r"[0-9+\-* ()]+", s):
        raise NameMapError(f"下标表达式不合法：{expr!r} → {s!r}")
    return int(eval(s, {"__builtins__": {}}, {}))  # noqa: S307  只含数字与 + - * 的表达式


def _parse_ops(ops, dims: dict | None = None) -> list:
    dims = dims or {}

    def dim(x):
        return int(dims[x]) if isinstance(x, str) else int(x)
    out = []
    for op in ops or []:
        if not isinstance(op, dict) or len(op) != 1:
            raise NameMapError(f"op 每项是单键字典：{op!r}")
        (k, v), = op.items()
        if k == "T":
            out.append(("T", tuple(int(x) for x in v)))
        elif k == "reshape":
            out.append(("reshape", tuple(dim(x) for x in v)))
        else:
            raise NameMapError(f"不认识的 op：{k}")
    return out


def apply_ops(arr: np.ndarray, ops) -> np.ndarray:
    for kind, arg in ops:
        arr = np.transpose(arr, arg) if kind == "T" else arr.reshape(arg)
    return np.ascontiguousarray(arr)


def invert_ops(arr: np.ndarray, ops, src_shape) -> np.ndarray:
    """apply_ops 的逆：src_shape 是正向输入（HF 张量）的形状。"""
    shapes = [tuple(src_shape)]
    for kind, arg in ops:                                # 先推出每一步之前的形状
        s = shapes[-1]
        shapes.append(tuple(s[i] for i in arg) if kind == "T" else tuple(arg))
    for (kind, arg), before in zip(reversed(ops), reversed(shapes[:-1])):
        arr = np.transpose(arr, np.argsort(arg)) if kind == "T" else arr.reshape(before)
    return np.ascontiguousarray(arr)


def forward_shape(src_shape, ops) -> tuple:
    s = tuple(src_shape)
    for kind, arg in ops:
        s = tuple(s[i] for i in arg) if kind == "T" else tuple(arg)
    return s


def leaf_index(index, ndim: int) -> tuple:
    """((轴, 下标), …) → numpy 下标元组（其余轴取全部）。"""
    idx = [slice(None)] * ndim
    for ax, i in index:
        idx[ax] = i
    return tuple(idx)


class NameMap:
    def __init__(self, spec: dict):
        if spec.get("schema_version") != 1:
            raise NameMapError("name_map schema_version 必须是 1")
        self.spec = spec
        self.dims = spec["dims"]
        self.ep = int(spec.get("megatron", {}).get("ep", 1))
        self.local_experts = int(spec.get("megatron", {}).get("local_experts", self.dims.get("experts", 0)))
        stacks = spec.get("maxtext", {}).get("stacks", {})
        self._stack_of: dict[int, tuple[str, str]] = {}
        for name, st in stacks.items():
            for i in st["layers"]:
                self._stack_of[int(i)] = (name, st["j"])
        self.entries = self._expand()

    @classmethod
    def load(cls, path) -> NameMap:
        return cls(yaml.safe_load(Path(path).read_text()))

    # ---- 展开
    def _layers(self, which: str) -> list[int]:
        n, fd = int(self.dims["layers"]), int(self.dims.get("first_dense", 0))
        return {"all": list(range(n)), "dense": list(range(fd)), "moe": list(range(fd, n))}[which]

    def _expand(self) -> list[ParamEntry]:
        out = []
        for p in self.spec["params"]:
            layers = self._layers(p["layers"]) if "layers" in p else [None]
            experts = range(int(self.dims["experts"])) if p.get("experts") else [None]
            for i in layers:
                for e in experts:
                    env = {}
                    if i is not None:
                        stack, jexpr = self._stack_of[i]
                        env.update(i=i, stack=stack, j=_eval_index(jexpr, {"i": i}))
                    if e is not None:
                        r = e // self.local_experts
                        env.update(e=e, r=r, k=e - r * self.local_experts)
                    pieces = []
                    for m in p.get("maxtext") or []:
                        idx = tuple((int(ax), _eval_index(v, env)) for ax, v in m.get("index", []))
                        pieces.append(Piece(hf=_fmt(m["hf"], **env), path=_fmt(m["path"], **env), index=idx,
                                            ops=_parse_ops(m.get("op"), self.dims)))
                    ent = ParamEntry(
                        canonical=_fmt(p["canonical"], **env), megatron=_fmt(p["megatron"], **env),
                        hf=[_fmt(h, **env) for h in p["hf"]], to_megatron=p.get("to_megatron", "identity"),
                        pieces=pieces, layer=i, expert=e, buffer=bool(p.get("buffer")),
                        status=p.get("status", ""), note=p.get("note", ""), megatron_template=p["megatron"])
                    hf_in_pieces = [pc.hf for pc in pieces]
                    if sorted(hf_in_pieces) != sorted(ent.hf):
                        raise NameMapError(f"{ent.canonical}：maxtext 各片的 hf {hf_in_pieces} 与 hf {ent.hf} 不一致")
                    out.append(ent)
        names = [e.canonical for e in out]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise NameMapError(f"规范名重复：{sorted(dup)[:5]}")
        return out

    # ---- 查询
    def hf_names(self) -> set[str]:
        return {h for e in self.entries for h in e.hf}

    def by_canonical(self, name: str) -> ParamEntry:
        for e in self.entries:
            if e.canonical == name:
                return e
        raise KeyError(name)

    def megatron_name(self, ent: ParamEntry, rank: int | None = None) -> str:
        """Megatron 名。专家参数按 EP rank 换本地号：rank 省略时取持有它的那个 rank。"""
        if ent.expert is None:
            return ent.megatron
        owner = ent.expert // self.local_experts
        if rank is not None and rank != owner:
            raise NameMapError(f"专家 {ent.expert} 不在 rank {rank} 上（在 rank {owner}）")
        env = {"i": ent.layer, "e": ent.expert, "k": ent.expert - owner * self.local_experts, "r": owner}
        return ent.megatron_template.format(**env)

    def expert_owner(self, ent: ParamEntry) -> int | None:
        return None if ent.expert is None else ent.expert // self.local_experts

    def maxtext_leaves(self) -> dict[str, list[Piece]]:
        out: dict[str, list[Piece]] = {}
        for e in self.entries:
            for pc in e.pieces:
                out.setdefault(pc.path, []).append(pc)
        return out

    # ---- 变换
    def to_megatron(self, ent: ParamEntry, hf_arrays: list[np.ndarray]) -> np.ndarray:
        kind = ent.to_megatron
        if kind == "identity":
            if len(hf_arrays) != 1:
                raise NameMapError(f"{ent.canonical}：identity 只能有一个来源")
            return np.ascontiguousarray(hf_arrays[0])
        if kind == "concat0":
            return np.ascontiguousarray(np.concatenate(hf_arrays, axis=0))
        if kind == "qkv_by_group":
            q, k, v = hf_arrays
            g, hd = int(self.dims["kv_heads"]), int(self.dims["head_dim"])
            qpg = int(self.dims["heads"]) // g
            d = q.shape[1]
            qg = q.reshape(g, qpg * hd, d)
            kg = k.reshape(g, hd, d)
            vg = v.reshape(g, hd, d)
            return np.ascontiguousarray(np.concatenate([qg, kg, vg], axis=1).reshape(-1, d))
        raise NameMapError(f"不认识的 to_megatron：{kind}")

    def from_megatron(self, ent: ParamEntry, arr: np.ndarray, hf_shapes: list) -> list[np.ndarray]:
        """to_megatron 的逆。"""
        kind = ent.to_megatron
        if kind == "identity":
            return [arr]
        if kind == "concat0":
            cuts = np.cumsum([s[0] for s in hf_shapes])[:-1]
            return [np.ascontiguousarray(x) for x in np.split(arr, cuts, axis=0)]
        if kind == "qkv_by_group":
            g, hd = int(self.dims["kv_heads"]), int(self.dims["head_dim"])
            qpg = int(self.dims["heads"]) // g
            d = arr.shape[1]
            a = arr.reshape(g, (qpg + 2) * hd, d)
            q = a[:, : qpg * hd].reshape(-1, d)
            k = a[:, qpg * hd: (qpg + 1) * hd].reshape(-1, d)
            v = a[:, (qpg + 1) * hd:].reshape(-1, d)
            return [np.ascontiguousarray(x) for x in (q, k, v)]
        raise NameMapError(f"不认识的 to_megatron：{kind}")

    @staticmethod
    def hf_from_leaf(leaf: np.ndarray, piece: Piece, hf_shape) -> np.ndarray:
        """从完整的 MaxText 叶子取回一个 HF 张量。"""
        return invert_ops(np.asarray(leaf[leaf_index(piece.index, leaf.ndim)]), piece.ops, hf_shape)

    @staticmethod
    def hf_from_slice(slice_arr: np.ndarray, piece: Piece, hf_shape) -> np.ndarray:
        """叶子已经按 piece.index 取好片时用。"""
        return invert_ops(np.asarray(slice_arr), piece.ops, hf_shape)

    def hf_shapes(self, ent: ParamEntry) -> list[tuple]:
        """由维度推出 HF 张量形状（不读权重）。"""
        d = self.dims
        D, H, G, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
        out = []
        for h in ent.hf:
            if h.endswith(("embed_tokens.weight", "lm_head.weight")):
                out.append((d["vocab"], D))
            elif h.endswith(("layernorm.weight", "model.norm.weight")):
                out.append((D,))
            elif h.endswith(("q_norm.weight", "k_norm.weight")):
                out.append((hd,))
            elif h.endswith("q_proj.weight"):
                out.append((H * hd, D))
            elif h.endswith(("k_proj.weight", "v_proj.weight")):
                out.append((G * hd, D))
            elif h.endswith("o_proj.weight"):
                out.append((D, H * hd))
            elif h.endswith("expert_bias"):
                out.append((d["experts"],))
            elif h.endswith("router.gate.weight"):
                out.append((d["experts"], D))
            elif ".experts." in h or ".shared_mlp." in h:
                f = d["moe_ffn"]
                out.append((D, f) if h.endswith("down_proj.weight") else (f, D))
            elif h.endswith("down_proj.weight"):
                out.append((D, d["ffn"]))
            elif h.endswith(("gate_proj.weight", "up_proj.weight")):
                out.append((d["ffn"], D))
            else:
                raise NameMapError(f"推不出形状：{h}")
        return out

    def leaf_shapes(self) -> dict[str, tuple]:
        """由各片推出每个 MaxText 叶子的完整形状（用于核对抽象模型、或转换时分配数组）。"""
        out: dict[str, list] = {}
        for ent in self.entries:
            for pc, shp in zip(ent.pieces, [self.hf_shapes(ent)[ent.hf.index(pc.hf)] for pc in ent.pieces]):
                sub = list(forward_shape(shp, pc.ops))
                full = out.setdefault(pc.path, None)
                need = len(sub) + len(pc.index)
                if full is None:
                    full = [0] * need
                    out[pc.path] = full
                # 把切掉的轴插回去
                it = iter(sub)
                axes = dict(pc.index)
                for ax in range(need):
                    if ax in axes:
                        full[ax] = max(full[ax], axes[ax] + 1)
                    else:
                        v = next(it)
                        if full[ax] not in (0, v):
                            raise NameMapError(f"{pc.path} 第 {ax} 维形状冲突：{full[ax]} vs {v}")
                        full[ax] = v
        return {k: tuple(v) for k, v in out.items()}
