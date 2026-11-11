"""数据回放：把 GPU golden 每 rank 每步每个 microbatch 原样存下的输入（data/batches/rank<r>.bin + .jsonl），
按 rank → microbatch 的顺序拼成每步的全局 batch，换成 MaxText 的输入字段。只依赖 numpy。

约定（TPU 设备 d ↔ GPU rank d）：
- 全局 batch 第 row = r·M + m 行 = GPU rank r 的第 m 个 microbatch（M = 每 rank 每步的 microbatch 数，golden 是 2）。
- MaxText 按 batch 维把全局 batch 均分到数据并行的各个分片（FSDP 8 时每个设备 per_device_batch_size = M 行）：
  第 d 个分片拿第 d·M … d·M+M−1 行，正好是 GPU rank d 的全部 microbatch。这里的「设备 d」指 batch 维上的第 d 个分片，
  它落在哪个物理设备上由 MaxText 的 mesh 决定，train_hy4.py 把对应关系写进 meta.json（data.device_of_shard）。
- 改用 per_device_batch_size=1 + gradient_accumulation_steps=M 时，MaxText 把全局 batch 按 [行 // M, 行 % M] 拆，
  第 m 个累积步拿 row % M == m 的那些行，设备 d 拿其中第 d 行 = GPU rank d 的第 m 个 microbatch：同一种行序两种排法都对得上。

字段：
  inputs = tokens，targets = labels，inputs_position = targets_position = position_ids（int32）；
  inputs_segmentation = 1（整条 4096 是一段，和 GPU 一样不按文档切注意力）；targets_segmentation = (loss_mask != 0)。
  只接受 0/1 的 loss_mask（混元 3 的 golden 全是 1）；不是 0/1 就报错——MaxText 只有「算 / 不算」两种权重。

用法：
  python replay.py check <golden 运行目录> [--steps 1:1001]   # 逐条核对 sha256，打印每步全局 batch 的摘要
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ("tokens", "labels", "loss_mask", "position_ids")
# 混元 4 的 GPU golden（HF 程序）只存 tokens / labels：没有 loss_mask（每个 token 都算 loss）和 position_ids（0…S−1）。
# 缺的字段按这两条默认补上，指纹只写 golden 里有的字段（与 golden 的 data/rank*.jsonl 同一组键）
DEFAULTS = {"loss_mask": lambda shape: np.ones(shape, np.float32),
            "position_ids": lambda shape: np.broadcast_to(np.arange(shape[-1], dtype=np.int64), shape).copy()}


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class ReplayError(Exception):
    pass


class GoldenBatches:
    """golden 运行目录下 data/batches/ 的只读视图。"""

    def __init__(self, run_dir):
        self.dir = Path(run_dir) / "data" / "batches"
        idx = sorted(self.dir.glob("rank*.jsonl"), key=lambda p: int(p.stem[4:]))
        if not idx:
            raise ReplayError(f"{self.dir} 下没有 rank*.jsonl")
        self.ranks = [int(p.stem[4:]) for p in idx]
        if self.ranks != list(range(len(self.ranks))):
            raise ReplayError(f"rank 编号不连续：{self.ranks}")
        self.index: dict[tuple[int, int, int], dict] = {}          # (step, rank, micro) → fields
        for p in idx:
            r = int(p.stem[4:])
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.index[(int(rec["step"]), r, int(rec["micro"]))] = rec["fields"]
        self.steps = sorted({s for s, _, _ in self.index})
        first = next(iter(self.index.values()))
        self.fields = tuple(k for k in FIELDS if k in first)          # golden 里实际存了的字段
        if "tokens" not in self.fields or "labels" not in self.fields:
            raise ReplayError(f"golden 的 batches 至少要有 tokens 与 labels，实际 {sorted(first)}")
        self.micro = 1 + max(m for _, _, m in self.index)
        for s in self.steps:
            for r in self.ranks:
                for m in range(self.micro):
                    if (s, r, m) not in self.index:
                        raise ReplayError(f"缺第 {s} 步 rank {r} micro {m}")
        self._bins = {r: np.memmap(self.dir / f"rank{r}.bin", dtype=np.uint8, mode="r") for r in self.ranks}

    @property
    def world(self) -> int:
        return len(self.ranks)

    @property
    def global_rows(self) -> int:
        return self.world * self.micro

    def field(self, step: int, rank: int, micro: int, name: str, verify: bool = True) -> np.ndarray:
        f = self.index[(step, rank, micro)][name]
        dt = np.dtype(f["dtype"])
        n = int(np.prod(f["shape"])) * dt.itemsize
        raw = bytes(self._bins[rank][f["offset"]: f["offset"] + n])
        a = np.frombuffer(raw, dtype=dt).reshape(f["shape"])
        if verify and hashlib.sha256(raw).hexdigest() != f["sha256"]:
            raise ReplayError(f"第 {step} 步 rank {rank} micro {micro} 的 {name} sha256 不符")
        return a

    def global_batch(self, step: int, verify: bool = True) -> dict[str, np.ndarray]:
        """{字段: [world·micro, S]}，第 r·M + m 行 = rank r 的 micro m（GPU 原 dtype）。"""
        out = {}
        for name in self.fields:
            rows = [self.field(step, r, m, name, verify).reshape(-1) for r in self.ranks for m in range(self.micro)]
            out[name] = np.stack(rows)
        for name in FIELDS:
            if name not in out:
                out[name] = DEFAULTS[name](out["tokens"].shape)
        return out


def to_maxtext(gb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """GPU 字段 → MaxText 输入字段（int32）。"""
    lm = gb["loss_mask"]
    if not np.all((lm == 0) | (lm == 1)):
        raise ReplayError("loss_mask 不是 0/1，MaxText 的 targets_segmentation 表达不了")
    for k in ("tokens", "labels", "position_ids"):
        if gb[k].min() < 0 or gb[k].max() >= 2 ** 31:
            raise ReplayError(f"{k} 超出 int32")
    pos = gb["position_ids"].astype(np.int32)
    return {
        "inputs": gb["tokens"].astype(np.int32),
        "targets": gb["labels"].astype(np.int32),
        "inputs_position": pos,
        "targets_position": pos.copy(),
        "inputs_segmentation": np.ones_like(pos),
        "targets_segmentation": (lm != 0).astype(np.int32),
    }


def from_maxtext(mt: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """to_maxtext 的逆，用 GPU 的 dtype（int64 / float32）还原，供 TPU 侧写同格式的输入指纹。"""
    return {
        "tokens": np.asarray(mt["inputs"]).astype(np.int64),
        "labels": np.asarray(mt["targets"]).astype(np.int64),
        "loss_mask": np.asarray(mt["targets_segmentation"]).astype(np.float32),
        "position_ids": np.asarray(mt["inputs_position"]).astype(np.int64),
    }


def fingerprints(gb: dict[str, np.ndarray], micro: int, step: int, fields=FIELDS) -> dict[int, list[dict]]:
    """全局 batch（GPU 字段与 dtype）→ {设备 d: [每个 microbatch 一条 data/rank<d>.jsonl 记录]}。
    每条与 GPU 侧同格式：四个字段是 [1, S] 张量原始字节的 sha256，nonpad = loss_mask 之和。"""
    rows = gb["tokens"].shape[0]
    if rows % micro:
        raise ReplayError(f"{rows} 行不能均分成每设备 {micro} 个 microbatch")
    out: dict[int, list[dict]] = {}
    for d in range(rows // micro):
        recs = []
        for m in range(micro):
            row = d * micro + m
            rec = {"step": step, "micro": m}
            for k in fields:
                rec[k] = sha(gb[k][row: row + 1])
            rec["nonpad"] = int(gb["loss_mask"][row].sum())
            recs.append(rec)
        out[d] = recs
    return out


def _parse_steps(s: str | None, have: list[int]) -> list[int]:
    if not s:
        return have
    a, b = (int(x) for x in s.split(":"))
    return [x for x in have if a <= x < b]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="逐条核对 sha256，并核对 golden 自己的 data/rank<r>.jsonl 指纹")
    c.add_argument("golden")
    c.add_argument("--steps", default=None, help="左闭右开，如 1:11")
    a = p.parse_args(argv)
    gb = GoldenBatches(a.golden)
    steps = _parse_steps(a.steps, gb.steps)
    fp_dir = Path(a.golden) / "data"
    golden_fp = {}
    for f in fp_dir.glob("rank*.jsonl"):
        r = int(f.stem[4:])
        for line in f.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                golden_fp[(rec["step"], r, rec["micro"])] = rec
    n_units, bad = 0, []
    mask_values, nonpad, tok_max = set(), set(), 0
    for s in steps:
        g = gb.global_batch(s)                       # 每个字段都核对了 sha256
        mask_values |= set(np.unique(g["loss_mask"]).tolist())
        nonpad |= set(g["loss_mask"].sum(-1).astype(int).tolist())
        tok_max = max(tok_max, int(g["tokens"].max()), int(g["labels"].max()))
        mt = to_maxtext(g)
        back = from_maxtext(mt)
        for k in gb.fields:
            if not np.array_equal(back[k], g[k]) or back[k].dtype != g[k].dtype:
                bad.append(f"{s}:{k} MaxText 往返不逐位")
        for d, recs in fingerprints(back, gb.micro, s, gb.fields).items():
            for rec in recs:
                n_units += 1
                ref = golden_fp.get((s, d, rec["micro"]))
                if ref is None or any(ref[k] != rec[k] for k in (*gb.fields, "nonpad")):
                    bad.append(f"{s}.r{d}.m{rec['micro']}")
    print(f"golden {a.golden}：{gb.world} 个 rank × {gb.micro} 个 microbatch，共 {len(gb.steps)} 步；核对 {len(steps)} 步、"
          f"{n_units} 个单元（每单元 {len(gb.fields)} 个字段：{'、'.join(gb.fields)}）")
    print(f"  loss_mask 取值 {sorted(mask_values)}；每个 microbatch 的 nonpad {sorted(nonpad)}；token / label 最大值 {tok_max}")
    print(f"  batches/*.bin 的 sha256 与索引一致；拼成全局 batch → MaxText 字段 → 还原，再算的指纹与 golden data/rank*.jsonl 不同 {len(bad)} 处 {bad[:5]}")
    print("OK" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
