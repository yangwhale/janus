"""整套参数导出 / 导入（TPU 侧）：与 GPU 侧 frameworks/megatron/param_swap.py 同一种文件格式，状态可以在两侧之间互换。

    <目录>/rank<r>.safetensors   Megatron 名字与布局、FP32；rank 0 存全部非专家参数 + 专家 0…23，rank r>0 只存专家 24r…24r+23
                                （本地名 weight<k>）；路由偏置名字前加 "buffer."
    <目录>/manifest.json         每个张量的 sha256（FP32 原始字节）与形状

导出：MaxText 状态 → name_map 的 Megatron 布局（与参数清单 params_sha 同一条路径，只是不转 BF16）。
导入：Megatron 布局 → HF 张量（name_map.from_megatron）→ MaxText 叶子（与 convert_hf_to_maxtext.py 同一种搬法，
     每个元素恰好写一次），按状态里原叶子的 sharding 放回；放回之后再按导出的路径读出来逐张量算 sha256 对 manifest。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from janus.dump import maxtext as jd
from janus.dump.name_map import apply_ops, leaf_index
from janus.store import read_safetensors_header, read_safetensors_tensor, write_safetensors


def _sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a, dtype=np.float32)
    return hashlib.sha256(a.reshape(-1).view(np.uint8)).hexdigest()


def _name(nm, ent) -> str:
    return ("buffer." if ent.buffer else "") + nm.megatron_name(ent)


def megatron_tensors(dump, nm, flat_state):
    """yield (rank, 名字, FP32 数组)：全部参数与 buffer，按 GPU EP 的持有关系分 rank（非专家参数只归 rank 0）。
    多进程时所有进程都要调（读设备上的状态是集合操作）。"""
    for ent, arr in dump._iter_megatron(flat_state, "param"):
        owner = nm.expert_owner(ent)
        yield (0 if owner is None else owner), _name(nm, ent), np.ascontiguousarray(arr, dtype=np.float32)


def export(dump, nm, flat_state, out_dir, step: int, source: dict, process_index: int) -> dict | None:
    out_dir = Path(out_dir)
    per_rank: dict[int, dict[str, np.ndarray]] = {}
    for r, name, arr in megatron_tensors(dump, nm, flat_state):
        per_rank.setdefault(r, {})[name] = arr
    # 每个进程都写一份到自己的本地盘（内容相同、sha 也各算一遍，进程之间用时对称；只有进程 0 那份会被上传）：写几十 GB 要几分钟，只让进程 0 写的话
    # 其余进程先到程序末尾的关停屏障，等超时就把整个作业判失败（t-exp-99 / t-exp-99b 两次都是这样，导出没传完）
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks = {}
    for r in sorted(per_rank):
        write_safetensors(out_dir / f"rank{r}.safetensors", per_rank[r])
        ranks[str(r)] = {k: {"sha256": _sha(v), "shape": list(v.shape)} for k, v in per_rank[r].items()}
        per_rank[r] = None
    if process_index != 0:
        return None
    man = {"format": "megatron-per-rank-fp32", "step": step,
           "state": f"θ_{step}（第 {step} 步更新之后的 FP32 主权重）", "source": source, "ranks": ranks}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=0))
    return man


def _fetch(src: str, rel: str, tmp: Path) -> Path:
    if "://" not in src:
        return Path(src) / rel
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / rel
    subprocess.run(["gcloud", "storage", "cp", "-q", f"{src.rstrip('/')}/{rel}", str(dst)], check=True)
    return dst


def load_leaves(src: str, nm, leaf_shapes: dict, tmp="/tmp/params_from") -> tuple[dict, dict]:
    """Megatron 布局的整套参数 → {MaxText 叶子路径（params/… 或 MoEBiasVar/…）: FP32 数组}。读入时逐张量核 sha256。"""
    tmp = Path(tmp)
    man = json.loads(_fetch(src, "manifest.json", tmp).read_text())
    leaves = {p: np.zeros(s, np.float32) for p, s in leaf_shapes.items()}
    written = {p: np.zeros(s, np.uint8) for p, s in leaf_shapes.items()}
    by_rank: dict[int, list] = {}
    for ent in nm.entries:
        owner = nm.expert_owner(ent)
        by_rank.setdefault(0 if owner is None else owner, []).append(ent)
    n = 0
    for r in sorted(by_rank):
        path = _fetch(src, f"rank{r}.safetensors", tmp)
        hdr, _ = read_safetensors_header(path)
        for ent in by_rank[r]:
            name = _name(nm, ent)
            if name not in hdr:
                raise SystemExit(f"[params-from] rank{r}.safetensors 里没有 {name}")
            arr, _ = read_safetensors_tensor(path, name)
            arr = np.asarray(arr, np.float32)
            if _sha(arr) != man["ranks"][str(r)][name]["sha256"]:
                raise SystemExit(f"[params-from] {name} 的 sha256 与 manifest 不符")
            hf = nm.from_megatron(ent, arr, nm.hf_shapes(ent))
            for pc in ent.pieces:
                y = apply_ops(hf[ent.hf.index(pc.hf)], pc.ops)
                idx = leaf_index(pc.index, leaves[pc.path].ndim)
                if leaves[pc.path][idx].shape != y.shape:
                    raise SystemExit(f"[params-from] {name} → {pc.path}{pc.index}：形状 {y.shape} ≠ {leaves[pc.path][idx].shape}")
                leaves[pc.path][idx] = y
                written[pc.path][idx] += 1
            n += 1
        if "://" in src:
            os.remove(path)
    for p, w in written.items():
        if not np.all(w == 1):
            raise SystemExit(f"[params-from] {p}：有元素没写或写了不止一次（min {w.min()}，max {w.max()}）")
    return leaves, {"source": man.get("source"), "step": man.get("step"), "tensors": n, "manifest": man}


def set_state(state, leaves: dict):
    """把叶子放回 NNX 状态（按原叶子的 sharding）。leaves 的键是 name_map 的叶子路径（params/… 或 MoEBiasVar/…）。"""
    import jax
    from flax import nnx
    from flax.nnx import variablelib
    want = {jd.state_key(p): v for p, v in leaves.items()}
    done = set()
    for path, var in nnx.to_flat_state(state):
        if isinstance(var, nnx.RngState):
            continue
        col = variablelib.variable_name_from_type(type(var), allow_register=True)
        key = col + "/" + "/".join(str(x) for x in path)
        if key not in want:
            continue
        old = var.get_value() if hasattr(var, "get_value") else var.value
        arr = want[key].astype(old.dtype)
        if tuple(arr.shape) != tuple(old.shape):
            raise SystemExit(f"[params-from] {key} 形状 {arr.shape} ≠ 状态里的 {old.shape}")
        new = jax.make_array_from_callback(arr.shape, old.sharding, lambda idx, a=arr: a[idx])
        if hasattr(var, "set_value"):
            var.set_value(new)
        else:
            var.value = new
        done.add(key)
    missing = sorted(set(want) - done)
    if missing:
        raise SystemExit(f"[params-from] 状态里找不到这些叶子：{missing[:5]}")
    return len(done)


def verify(dump, nm, flat_state, manifest: dict) -> dict:
    """放回之后按导出的路径读出来，逐张量 sha256 对 manifest。"""
    bad, n = [], 0
    for r, name, arr in megatron_tensors(dump, nm, flat_state):
        n += 1
        if _sha(arr) != manifest["ranks"][str(r)][name]["sha256"]:
            bad.append(name)
    return {"tensors": n, "mismatch": len(bad), "first_mismatch": bad[:5]}
