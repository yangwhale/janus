"""逐张量核对权重：name_map 覆盖、Megatron 布局、MaxText 检查点。四个子命令，全量不抽样。

  # 1) name_map 覆盖了 HF 目录里的全部张量（名字集合相等），且推出的形状和 HF 头里的一致
  python verify_ckpt.py names --hf /data/hy3-4l

  # 2) 按 name_map 把 HF 拼成 Megatron 布局（qkv 按 KV 组交错、gate|up 拼接、专家按 EP rank 分本地号），
  #    逐参数算 sha256，对 GPU 运行的 params_sha/rank<r>.init.json（核对 name_map 的 Megatron 一列）
  python verify_ckpt.py megatron --hf /data/hy3-4l --gpu-run <golden 运行目录>

  # 3) 读 MaxText 检查点，按 name_map 反向取回每个 HF 张量（FP32 → BF16，expert_bias 保持 FP32），
  #    逐张量算 sha256，对 HF 目录的 tensor_sha256.json（需要 MaxText 环境：orbax、jax）
  python verify_ckpt.py maxtext --ckpt <转换输出>/0/items --hf /data/hy3-4l [--janus-run <运行根>/hy3-4l-maxtext-ckpt]

  # 4) 把 HF 原件写成 janus 运行目录（参数清单 = tensor_sha256.json 的 sha256 + safetensors 头里的 dtype / 形状）
  python verify_ckpt.py hf-manifest --hf /data/hy3-4l --janus-run <运行根>/hy3-4l-hf

3) 给了 --janus-run 时，把「MaxText 检查点还原成 HF 命名、形状、dtype」的逐张量清单写成运行目录
（params_sha/rank0.init.json，另记 stored_dtype = 检查点里实际存的 dtype），配合 4) 用 janus 的正式检查比：
  python -m janus run models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-ckpt-convert.yaml   # checkpoint_equal 清单模式 + dtype_manifest
还原时 dtype 按检查点里存的来：存 FP32 的只有在 HF 原件是 BF16、且每个值都是 BF16 精确值时才转回 BF16，
所以 expert_bias 若被转换存成 BF16，dtype 清单会报出来；值被改过，sha256 会对不上。

每个子命令最后打印一行 `OK` 或 `FAIL`，退出码 0 / 1。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))           # 仓库根：import janus

from janus.dump.name_map import NameMap  # noqa: E402
from janus.store import read_safetensors_header, read_safetensors_tensor  # noqa: E402

NAME_MAP = HERE / "name_map.yaml"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class HF:
    """HF 目录：按名字读张量（保持原 dtype 的位模式）。"""

    def __init__(self, d):
        self.d = Path(d)
        self.wm = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]
        self.manifest = json.loads((self.d / "tensor_sha256.json").read_text())

    def header(self, name):
        h, _ = read_safetensors_header(self.d / self.wm[name])
        return h[name]

    def get(self, name) -> np.ndarray:
        return read_safetensors_tensor(self.d / self.wm[name], name)[0]


def write_manifest_run(path, run_name: str, side: str, framework: str, manifest: dict, extra: dict):
    """参数清单写成合规的 janus 运行目录（docs/reference/dump-format.md）：只有 meta.json、params_sha/rank0.init.json、DONE。"""
    import datetime

    from janus.store import RunWriter
    meta = {"schema_version": 1, "run_name": run_name, "side": side, "framework": framework, "model": "hunyuan3-4l",
            "code": {"repo": "janus", "commit": extra.pop("janus_commit", "unknown")}, "image": "n/a（CPU）",
            "versions": extra.pop("versions", {}), "hardware": {"accelerator": "cpu", "count": 1, "topology": "-"},
            "parallelism": {"dp": 1, "fsdp": 1, "tp": 1, "ep": 1, "cp": 1, "pp": 1},
            "config": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "checkpoint": extra.pop("checkpoint"), "data": {"manifest": "n/a", "manifest_sha256": None, "seed": None},
            "steps": {"first": 0, "last": 0, "convention": "golden_step = step"}, "determinism": {}, "dtypes": {},
            "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
            **extra}
    w = RunWriter(path, meta)
    (Path(path) / "params_sha").mkdir(exist_ok=True)
    (Path(path) / "params_sha" / "rank0.init.json").write_text(json.dumps(manifest, indent=0, sort_keys=True))
    w.done()
    print(f"已写运行目录 {path}（{len(manifest)} 项）")


def _norm_st_dtype(st: str) -> str:
    return {"BF16": "bfloat16", "F32": "float32", "F16": "float16", "F64": "float64"}.get(st, st)


def cmd_hf_manifest(a) -> bool:
    hf = HF(a.hf)
    man = {}
    for name, h in sorted(hf.manifest.items()):
        hd = hf.header(name)
        man[name] = {"sha256": h, "shape": list(hd["shape"]), "dtype": _norm_st_dtype(hd["dtype"])}
    msha = hashlib.sha256((Path(a.hf) / "tensor_sha256.json").read_bytes()).hexdigest()
    write_manifest_run(a.janus_run, Path(a.janus_run).name, "golden-hf", "hf", man,
                       {"checkpoint": {"path": str(a.hf), "manifest_sha256": msha}})
    return len(man) == len(hf.manifest)


def cmd_names(a) -> bool:
    nm, hf = NameMap.load(a.name_map), HF(a.hf)
    mine, theirs = nm.hf_names(), set(hf.manifest)
    only_map, only_hf = sorted(mine - theirs), sorted(theirs - mine)
    print(f"name_map 展开 {len(nm.entries)} 个参数项，覆盖 HF 张量 {len(mine)} 个；tensor_sha256.json {len(theirs)} 个")
    print(f"  只在 name_map：{len(only_map)} {only_map[:5]}")
    print(f"  只在 HF：{len(only_hf)} {only_hf[:5]}")
    bad_shape = []
    for ent in nm.entries:
        for h, shp in zip(ent.hf, nm.hf_shapes(ent)):
            if h in hf.wm and list(hf.header(h)["shape"]) != list(shp):
                bad_shape.append((h, hf.header(h)["shape"], shp))
    print(f"  形状不符：{len(bad_shape)} {bad_shape[:3]}")
    return not only_map and not only_hf and not bad_shape


def cmd_megatron(a) -> bool:
    nm, hf = NameMap.load(a.name_map), HF(a.hf)
    gdir = Path(a.gpu_run) / "params_sha"
    gpu = {int(f.name.split(".")[0][4:]): json.loads(f.read_text()) for f in sorted(gdir.glob(f"rank*.{a.tag}.json"))}
    if not gpu:
        print(f"{gdir} 下没有 rank*.{a.tag}.json")
        return False
    mine: dict[int, dict] = {r: {} for r in gpu}
    for ent in nm.entries:
        arr = nm.to_megatron(ent, [hf.get(h) for h in ent.hf])
        name = ("buffer." if ent.buffer else "") + nm.megatron_name(ent)
        rec = {"sha256": sha(arr), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        owner = nm.expert_owner(ent)
        for r in mine:
            if owner is None or owner == r:
                mine[r][name] = rec
    ok = True
    for r in sorted(gpu):
        g, m = gpu[r], mine[r]
        only_g, only_m = sorted(set(g) - set(m)), sorted(set(m) - set(g))
        differ = [n for n in sorted(set(g) & set(m)) if g[n]["sha256"] != m[n]["sha256"]]
        shape_bad = [n for n in sorted(set(g) & set(m)) if list(g[n]["shape"]) != m[n]["shape"]]
        print(f"rank {r}: GPU {len(g)} 个，name_map 拼出 {len(m)} 个；sha256 不同 {len(differ)}；形状不同 {len(shape_bad)}；"
              f"只在 GPU {len(only_g)} {only_g[:3]}；只在 name_map {len(only_m)} {only_m[:3]}")
        if differ:
            print("   不同的前几个：", differ[:5])
        ok &= not (only_g or only_m or differ or shape_bad)
    return ok


def cmd_maxtext(a) -> bool:
    import orbax.checkpoint as ocp  # MaxText 环境里才有

    nm, hf = NameMap.load(a.name_map), HF(a.hf)
    ckpt = Path(a.ckpt)
    tree = ocp.PyTreeCheckpointer().restore(str(ckpt))
    root = tree.get("params", tree)
    print("顶层集合：", sorted(root))

    def leaf(path):
        cur = root
        for k in path.split("/"):
            cur = cur[k]
        return np.asarray(cur)

    import ml_dtypes
    differ, n, cache, man = [], 0, {}, {}
    for ent in nm.entries:
        shapes = nm.hf_shapes(ent)
        for pc in ent.pieces:
            if pc.path not in cache:
                cache.clear()                            # 一次只留一个叶子在内存里
                cache[pc.path] = leaf(pc.path)
            full = cache[pc.path]
            x = nm.hf_from_leaf(full, pc, shapes[ent.hf.index(pc.hf)])
            stored = str(x.dtype)
            want = hf.header(pc.hf)["dtype"]
            # 按检查点里实际存的 dtype 还原：FP32 只在 HF 原件是 BF16、且每个值都是 BF16 精确值时转回 BF16
            if want == "BF16" and x.dtype == np.float32:
                y = x.astype(ml_dtypes.bfloat16)
                if np.array_equal(y.astype(np.float32), x):
                    x = y
                else:
                    differ.append(pc.hf + "（FP32 值不是 BF16 精确值）")
            n += 1
            h = sha(x)
            man[pc.hf] = {"sha256": h, "shape": list(x.shape), "dtype": str(x.dtype), "stored_dtype": stored,
                          "maxtext": {"path": pc.path, "index": [list(t) for t in pc.index]}}
            if h != hf.manifest[pc.hf]:
                differ.append(pc.hf)
    print(f"从 MaxText 检查点取回 {n} 个 HF 张量；与 tensor_sha256.json 不同 {len(differ)} 个 {differ[:5]}")
    if a.janus_run:
        import jax
        msha = hashlib.sha256((Path(a.hf) / "tensor_sha256.json").read_bytes()).hexdigest()
        write_manifest_run(a.janus_run, Path(a.janus_run).name, "tpu", "maxtext", man,
                           {"checkpoint": {"path": str(ckpt), "manifest_sha256": msha},
                            "versions": {"jax": jax.__version__, "orbax_checkpoint": getattr(ocp, "__version__", None)},
                            "name_map": {"path": str(a.name_map),
                                         "sha256": hashlib.sha256(Path(a.name_map).read_bytes()).hexdigest()}})
    return n == len(hf.manifest) and not differ


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name-map", default=str(NAME_MAP))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("names")
    s.add_argument("--hf", required=True)
    s = sub.add_parser("megatron")
    s.add_argument("--hf", required=True)
    s.add_argument("--gpu-run", required=True, help="GPU 运行目录（至少要有 params_sha/）")
    s.add_argument("--tag", default="init", choices=["init"], help="只有 init 和 HF 权重相同")
    s = sub.add_parser("maxtext")
    s.add_argument("--ckpt", required=True, help="MaxText 参数检查点（…/0/items）")
    s.add_argument("--hf", required=True)
    s.add_argument("--janus-run", default=None, help="把还原出的逐张量清单写成 janus 运行目录")
    s = sub.add_parser("hf-manifest")
    s.add_argument("--hf", required=True)
    s.add_argument("--janus-run", required=True)
    a = p.parse_args(argv)
    ok = {"names": cmd_names, "megatron": cmd_megatron, "maxtext": cmd_maxtext, "hf-manifest": cmd_hf_manifest}[a.cmd](a)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
