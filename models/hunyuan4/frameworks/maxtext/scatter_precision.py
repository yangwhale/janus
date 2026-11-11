"""词嵌入反向的 scatter-add 在 TPU 上到底按什么精度累加：同一批真实 token 下标、随机更新量，单设备上跑
x.at[idx].add(u)（训练里词嵌入反向就是这个算子：表先转 BF16 再取行，反向是对 BF16 表的 scatter-add），
在 `xla_allow_excess_precision` 开 / 关两种编译选项下，各对 numpy FP64 比，并对两种 CPU 仿真逐位比：
- E1「FP32 累加、最后舍一次」：每行在 FP32 里按下标出现顺序依次相加，结果舍成 BF16；
- E2「BF16 逐次累加」：每加一次就舍一次 BF16（同一顺序）。
另跑 FP32 更新量的 scatter-add 作对照（`embedding_grad_fp32` 打开时的写法）。按 token 在本批出现次数分桶报 rel_L2。

下标：golden 第 --step 步、第 --micro 个 microbatch 的全部行（8 个设备 × 4096 个 token = 32768 个，和训练里一个累积步
的词嵌入反向同一批下标；训练时嵌入维按 FSDP 切 8 份，每份 128 维，这里整表 1024 维在一个设备上算，每一维互不相干）。
更新量：N(0, 1e-3²) 的 BF16（随机种子 0）。只在进程 0、本机第 0 个设备上算；结果写 --out。

  SCRIPT=scatter_precision.py STAGE=mini YML=hy4-mini-t1.yml GOLDEN=<g2s-a> bash run.sh <运行名> 1
  python scatter_precision.py --golden <本地 golden 目录> --out <json> --cpu   # 本机 CPU 上验证脚本（CPU 后端，不代表 TPU）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay  # noqa: E402

V, D = 120832, 1024


def emulate(idx: np.ndarray, u: np.ndarray, acc: str) -> np.ndarray:
    """按下标出现顺序（稳定排序）逐行累加：acc='f32' 在 FP32 里加完再舍 BF16；acc='bf16' 每次加完都舍 BF16。"""
    import ml_dtypes

    bf = ml_dtypes.bfloat16
    order = np.argsort(idx, kind="stable")
    si, su = idx[order], u[order].astype(np.float32)
    first = np.r_[True, si[1:] != si[:-1]]
    seg = np.cumsum(first) - 1
    rank = np.arange(len(si)) - np.flatnonzero(first)[seg]
    rows = si[first]
    acc_v = np.zeros((len(rows), u.shape[1]), np.float32)
    for k in range(int(rank.max()) + 1):
        sel = rank == k
        s = seg[sel]
        if acc == "f32":
            acc_v[s] = acc_v[s] + su[sel]
        else:
            acc_v[s] = (acc_v[s] + su[sel]).astype(bf).astype(np.float32)
    out = np.zeros((V, u.shape[1]), np.float32)
    out[rows] = acc_v.astype(bf).astype(np.float32)
    return out


def rel(x: np.ndarray, ref: np.ndarray, rows=None) -> float:
    if rows is not None:
        x, ref = x[rows], ref[rows]
    return float(np.linalg.norm(x.astype(np.float64) - ref) / np.linalg.norm(ref))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--micro", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    a, _ = p.parse_known_args(argv)
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    if a.cpu:
        jax.config.update("jax_platforms", "cpu")
    gb = replay.GoldenBatches(Path(a.golden)).global_batch(a.step)
    m = 2
    idx = np.asarray(gb["tokens"][a.micro::m]).reshape(-1).astype(np.int32)  # 第 micro 个累积步的全部行
    rng = np.random.default_rng(0)
    u = (1e-3 * rng.standard_normal((idx.size, D))).astype(ml_dtypes.bfloat16)
    ref = np.zeros((V, D), np.float64)
    np.add.at(ref, idx, u.astype(np.float64))
    dev = jax.local_devices()[0]
    out = {"what": __doc__.split("\n")[0], "n_tokens": int(idx.size), "n_unique": int(np.unique(idx).size),
           "platform": dev.platform, "device_kind": dev.device_kind, "jax": jax.__version__, "runs": {}}
    try:
        from jax.extend.backend import get_backend
        out["platform_version"] = get_backend().platform_version
    except Exception:  # noqa: BLE001
        pass
    counts = np.bincount(idx, minlength=V)
    buckets = {"1": counts == 1, "2-9": (counts >= 2) & (counts < 10), "10-99": (counts >= 10) & (counts < 100),
               ">=100": counts >= 100}
    out["bucket_rows"] = {k: int(v.sum()) for k, v in buckets.items()}
    out["max_count"] = int(counts.max())
    if jax.process_index() == 0:
        e1 = emulate(idx, u, "f32")
        e2 = emulate(idx, u, "bf16")
        out["emulate"] = {n: {"rel_vs_fp64": rel(e, ref), "by_count": {k: rel(e, ref, b) for k, b in buckets.items()}}
                          for n, e in (("E1_f32_acc_round_once", e1), ("E2_bf16_round_each_add", e2))}
    xi = jax.device_put(jnp.asarray(idx), dev)
    for upd in ("bf16", "f32"):
        uu = jax.device_put(jnp.asarray(u if upd == "bf16" else u.astype(np.float32)), dev)

        def f(i, v, dt=uu.dtype):
            return jnp.zeros((V, D), dt).at[i].add(v)

        for excess in (True, False):
            name = f"{upd}_updates_excess_{'on' if excess else 'off'}"
            try:
                comp = jax.jit(f).lower(xi, uu).compile(compiler_options={"xla_allow_excess_precision": excess})
                y = np.asarray(comp(xi, uu)).astype(np.float32)
            except Exception as e:  # noqa: BLE001
                out["runs"][name] = {"error": repr(e)[:500]}
                continue
            r = {"rel_vs_fp64": rel(y, ref), "by_count": {k: rel(y, ref, b) for k, b in buckets.items()}}
            if jax.process_index() == 0 and upd == "bf16":
                for n, e in (("E1", e1), ("E2", e2)):
                    r[f"bitwise_rows_equal_{n}"] = float(np.mean(np.all(y[counts > 0] == e[counts > 0], axis=1)))
                    r[f"bitwise_rows_equal_{n}_count>=100"] = float(np.mean(np.all(y[counts >= 100] == e[counts >= 100], axis=1)))
            out["runs"][name] = r
            print(name, json.dumps({k: v for k, v in r.items() if k != "by_count"}), flush=True)
    if jax.process_index() == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(out["emulate"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
