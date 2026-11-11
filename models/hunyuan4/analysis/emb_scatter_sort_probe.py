"""词嵌入反向的 BF16 scatter-add：编译器什么时候在前面插排序、排没排序时各按什么精度累加。

纯数据并行（DP）下词嵌入梯度离 FP64 远、FSDP 不远；两边优化后 HLO 里 scatter 都是 BF16，差别是 FSDP 那份前面多了
sort、标了 indices_are_sorted=true（04-tpu-self/b1-grad-reduction.md 第四节）。这个脚本把那一个算子单拿出来量。
子命令：

  prep       （CPU）从 golden 运行目录取第 1 步全局批的 token（16 × 4096），存成 npz，给 run 用
  aot        （CPU，只编不跑）用 libtpu 为 v5p 交叉编译：
               ① 插不插 sort 的门槛：表行数 V × 列数 D 的几组，各二分出最小的插 sort 的下标个数 N；
               ② 8 个设备上照 FSDP（表按列切、下标收齐）与 DP（各卡本地 scatter 再跨卡求和；另一份先在 JAX 里排序）各编一份，
                  数每个设备程序里的 sort；
               ③ 单设备每格（N × 标不标 indices_are_sorted × excess precision 开关）的优化后 HLO 存进 --out
  aot-one    （CPU）只编一格，给逐 pass 的 HLO dump 与 TPU 后端（LLO）dump 用，dump 开关由环境变量给（见文末）
  llo-count  数 LLO dump 里 scatter 那颗 kernel 的 final_bundles：bundle 数与几类向量指令的静态条数
  run        （TPU，单设备 + 8 设备）真算：表 [120832, 1024]，更新量 N(0, 1e-3²) 的 BF16（随机种子 0），下标取 prep 的 token：
               N = 2048 / 4096 / 8192 / 16384 / 32768（第 0 条序列的前一半、第 0 条、第 0–1 条、0–3 条、0–7 条）。每个 N 跑：
                 unsorted      原顺序下标，x.at[i].add(u)（训练里词嵌入反向就是这个）
                 unsorted_buf  同上，但零缓冲从 jit 外面传进来（DP 训练步里零缓冲在融合外）
                 presorted     下标在 numpy 里稳定排好序（更新量跟着换），标 indices_are_sorted=True
                 presorted_nf  排好序但不标
                 jaxsort       jit 里先 argsort（稳定）再标 indices_are_sorted=True 做 scatter：在 JAX 里显式排序的修法
                 f32           FP32 更新量、原顺序（embedding_grad_fp32 的写法）
               以上 BF16 的各在 xla_allow_excess_precision 开 / 关下编译。每格对 numpy FP64 的 rel_L2、按本次 scatter 里出现次数
               分桶的 rel_L2、与两种 CPU 仿真逐行逐位相同的比例（E1：每行按下标出现顺序在 FP32 里累加、最后舍一次 BF16；
               E2：每加一次舍一次 BF16），优化后 HLO 里 sort 的个数与 scatter 是否标了 sorted、单次耗时（20 次中位数）。
               --sharded 另在 8 个设备上照 FSDP / DP 各跑一份（DP 跨卡用 BF16 psum，和训练里一样）。
  结论在 04-tpu-self/b1-grad-reduction.md 第七节。

  python models/hunyuan4/analysis/emb_scatter_sort_probe.py prep --golden <GPU g3-a 运行目录> --out tokens.npz
  JAX_PLATFORMS=cpu python models/hunyuan4/analysis/emb_scatter_sort_probe.py aot --out <目录>
  # 每个 pass 之后的 HLO（找插 sort 的 pass）：
  JAX_PLATFORMS=cpu XLA_FLAGS="--xla_dump_to=<目录> --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*" \
    python models/hunyuan4/analysis/emb_scatter_sort_probe.py aot-one --n 16384 --out <目录>
  # TPU 后端（LLO）每个 pass 的产物：开关只认 LIBTPU_INIT_ARGS
  JAX_PLATFORMS=cpu LIBTPU_INIT_ARGS="--xla_jf_dump_to=<目录>/jf --xla_jf_dump_llo_text=true" \
    python models/hunyuan4/analysis/emb_scatter_sort_probe.py aot-one --n 4096 --flag 1 --excess 1 --out <目录>
  python models/hunyuan4/analysis/emb_scatter_sort_probe.py llo-count <目录>/jf … --out llo.json
  python models/hunyuan4/analysis/emb_scatter_sort_probe.py run --distributed --tokens tokens.npz --out <json> --hlo-dir <目录> --sharded
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

V, D = 120832, 1024
NS = (2048, 4096, 8192, 16384, 32768)
BUCKETS = (("1", 1, 1), ("2-9", 2, 9), ("10-99", 10, 99), ("100+", 100, 1 << 30))


# ---------------------------------------------------------------- 仿真与指标（numpy）
def compact(idx: np.ndarray):
    """下标 → (去重后的行号, 每个下标对应的紧凑行号, 每行出现次数)。只在出现过的行上算，省内存。"""
    rows, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
    return rows, inv, cnt


def emulate(inv: np.ndarray, u: np.ndarray, nrows: int, acc: str) -> np.ndarray:
    """每行按下标出现的先后（稳定顺序）依次相加：acc='f32' 在 FP32 里加完舍一次 BF16；'bf16' 每加一次舍一次 BF16。"""
    import ml_dtypes

    bf = ml_dtypes.bfloat16
    order = np.argsort(inv, kind="stable")
    si, su = inv[order], u[order].astype(np.float32)
    first = np.r_[True, si[1:] != si[:-1]]
    seg = np.cumsum(first) - 1
    rank = np.arange(len(si)) - np.flatnonzero(first)[seg]
    out = np.zeros((nrows, u.shape[1]), np.float32)
    for k in range(int(rank.max()) + 1):
        sel = rank == k
        s = seg[sel]
        if acc == "f32":
            out[s] = out[s] + su[sel]
        else:
            out[s] = (out[s] + su[sel]).astype(bf).astype(np.float32)
    return out.astype(bf).astype(np.float32) if acc == "f32" else out


def rel(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float64) - ref) / np.linalg.norm(ref))


def metrics(y: np.ndarray, ref: np.ndarray, cnt: np.ndarray, emus: dict | None) -> dict:
    """y、ref 都是紧凑行 [行数, D]。"""
    r = {"rel_vs_fp64": rel(y, ref), "by_count": {}}
    for name, lo, hi in BUCKETS:
        m = (cnt >= lo) & (cnt <= hi)
        if m.any():
            r["by_count"][name] = rel(y[m], ref[m])
    if emus:
        for n, e in emus.items():
            eq = np.all(y.view(np.uint32) == e.view(np.uint32), axis=1)
            r[f"rows_bitwise_equal_{n}"] = float(eq.mean())
            r[f"rows_bitwise_equal_{n}_count100+"] = float(eq[cnt >= 100].mean()) if (cnt >= 100).any() else None
    return r


def hlo_facts(text: str) -> dict:
    lines = text.splitlines()
    return {
        "n_sort": sum(" sort(" in ln for ln in lines),
        "n_scatter": sum(" scatter(" in ln for ln in lines),
        "n_scatter_sorted": sum(" scatter(" in ln and "indices_are_sorted=true" in ln for ln in lines),
        "scatter_out_dtypes": sorted({ln.split("=")[1].split("[")[0].strip() for ln in lines if " scatter(" in ln}),
    }


# ---------------------------------------------------------------- prep
def cmd_prep(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frameworks" / "maxtext"))
    import replay  # noqa: E402  frameworks/maxtext/replay.py：golden 的 data/batches 读法（含 sha256 核对）

    gb = replay.GoldenBatches(Path(a.golden)).global_batch(1)
    tok = np.asarray(gb["tokens"]).astype(np.int32)
    np.savez(a.out, tokens=tok)
    c = np.bincount(tok.reshape(-1), minlength=V)
    print(json.dumps({"shape": list(tok.shape), "n_unique": int((c > 0).sum()), "max_count": int(c.max())}))


# ---------------------------------------------------------------- aot
def cmd_aot(a):
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    topo1 = topologies.get_topology_desc(platform="tpu", topology_name="v5p:2x2x1")
    sh1 = jax.sharding.SingleDeviceSharding(topo1.devices[0])

    def S(s, d, sh=sh1):
        return jax.ShapeDtypeStruct(s, d, sharding=sh)

    def compile1(v, d, n, sort_flag=False, excess=True):
        def f(i, u):
            return jnp.zeros((v, d), jnp.bfloat16).at[i].add(u, indices_are_sorted=sort_flag)

        c = jax.jit(f).lower(S((n,), jnp.int32), S((n, d), jnp.bfloat16))
        return c.compile(compiler_options={"xla_allow_excess_precision": excess}).as_text()

    res = {"platform_version": None, "threshold": [], "cells": {}, "sharded": {}}
    try:
        from jax.extend.backend import get_backend

        res["platform_version"] = get_backend("tpu").platform_version
    except Exception:  # noqa: BLE001
        pass
    # ① 门槛
    for v, d in ((1000, 1024), (30000, 128), (30000, 1024), (60000, 128), (120832, 128), (120832, 1024), (480000, 128)):
        lo, hi = 1, 1 << 16
        if hlo_facts(compile1(v, d, hi))["n_sort"] == 0:
            res["threshold"].append({"V": v, "D": d, "min_n_with_sort": None, "checked_up_to": hi})
            print(v, d, "no sort up to", hi, flush=True)
            continue
        while hi - lo > 1:
            m = (lo + hi) // 2
            if hlo_facts(compile1(v, d, m))["n_sort"]:
                hi = m
            else:
                lo = m
        res["threshold"].append({"V": v, "D": d, "min_n_with_sort": hi, "V_over_8": v / 8})
        print(v, d, "min N with sort", hi, flush=True)
    # ③ 每格 HLO（单设备）
    for n in NS:
        for flag in (False, True):
            for ex in (True, False):
                name = f"n{n}_{'flag' if flag else 'noflag'}_excess_{'on' if ex else 'off'}"
                t = compile1(V, D, n, flag, ex)
                (out / f"{name}.hlo.txt").write_text(t)
                res["cells"][name] = hlo_facts(t)
    # ② 8 个设备：FSDP（表按列切）与 DP（各卡本地 scatter + 跨卡求和）
    topo8 = topologies.get_topology_desc(platform="tpu", topology_name="v5p:2x2x2")
    mesh = Mesh(np.array(topo8.devices).reshape(8), ("x",))
    nl = 4096

    def fsdp(i, u):  # i: [8·4096] 收齐的下标（复制），u: [8·4096, D] 按列切
        return jnp.zeros((V, D), jnp.bfloat16).at[i].add(u)

    c = jax.jit(fsdp, out_shardings=NamedSharding(mesh, P(None, "x"))).lower(
        S((8 * nl,), jnp.int32, NamedSharding(mesh, P())),
        S((8 * nl, D), jnp.bfloat16, NamedSharding(mesh, P(None, "x"))),
    )
    t = c.compile().as_text()
    (out / "sharded_fsdp8.hlo.txt").write_text(t)
    res["sharded"]["fsdp8"] = hlo_facts(t)

    for jsort in (False, True):

        def dp_local(i, u, jsort=jsort):
            if jsort:
                o = jnp.argsort(i, stable=True)
                i, u = i[o], u[o]
            g = jnp.zeros((V, D), jnp.bfloat16).at[i].add(u, indices_are_sorted=jsort)
            return jax.lax.psum(g, "x")

        f = jax.shard_map(dp_local, mesh=mesh, in_specs=(P("x"), P("x")), out_specs=P())
        c = jax.jit(f).lower(
            S((8 * nl,), jnp.int32, NamedSharding(mesh, P("x"))),
            S((8 * nl, D), jnp.bfloat16, NamedSharding(mesh, P("x"))),
        )
        t = c.compile().as_text()
        k = "dp8_jaxsort" if jsort else "dp8"
        (out / f"sharded_{k}.hlo.txt").write_text(t)
        res["sharded"][k] = hlo_facts(t)
    (out / "aot.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res["sharded"], ensure_ascii=False))


# ---------------------------------------------------------------- run
def cmd_run(a):
    import jax

    if a.distributed:
        jax.distributed.initialize()
    import jax.numpy as jnp
    import ml_dtypes

    tok = np.load(a.tokens)["tokens"].reshape(-1).astype(np.int32)
    rng = np.random.default_rng(0)
    u_all = (1e-3 * rng.standard_normal((NS[-1], D))).astype(ml_dtypes.bfloat16)
    dev = jax.local_devices()[0]
    p0 = jax.process_index() == 0
    hdir = Path(a.hlo_dir) if a.hlo_dir else None
    if hdir and p0:
        hdir.mkdir(parents=True, exist_ok=True)
    res = {
        "what": __doc__.split("\n")[0],
        "V": V,
        "D": D,
        "platform": dev.platform,
        "device_kind": dev.device_kind,
        "jax": jax.__version__,
        "updates": "N(0, 1e-3^2) bf16, seed 0",
        "cells": {},
    }
    try:
        from jax.extend.backend import get_backend

        res["platform_version"] = get_backend().platform_version
    except Exception:  # noqa: BLE001
        pass

    def timed(comp, *args):
        y = comp(*args)
        y.block_until_ready()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            comp(*args).block_until_ready()
            ts.append(time.perf_counter() - t0)
        return y, float(np.median(ts))

    ns = [n for n in NS if n in a.ns] if a.ns else NS
    for n in ns:
        idx = tok[:n]
        u = u_all[:n]
        rows, inv, cnt = compact(idx)
        ref = np.zeros((len(rows), D), np.float64)
        np.add.at(ref, inv, u.astype(np.float64))
        emus = {"E1": emulate(inv, u, len(rows), "f32"), "E2": emulate(inv, u, len(rows), "bf16")}
        o = np.argsort(idx, kind="stable")
        cell_n = {
            "n_rows_touched": int(len(rows)),
            "max_count": int(cnt.max()),
            "bucket_rows": {k: int(((cnt >= lo) & (cnt <= hi)).sum()) for k, lo, hi in BUCKETS},
            "emulate": {k: metrics(e, ref, cnt, None) for k, e in emus.items()},
            "runs": {},
        }
        xi = jax.device_put(jnp.asarray(idx), dev)
        xu = jax.device_put(jnp.asarray(u), dev)
        si = jax.device_put(jnp.asarray(idx[o]), dev)
        su = jax.device_put(jnp.asarray(u[o]), dev)
        xu32 = jax.device_put(jnp.asarray(u.astype(np.float32)), dev)
        zb = jax.device_put(jnp.zeros((V, D), jnp.bfloat16), dev)

        def f_plain(i, v, flag=False):
            return jnp.zeros((V, D), v.dtype).at[i].add(v, indices_are_sorted=flag)

        def f_buf(z, i, v):
            return z.at[i].add(v)

        def f_jaxsort(i, v):
            oo = jnp.argsort(i, stable=True)
            return jnp.zeros((V, D), v.dtype).at[i[oo]].add(v[oo], indices_are_sorted=True)

        variants = {
            "unsorted": (lambda i, v: f_plain(i, v), (xi, xu)),
            "unsorted_buf": (f_buf, (zb, xi, xu)),
            "presorted": (lambda i, v: f_plain(i, v, True), (si, su)),
            "presorted_nf": (lambda i, v: f_plain(i, v), (si, su)),
            "jaxsort": (f_jaxsort, (xi, xu)),
        }
        for vname, (fn, args) in variants.items():
            for ex in (True, False):
                name = f"bf16_{vname}_excess_{'on' if ex else 'off'}"
                comp = jax.jit(fn).lower(*args).compile(compiler_options={"xla_allow_excess_precision": ex})
                txt = comp.as_text()
                if hdir and p0:
                    (hdir / f"n{n}_{name}.hlo.txt").write_text(txt)
                y, sec = timed(comp, *args)
                y = np.asarray(y)[rows].astype(np.float32)
                r = metrics(y, ref, cnt, emus)
                r.update(hlo_facts(txt), seconds_median=sec)
                cell_n["runs"][name] = r
                if p0:
                    print(n, name, json.dumps({k: v for k, v in r.items() if k != "by_count"}), flush=True)
        comp = jax.jit(lambda i, v: f_plain(i, v)).lower(xi, xu32).compile()
        txt = comp.as_text()
        if hdir and p0:
            (hdir / f"n{n}_f32_unsorted.hlo.txt").write_text(txt)
        y, sec = timed(comp, xi, xu32)
        r = metrics(np.asarray(y)[rows].astype(np.float32), ref, cnt, None)
        r.update(hlo_facts(txt), seconds_median=sec)
        cell_n["runs"]["f32_unsorted"] = r
        res["cells"][str(n)] = cell_n
    if a.sharded:
        res["sharded"] = run_sharded(tok, u_all, hdir if p0 else None)
    if p0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        print("wrote", a.out, flush=True)


def run_sharded(tok, u_all, hdir):
    """8 个设备：FSDP（全局 32768 个下标，表按列切）与 DP（每卡本地 4096 个、整表 scatter，再 BF16 psum）。"""
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()).reshape(-1), ("x",))
    nd = mesh.devices.size
    nl = 4096
    idx, u = tok[: nd * nl], u_all[: nd * nl]
    rows, inv, cnt = compact(idx)
    ref = np.zeros((len(rows), D), np.float64)
    np.add.at(ref, inv, u.astype(np.float64))

    def put(x, spec):
        return jax.make_array_from_callback(x.shape, NamedSharding(mesh, spec), lambda s: x[s])

    out = {"n_devices": int(nd), "tokens_per_device": nl, "runs": {}}

    def fsdp(i, v):
        return jnp.zeros((V, D), jnp.bfloat16).at[i].add(v)

    def dp_body(i, v, jsort):
        if jsort:
            o = jnp.argsort(i, stable=True)
            i, v = i[o], v[o]
        return jax.lax.psum(jnp.zeros((V, D), jnp.bfloat16).at[i].add(v, indices_are_sorted=jsort), "x")

    cases = {
        "fsdp": (
            jax.jit(fsdp, out_shardings=NamedSharding(mesh, P(None, "x"))),
            (put(idx, P()), put(np.asarray(u), P(None, "x"))),
        ),
    }
    for js in (False, True):
        f = jax.shard_map(lambda i, v, js=js: dp_body(i, v, js), mesh=mesh, in_specs=(P("x"), P("x")), out_specs=P())
        cases["dp_jaxsort" if js else "dp"] = (jax.jit(f), (put(idx, P("x")), put(np.asarray(u), P("x"))))
    for name, (fn, args) in cases.items():
        for ex in (True, False):
            k = f"{name}_excess_{'on' if ex else 'off'}"
            comp = fn.lower(*args).compile(compiler_options={"xla_allow_excess_precision": ex})
            txt = comp.as_text()
            if hdir:
                (hdir / f"sharded_{k}.hlo.txt").write_text(txt)
            y = comp(*args)
            y = np.asarray(multihost_utils.process_allgather(y, tiled=True))[rows].astype(np.float32)
            r = metrics(y, ref, cnt, None)
            r.update(hlo_facts(txt))
            out["runs"][k] = r
            if jax.process_index() == 0:
                print("sharded", k, json.dumps({kk: vv for kk, vv in r.items() if kk != "by_count"}), flush=True)
    return out


# ---------------------------------------------------------------- aot-one / llo-count：逐 pass HLO 与 TPU 后端 dump
def cmd_aot_one(a):
    """只编一格（单设备 v5p，表 [V, D] 的 BF16 scatter-add），给逐 pass dump / LLO dump 用：开关由调用方的环境变量给。"""
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies

    sh = jax.sharding.SingleDeviceSharding(
        topologies.get_topology_desc(platform="tpu", topology_name="v5p:2x2x1").devices[0]
    )

    def embscatter(i, u):
        return jnp.zeros((V, D), jnp.bfloat16).at[i].add(u, indices_are_sorted=bool(a.flag))

    args = (
        jax.ShapeDtypeStruct((a.n,), jnp.int32, sharding=sh),
        jax.ShapeDtypeStruct((a.n, D), jnp.bfloat16, sharding=sh),
    )
    c = jax.jit(embscatter).lower(*args).compile(compiler_options={"xla_allow_excess_precision": bool(a.excess)})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "after_optimizations.hlo.txt").write_text(c.as_text())


def cmd_llo_count(a):
    """每个 LLO dump 目录里 scatter 那颗 kernel 的 final_bundles：数 bundle 与几类向量指令（静态条数，不是执行次数）。"""
    import re

    ops = ("vld", "vst", "vadd.f32", "vunpack.c.l.bf16", "vunpack.i.l.bf16", "vpack.c.bf16", "vpack.i.bf16")
    res = {}
    for d in a.dirs:
        fs = [f for f in sorted(Path(d).glob("*scatter*final_bundles.txt")) if "schedule-analysis" not in f.name]
        if not fs:
            res[d] = {"error": "没有 scatter 的 final_bundles"}
            continue
        t = fs[0].read_text()
        r = {"kernel": re.sub(r"^[0-9]+-", "", fs[0].name).split("-")[0], "bundles": t.count("{")}
        r.update({o: len(re.findall(r"\b" + re.escape(o) + r"\b", t)) for o in ops})
        res[Path(d).parent.name if Path(d).name == "jf" else Path(d).name] = r
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("prep")
    q.add_argument("--golden", required=True)
    q.add_argument("--out", required=True)
    q = sp.add_parser("aot")
    q.add_argument("--out", required=True)
    q = sp.add_parser("aot-one")
    q.add_argument("--n", type=int, default=4096)
    q.add_argument("--flag", type=int, default=0, help="1 = indices_are_sorted=True")
    q.add_argument("--excess", type=int, default=1)
    q.add_argument("--out", required=True)
    q = sp.add_parser("llo-count")
    q.add_argument("dirs", nargs="+")
    q.add_argument("--out")
    q = sp.add_parser("run")
    q.add_argument("--tokens", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--hlo-dir")
    q.add_argument("--ns", type=int, nargs="*")
    q.add_argument("--sharded", action="store_true")
    q.add_argument("--distributed", action="store_true", help="多主机切片上先 jax.distributed.initialize()")
    a = p.parse_args(argv)
    {"prep": cmd_prep, "aot": cmd_aot, "aot-one": cmd_aot_one, "llo-count": cmd_llo_count, "run": cmd_run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
