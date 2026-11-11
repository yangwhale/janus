"""一次实际排查：TPU 上 jit 把 2 行 / 4 行的行块算错（逐算子执行正确）的复现与扫描。

一个进程只测一种「libtpu × LIBTPU_INIT_ARGS」组合；主进程按 --libtpu / --flags 逐个起子进程，汇总成一份 JSON。
每个用例在三处各算一遍：宿主机 CPU 上 jit（参考）、TPU 上 jit、TPU 上 jax.disable_jit() 逐算子执行，报 out1 的相对误差
（max|差| / max|参考|）。

用例：
  issue      issue 正文里的函数，一字不改
  grid       小块行数 k × rows 块行数 m 的网格（宽度 2）
  square     k = 宽度 w 的几种取值
  barrier    出错用例上，对 a3[:k] 的读取加 optimization_barrier（workaround ②）
  align8     小块行数补到 8、16（workaround ③）

用法（v5p 单设备；镜像同 frameworks/maxtext/run.sh）：
  python jit_miscompile_probe.py --out probe.json \
      --libtpu default --libtpu /tmp/lt_0.0.52 \
      --flags= --flags=--xla_jf_enable_multi_output_fusion=false
  其中 /tmp/lt_X 是 `pip install --no-deps --target /tmp/lt_X libtpu==X` 装出来的目录；default 表示镜像自带的那份。
"""

import argparse
import json
import os
import subprocess
import sys

BAD_SIGNAL = 1e-3  # 相对误差超过它算「算错」；正常结果在 1e-7 量级


def _child(cases):
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax import lax

  jax.config.update("jax_default_matmul_precision", "highest")

  def issue_f(a0, a1, a2, a3, a4, a5, a6, a7, a9):
    t = a1 @ a2.T
    r = jnp.sum(t * a0, axis=1)
    t14 = jnp.pad(jnp.broadcast_to(r[:, None], (2, 2)), ((0, 4), (0, 0)))
    acc = a5 + a3.at[:2].set(a4) + jnp.pad(a3[:2] + a6 * a0, ((0, 4), (0, 0)))
    qkv = jnp.pad(jnp.concatenate([a7, t14]), ((0, 0), (4, 0)))
    rows = qkv[:6][2:6]
    out1 = acc + jnp.pad(rows @ a9.T, ((2, 0), (0, 0)))
    return jnp.sum(t, axis=0), out1

  def make(k=2, m=4, w=2, d=2, barrier=False, dyn=False):
    """k=w=d=2、m=4 时与 issue 的函数相同。"""
    rows_total = k + m

    def f(a0, a1, a2, a3, a4, a5, a6, a7, a9):
      t = a1 @ a2.T
      r = jnp.sum(t * a0, axis=1)
      t14 = jnp.pad(jnp.broadcast_to(r[:, None], (w, w)), ((0, rows_total - w), (0, 0)))
      upd = lax.dynamic_update_slice(a3, a4, (0, 0)) if dyn else a3.at[:k].set(a4)
      head = a3[:k]
      if barrier:
        head = lax.optimization_barrier(head)
      acc = a5 + upd + jnp.pad(head + a6 * a0, ((0, m), (0, 0)))
      qkv = jnp.pad(jnp.concatenate([a7, t14]), ((0, 0), (4, 0)))
      rows = qkv[:rows_total][k:rows_total]
      out1 = acc + jnp.pad(rows @ a9.T, ((k, 0), (0, 0)))
      return jnp.sum(t, axis=0), out1

    rng = np.random.default_rng(0)
    n = lambda *s: rng.standard_normal(s).astype(np.float32)
    args = [n(w), n(w, d), n(w, d), n(rows_total, w), n(k, w), n(rows_total, w), n(k, w),
            n(rows_total, w), n(w, w + 4)]
    return f, args

  cpu, tpu = jax.devices("cpu")[0], jax.devices()[0]
  rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
  out = []
  for c in cases:
    if c.get("issue"):
      rng = np.random.default_rng(0)
      n = lambda *s: rng.standard_normal(s).astype(np.float32)
      f, args = issue_f, [n(2), n(2, 2), n(2, 2), n(6, 2), n(2, 2), n(6, 2), n(2, 2), n(6, 2), n(2, 6)]
    else:
      f, args = make(**c)
    ref = jax.device_get(jax.jit(f)(*jax.device_put(args, cpu)))
    got = jax.device_get(jax.jit(f)(*jax.device_put(args, tpu)))
    with jax.disable_jit():
      eager = jax.device_get(f(*jax.device_put(args, tpu)))
    out.append(dict(case=c, jit_rel=rel(got[1], ref[1]), eager_rel=rel(eager[1], ref[1]),
                    out0_jit_rel=rel(got[0], ref[0])))
  so = [l.split()[-1] for l in open("/proc/self/maps") if "libtpu" in l and l.strip().endswith(".so")]
  build = ""
  if so:
    raw = open(so[0], "rb").read()
    i = raw.find(b"Built on ")
    if i >= 0:
      build = raw[i:i + 200].split(b"\0")[0].decode(errors="replace").replace("\n", " ").strip()
      j = raw.find(b"changelist ", i)
      if j >= 0:
        build += " | " + raw[j:j + 160].split(b"\0")[0].decode(errors="replace").split(" in a ")[0].strip()
  import importlib.metadata as md
  try:
    pip_v = md.version("libtpu")
  except md.PackageNotFoundError:
    pip_v = None
  return dict(jax=jax.__version__, device_kind=tpu.device_kind, libtpu_pip=pip_v, libtpu_so=so[:1],
              libtpu_build=build, libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""), results=out)


def _cases(which):
  c = []
  if "issue" in which:
    c.append({"issue": True})
    c.append({"dyn": True})
  if "grid" in which:
    c += [{"k": k, "m": m} for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16) for m in (1, 2, 3, 4, 5, 6, 8, 12, 16)]
  if "square" in which:
    c += [{"k": k, "m": m, "w": k} for k in (1, 3, 4, 5, 8, 16) for m in (4, 8, 12)]
    c += [{"k": 2, "m": 4, "d": 8}, {"k": 2, "m": 4, "d": 128}]
  bad = [{"k": 2, "m": m} for m in (3, 4, 6, 8, 16)] + [{"k": 4, "m": m, "w": 4} for m in (4, 8, 12)]
  if "barrier" in which:
    c += [dict(b, barrier=True) for b in bad + [{"dyn": True}]]
  if "align8" in which:
    c += [{"k": k, "m": m} for k in (8, 16) for m in (3, 4, 6, 8, 16)]
    c += [{"k": k, "m": m, "w": 4} for k in (8, 16) for m in (4, 8, 12)]
  return c


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--libtpu", action="append", default=None, help="default 或 pip --target 装出的目录；可重复")
  ap.add_argument("--flags", action="append", default=None, help="LIBTPU_INIT_ARGS 的一种取值，写成 --flags=<值>；可重复，--flags= 表示不加")
  ap.add_argument("--cases", default="issue,grid,square,barrier,align8")
  ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  a = ap.parse_args()
  if a.child:
    print("PROBE_JSON " + json.dumps(_child(json.loads(a.cases))), flush=True)
    return
  runs = []
  for lt in a.libtpu or ["default"]:
    for fl in a.flags if a.flags is not None else [""]:
      env = dict(os.environ, LIBTPU_INIT_ARGS=fl, JAX_PLATFORMS="tpu,cpu")
      if lt != "default":
        env["PYTHONPATH"] = lt + os.pathsep + env.get("PYTHONPATH", "")
        env["TPU_LIBRARY_PATH"] = os.path.join(lt, "libtpu", "libtpu.so")
      cases = _cases(a.cases.split(",")) if fl == "" else _cases(["issue", "grid", "square"])
      p = subprocess.run([sys.executable, __file__, "--out", "-", "--child", "--cases", json.dumps(cases)],
                         env=env, capture_output=True, text=True)
      line = [l for l in p.stdout.splitlines() if l.startswith("PROBE_JSON ")]
      if not line:
        runs.append(dict(libtpu=lt, flags=fl, error=(p.stderr or p.stdout)[-2000:]))
        print(f"✗ {lt} {fl!r}：子进程没给结果", file=sys.stderr)
        continue
      r = json.loads(line[0][len("PROBE_JSON "):])
      r.update(libtpu_dir=lt)
      n_bad = sum(x["jit_rel"] > BAD_SIGNAL for x in r["results"])
      r["n_cases"], r["n_jit_wrong"] = len(r["results"]), n_bad
      r["n_eager_wrong"] = sum(x["eager_rel"] > BAD_SIGNAL for x in r["results"])
      runs.append(r)
      print(f"{r['libtpu_pip']} {fl!r}: {n_bad}/{len(r['results'])} 个用例 jit 算错；"
            f"逐算子算错 {r['n_eager_wrong']}", flush=True)
  json.dump(dict(issue="一次实际排查", bad_signal=BAD_SIGNAL, runs=runs),
            open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
  main()
