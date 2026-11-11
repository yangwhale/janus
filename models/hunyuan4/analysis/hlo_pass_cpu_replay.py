"""把 TPU 编译时每个 pass 之后的 HLO 拿到 CPU 上执行，找第一个改坏语义的 pass（案例 16 的定位方法）。

TPU 上 jit 结果错、逐算子执行对时，先问：是哪个 HLO pass 把程序改错了，还是 HLO 一直是对的、错在更底层的代码生成。
做法：编译时加 `--xla_dump_to=<目录> --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text`，得到每个 pass 之后的
HLO 文本；本脚本把每一份改写成 CPU 能编的形式，用同一组输入在 CPU 上执行，和 `before_optimizations` 在 CPU 上
的结果比。第一份结果变了的，就是改坏语义的 pass；全都对，说明 HLO 层面没错，问题在 HLO 之下（融合的代码生成、
缓冲分配等），CPU 回放看不到。

改写（只为让 CPU 能编，不改语义）：
  - 去掉 TPU 的布局与分块注解（`{1,0:T(8,128)S(1)}` 之类）、metadata、backend_config；
  - `copy-start` / `copy-done`（HBM 与片上内存之间的异步拷贝）改成 `copy`；
  - fusion 内联成普通指令（含嵌套），`%x#k` 的元组分量写法换成对应指令。
限制：
  - 去掉布局后，改变物理布局的 `bitcast` 会被当成同形状重解释；遇到「转置式」bitcast 的程序结果不可信（#71 里没有）；
  - 含集合通信、custom call、自定义 kernel 的程序跑不了（报 ERR）；
  - 单进程会被 XLA 的 CHECK 直接杀掉，所以每份 HLO 在子进程里跑。

用法：
  python hlo_pass_cpu_replay.py --dump <dump 目录> --module 'module_0000.jit_f.cl_*' --issue71 --out replay.json
  python hlo_pass_cpu_replay.py --dump <dump 目录> --module '<模块名前缀>' --inputs inputs.npz --out replay.json
  inputs.npz 里按 arg0、arg1… 存实参（顺序同 ENTRY 的参数）。
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

# ---------- 改写 ----------

_INS = re.compile(r"^(\s*)(ROOT )?%(\S+) = (.*?)\s([a-z][a-z0-9\-]*)\((.*)$")
_HDR = re.compile(r"^(ENTRY )?%(\S+) \(.*\{\s*$")


def _split_ops(rest):
  depth = 1
  for i, ch in enumerate(rest):
    if ch == "(":
      depth += 1
    elif ch == ")":
      depth -= 1
      if depth == 0:
        return rest[:i], rest[i + 1:]
  raise ValueError(rest[:200])


def _clean(text):
  t = re.sub(r"\{[0-9,]*:[^{}]*(\{[^{}]*\})?[^{}]*\}", "", text)
  t = re.sub(r", metadata=\{[^}]*\}", "", t)
  t = re.sub(r", backend_config=\{.*?\}(?=\n|, |$)", "", t)
  t = re.sub(r"^(HloModule [^,\n]*).*$", r"\1", t, flags=re.M)
  lines = t.split("\n")
  first = next(k for k, l in enumerate(lines) if l.startswith("%") or l.startswith("ENTRY"))
  t = lines[0] + "\n\n" + "\n".join(lines[first:])
  t = re.sub(r"^(\s*%\S+ = )\((\w+\[[\d,]*\]), (\w+\[[\d,]*\]), u32\[\]\) copy-start\((%[^)]+)\).*$",
             r"\1\3 copy(\4)", t, flags=re.M)
  t = t.replace("copy-done(", "copy(")
  t = re.sub(r", cross_program_prefetch_index=\d+", "", t)
  return t


def _parse(text):
  comps, order, cur, entry = {}, [], None, None
  for line in text.split("\n"):
    m = _HDR.match(line)
    if m:
      cur = m.group(2)
      comps[cur] = []
      order.append(cur)
      if m.group(1):
        entry = cur
      continue
    if line.strip() == "}":
      cur = None
      continue
    if cur is not None and line.strip():
      comps[cur].append(line)
  return comps, order, entry


def _instr(line):
  _, root, name, shape, op, rest = _INS.match(line).groups()
  ops, tail = _split_ops(rest)
  return dict(root=bool(root), name=name, shape=shape, op=op, ops=ops, tail=tail)


def _inline(text):
  comps, order, entry = _parse(text)
  fused = set()
  for c in order:
    for line in comps[c]:
      m = re.search(r" fusion\(.*calls=%([\w.\-]+)", line)
      if m:
        fused.add(m.group(1))
  tuples = {}

  def subst(ops, mp):
    ops = re.sub(r"%([\w.\-]+)#(\d+)", lambda m: "%" + tuples[mp.get(m.group(1), m.group(1))][int(m.group(2))], ops)
    return re.sub(r"%([\w.\-]+)", lambda m: "%" + mp.get(m.group(1), m.group(1)), ops)

  def expand(cname, args, out, prefix, rootname):
    mp = {}
    for line in comps[cname]:
      d = _instr(line)
      if d["op"] == "parameter":
        mp[d["name"]] = args[int(d["ops"])]
        continue
      new = rootname if d["root"] else prefix + d["name"]
      ops = subst(d["ops"], mp)
      if d["op"] == "fusion":
        callee = re.search(r"calls=%([\w.\-]+)", d["tail"]).group(1)
        expand(callee, re.findall(r"%([\w.\-]+)", ops), out, new + "__", new)
      else:
        out.append(dict(d, name=new, ops=ops))
        if d["op"] == "tuple":
          tuples[new] = re.findall(r"%([\w.\-]+)", ops)
      mp[d["name"]] = new

  emitted, rootname = [], None
  for line in comps[entry]:
    if _instr(line)["root"]:
      rootname = _instr(line)["name"]
  for line in comps[entry]:
    d = _instr(line)
    if d["op"] == "fusion":
      callee = re.search(r"calls=%([\w.\-]+)", d["tail"]).group(1)
      expand(callee, re.findall(r"%([\w.\-]+)", subst(d["ops"], {})), emitted, d["name"] + "__", d["name"])
    else:
      emitted.append(dict(d, ops=subst(d["ops"], {})))
      if d["op"] == "tuple":
        tuples[d["name"]] = re.findall(r"%([\w.\-]+)", d["ops"])
  lines = text.split("\n")
  out, keep = [lines[0], ""], False
  for line in lines[1:]:
    m = _HDR.match(line)
    if m:
      keep = m.group(2) not in fused and m.group(2) != entry
      if keep:
        out.append(line)
      continue
    if keep:
      out.append(line)
      if line.strip() == "}":
        keep = False
  out.append([l for l in lines if l.startswith("ENTRY ")][0])
  for d in emitted:
    out.append(f"  {'ROOT ' if d['name'] == rootname else ''}%{d['name']} = {d['shape']} {d['op']}({d['ops']}){d['tail']}")
  out.append("}")
  return "\n".join(out)


def to_cpu_text(text):
  t = _clean(text)
  return _inline(t) if " fusion(" in t else t


# ---------- 执行 ----------

def _issue71_inputs():
  import numpy as np
  rng = np.random.default_rng(0)
  n = lambda *s: rng.standard_normal(s).astype(np.float32)
  return [n(2), n(2, 2), n(2, 2), n(6, 2), n(2, 2), n(6, 2), n(2, 2), n(6, 2), n(2, 6)]


def _run_one(path, inputs):
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  import jax
  import jax.extend  # noqa: F401
  import numpy as np
  from jax._src.lib import xla_client as xc
  m = xc.hlo.hlo_module_from_text(to_cpu_text(open(path).read()))
  mlir = xc._xla.mlir.hlo_to_stablehlo(m.as_serialized_hlo_module_proto())
  be = jax.extend.backend.get_backend("cpu")
  exe = be.compile_and_load(mlir, be.devices()[:1], xc.CompileOptions())
  outs = exe.execute([jax.device_put(a, be.devices()[0]) for a in inputs])
  return [np.asarray(o) for o in outs]


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--dump")
  ap.add_argument("--module", help="dump 目录里模块文件名的前缀（可带通配），如 'module_0000.jit_f.cl_*'")
  ap.add_argument("--inputs", help="npz，键 arg0、arg1…")
  ap.add_argument("--issue71", action="store_true", help="用 #71 最小复现的输入")
  ap.add_argument("--out")
  ap.add_argument("--one", help=argparse.SUPPRESS)
  a = ap.parse_args()
  import numpy as np
  if a.issue71:
    inputs = _issue71_inputs()
  else:
    z = np.load(a.inputs)
    inputs = [z[f"arg{i}"] for i in range(len(z.files))]
  if a.one:
    outs = _run_one(a.one, inputs)
    np.savez(a.one + ".cpu_out.npz", *outs)
    print("REPLAY_OK")
    return
  before = sorted(glob.glob(os.path.join(a.dump, a.module + ".before_optimizations.txt")))
  passes = sorted(p for p in glob.glob(os.path.join(a.dump, a.module + ".[0-9][0-9][0-9][0-9].*.txt")))
  final = sorted(glob.glob(os.path.join(a.dump, a.module + ".after_optimizations.txt")))
  if len(before) != 1:
    sys.exit(f"--module 要恰好匹配一个模块，before_optimizations 匹配到 {len(before)} 个")
  rows, ref = [], None
  for path in before + passes + final:
    p = subprocess.run([sys.executable, __file__, "--one", path] + (["--issue71"] if a.issue71 else ["--inputs", a.inputs]),
                       capture_output=True, text=True)
    name = os.path.basename(path)
    if "REPLAY_OK" not in p.stdout:
      err = [l for l in (p.stderr + p.stdout).splitlines() if "rror" in l or "Check failed" in l]
      rows.append(dict(file=name, status="ERR", detail=(err[-1] if err else "子进程失败")[:300]))
      print("ERR ", name[:120])
      continue
    z = np.load(path + ".cpu_out.npz")
    outs = [z[k] for k in sorted(z.files, key=lambda k: int(k.split("_")[1]))]
    os.remove(path + ".cpu_out.npz")
    if ref is None:
      ref = outs
    rel = [float(np.max(np.abs(o - r)) / max(float(np.max(np.abs(r))), 1e-30)) if o.shape == r.shape else None
           for o, r in zip(outs, ref)]
    rows.append(dict(file=name, status="OK", max_rel_vs_before=rel))
    print(f"{max(x for x in rel if x is not None):.2e} {name[:120]}")
  if a.out:
    first_bad = next((r["file"] for r in rows if r["status"] == "OK" and max(x for x in r["max_rel_vs_before"] if x is not None) > 1e-5), None)
    json.dump(dict(dump=os.path.basename(os.path.normpath(a.dump)), module=a.module, n_files=len(rows),
                   n_err=sum(r["status"] == "ERR" for r in rows), first_changed=first_bad, rows=rows),
              open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
  main()
