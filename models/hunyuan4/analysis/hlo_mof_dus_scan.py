"""在 TPU 优化后 HLO 里找 一次实际排查 的特征（案例 16）。

特征：一颗多输出 fusion（根是 tuple）里有 dynamic-update-slice，而且 DUS 原地写的那块缓冲（DUS 的第 0 个操作数）
在同一颗 fusion 里还被别的指令读。libtpu 0.0.46、0.0.48、0.0.51、0.0.52（实测；0.0.47、0.0.49 没测）会把「含 DUS 的 loop fusion」和读同一缓冲的兄弟算子
合成这样一颗 fusion，部分形状下 TPU 生成的代码把兄弟输出算错（案例 16）。

读的是 `*.after_optimizations.txt`（XLA_DUMP=1 的运行目录里 `hlo/train_step.after_optimizations.txt`，或
--xla_dump_to 目录下的同名文件）。只做文本匹配，不需要 TPU。

  python hlo_mof_dus_scan.py <after_optimizations.txt> … [--json 输出.json]

输出：每个文件一行「多输出 fusion 含 DUS 的个数 / 其中带 #71 特征的个数」；带特征的逐个列出 fusion 名、DUS 目标与
更新的形状和分块、读目标缓冲的指令。特征数为 0 不等于证明没问题（触发条件没有完全查清，见案例 16「拿不准的」）。
"""

import argparse
import json
import re
import sys

_HDR = re.compile(r"^(ENTRY )?%(\S+) \(.*\{\s*$")
_INS = re.compile(r"^\s*(ROOT )?%(\S+) = (.*?)\s([a-z][a-z0-9\-]*)\((.*)$")
_TILE = re.compile(r"T\((\d+),(\d+)\)")


def parse_computations(text):
  """{计算名: [指令行]}。"""
  comps, cur = {}, None
  for line in text.splitlines():
    m = _HDR.match(line)
    if m:
      cur = m.group(2)
      comps[cur] = []
      continue
    if line.strip() == "}":
      cur = None
      continue
    if cur is not None and line.strip():
      comps[cur].append(line)
  return comps


def _operands(rest):
  depth, buf = 1, []
  for ch in rest:
    if ch == "(":
      depth += 1
    elif ch == ")":
      depth -= 1
      if depth == 0:
        break
    buf.append(ch)
  return re.findall(r"%([\w.\-]+)", "".join(buf))


def _tiles(shape):
  return sorted({f"{a},{b}" for a, b in _TILE.findall(shape or "")})


def scan_text(text):
  comps = parse_computations(text)
  callers = {}
  for lines in comps.values():
    for line in lines:
      m = re.search(r" fusion\(.*calls=%([\w.\-]+)", line)
      if m and _INS.match(line):
        callers.setdefault(m.group(1), []).append(_INS.match(line).group(2))
  hits = []
  for name, lines in comps.items():
    if name not in callers:
      continue
    root = [l for l in lines if l.lstrip().startswith("ROOT ")]
    if not root or " tuple(" not in root[0]:
      continue
    parsed = [(l, _INS.match(l)) for l in lines]
    shape_of = {m.group(2): m.group(3) for _, m in parsed if m}
    for line, m in parsed:
      if not m or m.group(4) != "dynamic-update-slice":
        continue
      ops = _operands(m.group(5))
      base = ops[0] if ops else None
      readers = []
      for other, mo in parsed:
        if other is line or not mo or mo.group(2) == base:
          continue
        if base in _operands(mo.group(5)):
          readers.append(other.strip())
      hits.append(dict(
          fusion=callers[name][0], computation=name, dus=m.group(2),
          target=m.group(3), target_tiles=_tiles(m.group(3)),
          update=shape_of.get(ops[1]) if len(ops) > 1 else None,
          update_tiles=_tiles(shape_of.get(ops[1]) if len(ops) > 1 else None),
          n_outputs=len(_operands(root[0].split(" tuple(", 1)[1])),
          readers_of_target=[re.sub(r", metadata=\{[^}]*\}|, backend_config=.*$", "", r)[:200] for r in readers],
      ))
  return dict(mof_with_dus=len(hits), with_71_signature=sum(bool(h["readers_of_target"]) for h in hits), hits=hits)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("files", nargs="+")
  ap.add_argument("--json")
  a = ap.parse_args()
  out, flagged = {}, 0
  for f in a.files:
    with open(f, errors="replace") as fh:
      r = scan_text(fh.read())
    out[f] = r
    flagged += r["with_71_signature"]
    print(f"{r['mof_with_dus']:4d} {r['with_71_signature']:4d}  {f}")
    for h in r["hits"]:
      if h["readers_of_target"]:
        print(f"      {h['fusion']}: {h['dus']} 目标 {h['target'].split('{')[0]} 分块 {h['target_tiles']}，"
              f"更新 {(h['update'] or '?').split('{')[0]} 分块 {h['update_tiles']}，{h['n_outputs']} 个输出")
        for r_ in h["readers_of_target"][:3]:
          print(f"        读目标缓冲：{r_}")
  if a.json:
    with open(a.json, "w") as fh:
      json.dump(out, fh, indent=1, ensure_ascii=False)
  sys.exit(1 if flagged else 0)


if __name__ == "__main__":
  main()
