"""按现行的程序指纹（tpu_self_compare.hlo_program_fingerprint，规范化同 04-tpu-self/hlo_screen.py）重算已提交 TPU 自比结果里的
hlo.program_fingerprint 与 hlo.program_equal，其余字段不动。只读运行目录 hlo/ 里 TPU 上存下的优化后 HLO，不重跑比较。

    python models/hunyuan4/analysis/refresh_program_fingerprint.py --root <运行目录根> [--write]

不加 --write 只列出会变的文件与 program_equal 的新旧值。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE.parents[0] / "alignments" / "04-tpu-self" / "results"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    import tpu_self_compare as t

    fps: dict[str, str | None] = {}

    def fp(run):
        if run not in fps:
            x = t.hlo_text(str(a.root / run))
            fps[run] = t.hlo_program_fingerprint(x) if x else None
        return fps[run]

    changed, missing = [], []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        h = d.get("hlo") if isinstance(d, dict) else None
        if not isinstance(h, dict) or "program_fingerprint" not in h:
            continue
        ft, fg = fp(d["test"]), fp(d["golden"])
        if ft is None or fg is None:
            missing.append(f.name)
            continue
        old = h.get("program_equal")
        h["program_fingerprint"] = {"test": ft, "golden": fg}
        h["program_equal"] = ft == fg
        if old != h["program_equal"]:
            changed.append((f.name, old, h["program_equal"]))
        if a.write:
            f.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    for name, old, new in changed:
        print(f"{name}: program_equal {old} → {new}")
    print(f"program_equal 变了的 {len(changed)} 份；运行目录里没有 HLO、没重算的 {len(missing)} 份：{missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
