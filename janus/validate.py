"""`janus validate`：只校验检查项目录，不读运行目录、不跑检查。

写目录、改目录之后先跑它：YAML 能不能读、`include` / `vars` / `foreach` 展开后是什么、载入时的六条校验规则
（docs/reference/catalog-schema.md「校验规则」第 1–6 条）过不过。规则与 ``janus run`` 开跑前用的是同一份
（``janus.catalog.load``），这里过了，``janus run`` 就不会因为目录不合规而退出码 2。

``--all <目录>`` 递归校验下面所有 ``catalogs/*.yaml``；被别的目录 ``include`` 的片段文件（如
``regression-common.yaml``）本身不是完整目录，跳过，由引用它的目录一并校验。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

from janus.catalog import CatalogError, load


def fragments(paths: list[Path]) -> set[Path]:
    """这些目录文件里被 include 的片段（解析成绝对路径）。读不了的文件跳过，由校验本身报错。"""
    out = set()
    for p in paths:
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(d, dict):
            out |= {(p.parent / inc).resolve() for inc in d.get("include") or []}
    return out


def collect(root: Path) -> list[Path]:
    """root 下所有 catalogs/*.yaml，去掉被 include 的片段。"""
    found = sorted(p for p in root.rglob("catalogs/*.yaml") if p.is_file())
    frags = fragments(found)
    return [p for p in found if p.resolve() not in frags]


def check(path: Path) -> tuple[bool, str]:
    """校验一份目录，返回 (是否通过, 一行说明)。"""
    try:
        cat = load(path)
    except CatalogError as e:
        return False, f"FAIL {path}：{e}"
    except OSError as e:
        return False, f"FAIL {path}：读不了：{e}"
    levels = Counter(c["level"] for c in cat.checks)
    by = "、".join(f"{lv} {n}" for lv, n in sorted(levels.items()))
    return True, f"ok   {path}：{len(cat.checks)} 条检查（{by}）"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="janus validate", description="只校验检查项目录（不读运行目录、不跑检查）。")
    ap.add_argument("catalogs", nargs="*", type=Path, help="目录文件")
    ap.add_argument(
        "--all", type=Path, metavar="文件夹", help="递归校验这个文件夹下所有 catalogs/*.yaml（跳过被 include 的片段）"
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="只打印不通过的")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    paths = list(a.catalogs) + (collect(a.all) if a.all else [])
    if not paths:
        ap.print_usage(sys.stderr)
        print("janus validate：没有要校验的目录文件", file=sys.stderr)
        return 2
    bad = 0
    for p in paths:
        ok, line = check(p)
        bad += not ok
        if not ok or not a.quiet:
            print(line, file=sys.stdout if ok else sys.stderr)
    print(f"{len(paths) - bad} / {len(paths)} 通过")
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
