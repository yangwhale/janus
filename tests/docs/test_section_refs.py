"""文档里「见 某文件「某节」」这种引用，要指到目标文件里真实存在的位置（改标题、删一节后防失联）。

断链检查（test_links.py）只管文件在不在；这里管文件里那一节还在不在。认的写法：一个 .md 路径（链接、反引号或裸写）
后面紧跟（可隔「的 / 里 / 中」）「标题」。「A · B」表示 A 下面的 B，两段分别核。

「标题」算存在，只要它是目标文件里下面任何一处的子串：
- 一个 Markdown 标题；
- 一段加粗 ``**…**``；
- 表格一行的前两格（「某某」行）；
- 列表项开头冒号前的引导语（``- 和 GPU 抓数的不同：``）。

不查：围栏代码块里的文字、CHANGELOG 与路径变更记录（记的是旧名字）、实践的 ``results/``（生成时的快照）、标题含「路径对照」的节。
"""

import re
from pathlib import Path

from helpers import ROOT

SKIP_FILES = {"CHANGELOG.md", "docs/dev/CHANGES-PATHS.md"}
REF = re.compile(
    r"(?:\[[^\]]*\]\(([^)#\s]+\.md)[^)]*\)"  # [文字](路径.md)
    r"|`([\w./-]*[\w-]+\.md)`"  # `路径.md`
    r"|(?<![\w/`])((?:\.\./|[\w-]+/)*[\w.-]+\.md)(?![\w`]))"  # 裸写的 路径.md
    r"\s*(?:的|里|中)?\s*「([^」]{2,60})」"
)
HEADING = re.compile(r"#{1,6} ")
LEAD = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(.+?)[：:]")


def md_files(root):
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root).as_posix()
        if rel in SKIP_FILES or "/results/" in rel or any(part.startswith(".") for part in p.parts[len(root.parts) :]):
            continue
        if any(part in {"node_modules", "__pycache__"} or part.endswith(".egg-info") for part in p.parts):
            continue
        yield p


def anchors(path: Path) -> list[str]:
    """目标文件里能被「…」引用的位置：标题、加粗、表格前两格、列表引导语。"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.replace("`", "")
        if HEADING.match(s):
            out.append(HEADING.sub("", s).strip())
        out += re.findall(r"\*\*([^*]+)\*\*", s)
        if s.lstrip().startswith("|"):
            out += [c.strip().replace("*", "") for c in s.strip().strip("|").split("|")[:2]]
        m = LEAD.match(s)
        if m:
            out.append(m.group(1).replace("*", ""))
    return out


def references(path: Path, root: Path):
    """(行号, 目标文件, 「标题」)。目标路径按 本文件目录 → 各级上层目录 → 仓库根 依次找。"""
    fence, skip = False, False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if HEADING.match(line):
            skip = "路径对照" in line
        if fence or skip:
            continue
        for m in REF.finditer(line):
            target = m.group(1) or m.group(2) or m.group(3)
            cands = [path.parent / target, *(d / target for d in path.parents if root in d.parents or d == root)]
            hit = next((c.resolve() for c in cands if c.is_file()), None)
            if hit is not None:
                yield n, hit, m.group(4).replace("`", "")


def check(root: Path) -> list[str]:
    bad, cache = [], {}
    for f in md_files(root):
        for n, target, title in references(f, root):
            an = cache.setdefault(target, anchors(target))
            for part in title.split(" · "):
                if not any(part in a for a in an):
                    bad.append(
                        f"{f.relative_to(root)}:{n}: 「{title}」在 {target.relative_to(root)} 里找不到（「{part}」）"
                    )
    return bad


def test_section_references_resolve():
    bad = check(ROOT)
    assert not bad, "引用的节不存在（改了标题就同步改引用）：\n" + "\n".join(bad)


def test_checker_catches_missing_section(tmp_path):
    """工具先自证：引用一个不存在的节必须报出来；标题、加粗、表格行、列表引导语、「A · B」都要认。"""
    (tmp_path / "t.md").write_text(
        "# 一、原则\n\n**同程序重跑**逐位。\n\n| 精度 | 路由 GEMM | x |\n|---|---|---|\n\n- 和 GPU 的不同：几条\n",
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text(
        "见 [t](t.md)「原则」、`t.md`「同程序重跑」、t.md 的「路由 GEMM」、t.md「和 GPU 的不同」、t.md「精度 · 路由 GEMM」。\n"
        "坏的：t.md「不存在的节」。\n```\nt.md「代码块里的不算」\n```\n",
        encoding="utf-8",
    )
    bad = check(tmp_path)
    assert len(bad) == 1 and "不存在的节" in bad[0], bad
