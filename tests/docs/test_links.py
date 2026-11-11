"""仓库里所有 Markdown 的相对链接都要指到存在的文件或目录（搬目录后防断链）；带 ``#锚点`` 的要指到目标文件里真实存在的标题。

锚点按 GitHub 的规则从标题生成（中文标题也一样）：去掉 Markdown 记号与 HTML 标签，转小写，删掉字母、数字、``_``、``-``、
空格以外的字符（全角标点也删），空格换成 ``-``；同名标题依次加 ``-1``、``-2``。另认 ``<a id="…">`` / ``<a name="…">``。
围栏代码块、HTML 注释、行内代码里的东西不算链接也不算标题。
"""

import re
import sys
from urllib.parse import unquote

from helpers import ROOT  # noqa: E402  仓库根

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
HTML_ANCHOR = re.compile(r"<a\s+[^>]*?(?:id|name)=\"([^\"]+)\"")


def md_files():
    for p in ROOT.rglob("*.md"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        yield p


def prose_lines(text):
    """(行号, 行) ：跳过围栏代码块与 HTML 注释。"""
    fence = None
    in_comment = False
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if m and not in_comment:
            fence = m.group(1)
            continue
        if in_comment:
            if "-->" in line:
                in_comment = False
                line = line.split("-->", 1)[1]
            else:
                continue
        line = re.sub(r"<!--.*?-->", "", line)
        if "<!--" in line:
            line, in_comment = line.split("<!--", 1)[0], True
        yield n, line


def links(path):
    for n, line in prose_lines(path.read_text(encoding="utf-8")):
        line = re.sub(r"`[^`]*`", "", line)  # 行内代码里的方括号不算链接
        for m in LINK.finditer(line):
            yield n, m.group(1) or m.group(2)


def github_slug(title):
    """GitHub 给标题生成的锚点（不含重名后缀）。"""
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", title)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"</?[A-Za-z][^>]*>", "", t)
    t = t.replace("`", "")
    t = re.sub(r"(\*\*|__|\*|~~)", "", t)
    t = re.sub(r"[^\w\- ]", "", t.strip().lower())
    return t.replace(" ", "-")


_ANCHORS = {}


def anchors(path):
    if path not in _ANCHORS:
        found, seen = set(), {}
        for _, line in prose_lines(path.read_text(encoding="utf-8")):
            found.update(HTML_ANCHOR.findall(line))
            m = HEADING.match(line)
            if m:
                base = github_slug(m.group(2))
                k = seen.get(base, 0)
                seen[base] = k + 1
                found.add(base if k == 0 else f"{base}-{k}")
        _ANCHORS[path] = found
    return _ANCHORS[path]


def broken_links():
    bad = []
    for f in md_files():
        for n, target in links(f):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue  # 外链
            path, _, frag = target.partition("#")
            t = unquote(path)
            if not t:
                dest = f
            else:
                dest = (ROOT / t.lstrip("/")) if t.startswith("/") else (f.parent / t)
            if not dest.exists():
                bad.append(f"{f.relative_to(ROOT)}:{n}: {target}")
            elif frag and dest.is_file() and dest.suffix == ".md" and not re.fullmatch(r"L\d+(-L\d+)?", frag):
                if unquote(frag).lower() not in {a.lower() for a in anchors(dest.resolve())}:
                    bad.append(f"{f.relative_to(ROOT)}:{n}: {target}（锚点不存在）")
    return bad


def test_relative_markdown_links_resolve():
    bad = broken_links()
    assert not bad, "断链：\n" + "\n".join(bad)


def test_github_slug_rules():
    assert github_slug("5.6 签字清单：现在离签字还差什么") == "56-签字清单现在离签字还差什么"
    assert github_slug("1. 起 Pod、把仓库放进去") == "1-起-pod把仓库放进去"
    assert (
        github_slug("5.3 一份检查项目录的结果：report.md 与 results.json")
        == "53-一份检查项目录的结果reportmd-与-resultsjson"
    )
    assert github_slug("2026-11-04 · 渲染模块合进 `janus/report/`（#54）") == "2026-11-04--渲染模块合进-janusreport54"
    assert github_slug("实践 <实践名> · 设置") == "实践-实践名--设置"


def test_link_checker_catches_broken_link(tmp_path, monkeypatch):
    """工具先自证：故意写断链和断锚点，检查必须报出来；好的链接、代码里的链接不误报。"""
    (tmp_path / "a.md").write_text(
        "# 标题\n[ok](b.md) [坏](missing/x.md) `[不算](nope.md)`\n"
        "[好锚点](b.md#第-2-节重名) [第二个同名](b.md#重名-1) [本页](#标题) [坏锚点](b.md#没有这节)\n"
        "```\n[围栏里不算](fence.md)\n```\n<!-- [注释里不算](comment.md) -->\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text("## 第 2 节：重名\n## 重名\n## 重名\n```\n# 不是标题\n```\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    _ANCHORS.clear()
    bad = "\n".join(broken_links())
    _ANCHORS.clear()
    assert "missing/x.md" in bad and "b.md#没有这节" in bad
    for ok in ("nope.md", "fence.md", "comment.md", "b.md#第-2-节重名", "b.md#重名-1", "#标题"):
        assert ok not in bad, ok
