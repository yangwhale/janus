"""仓库卫生：不进内部链接与真实桶名（仓库公开，见 SECURITY.md）、不进人名与内部过程用语、不引用没公开的仓库，包内模块都有说明。

模式按片段拼起来写，免得本文件自己命中。
"""

import ast
import re
import subprocess

from helpers import ROOT  # noqa: E402  仓库根

TEXT_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml", ".toml", ".txt", ".json", ".jsonl", ".html", ".cfg", ""}

INTERNAL = [
    re.compile(r"(?<![\w.-])" + "go" + r"/[A-Za-z0-9_-]"),  # 内部短链
    re.compile("google" + "plex", re.I),
    re.compile(r"\b" + "corp" + r"\.google\b", re.I),
    re.compile(r"\b" + "cc" + r"\.higcp\.com\b", re.I),
]
# 集群特有的提交设置（队列、预留、归属注解、我们自己的集群与节点池名）只放在仓库外的站点配置里（JANUS_SITE_PROFILE）
SITE_SPECIFIC = [
    re.compile("kueue" + r"\.x-k8s\.io/"),
    re.compile("ownership" + r"\.tpu\."),
    re.compile("reservation" + "-name"),
    re.compile("shared" + r"-v(5p|7x)\b"),
    re.compile("shared" + "-capacity"),
    re.compile("boda" + "borg"),
    re.compile("launch" + "pad", re.I),
    re.compile("janus" + r"-v5p-8"),
    re.compile(r"/tmp/" + r"kc-"),
    re.compile("共享" + r"(集群|容量| ?v5p| ?v7x)"),
]
# 文档不写个人标识与内部过程用语：谁拍的板写「项目负责人」或直接写规则；不提 AI 助手；不用 issue 的内部简写。
# AGENTS.md 本来就是写给 AI 助手的，LICENSE / NOTICE 的版权行写版权人姓名，都不查。
PROCESS = [
    re.compile(r"\b" + "Chr" + r"is\b"),
    re.compile(r"\b[Aa]" + "gent" + r"s?\b"),
    re.compile(r"\b" + "PM" + r"#\d"),
]
PROCESS_SKIP = {"AGENTS.md", "LICENSE", "NOTICE"}  # 许可证的版权行写版权人姓名
# 公开版不引用没公开的仓库、它们的 issue / PR 号与评论链接，也不写协作关系（见 docs/public-edition.md）。
PRIVATE_REFS = [
    re.compile("tomur" + "top", re.I),
    re.compile("project" + "-management"),
    re.compile(r"(?<![\w/.-])(?:maxtext|janus)" + r"\s?#\d"),
    re.compile(r"\bMT" + r"#\d"),
    re.compile("issue" + "comment"),
    re.compile("@" + r"google\.com"),
    re.compile("客" + "户(?!端)"),
    re.compile("腾" + "讯"),
    re.compile("蚂" + "蚁"),
    re.compile("原" + "话"),
]
# 本机绝对路径会带出用户名（结果文件里的目录路径等），一律写成仓库相对路径（janus.catalog.display_path）
LOCAL_ABS_PATH = re.compile(r"(?<![\w.])/(" + "ho" + "me" + "|" + "Us" + "ers" + r")/[A-Za-z_][\w.-]*/")
# 桶名只许写占位：gs://<POC 桶>/…、gs://…、gs://$VAR、gs://{…}。紧跟字母数字就是写了真名。
REAL_BUCKET = re.compile("gs" + r"://[A-Za-z0-9]")
# 合并冲突标记：解冲突漏掉一块时，make check 别的检查都不报，读者会直接看到两版原文
CONFLICT_MARKER = re.compile(r"^(" + "<" * 7 + "|" + ">" * 7 + r")( |$)|^" + "=" * 7 + "$")


def tracked_text_files():
    # 已跟踪的加上还没 add、也没被忽略的新文件：新结果文件常常在 add 之前跑检查，只看已跟踪的会漏掉
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True
    )
    if out.returncode != 0:  # 不在 git 工作区里（例如解压的源码包）：退回遍历
        files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    else:
        files = out.stdout.splitlines()
    for rel in files:
        p = ROOT / rel
        if p.suffix in TEXT_SUFFIXES and p.is_file() and p.stat().st_size < 5_000_000:
            yield rel, p


def scan(patterns):
    bad = []
    for rel, p in tracked_text_files():
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for pat in patterns:
                if pat.search(line):
                    bad.append(f"{rel}:{n}: {line.strip()[:120]}")
    return bad


def test_no_internal_links():
    bad = scan(INTERNAL)
    assert not bad, "仓库里不能有内部链接：\n" + "\n".join(bad)


def test_no_real_bucket_names():
    bad = scan([REAL_BUCKET])
    assert not bad, "桶名写成占位 gs://<POC 桶>/…：\n" + "\n".join(bad)


def test_no_site_specific_submission_settings():
    bad = scan(SITE_SPECIFIC)
    assert not bad, "集群特有的提交设置放在仓库外的站点配置里（JANUS_SITE_PROFILE），不进仓库：\n" + "\n".join(bad)


def test_no_personal_names_or_process_chatter():
    bad = [b for b in scan(PROCESS) if b.split(":", 1)[0] not in PROCESS_SKIP]
    assert not bad, (
        "文档里不写人名、不提 AI 助手、issue 不用内部简写（改成「项目负责人」、直接写规则）：\n" + "\n".join(bad)
    )


def test_no_private_repo_refs():
    bad = scan(PRIVATE_REFS)
    assert not bad, "公开版不引用没公开的仓库与它们的 issue / PR，不写协作关系：\n" + "\n".join(bad)


def test_no_local_absolute_paths():
    bad = scan([LOCAL_ABS_PATH])
    assert not bad, "不写本机绝对路径（会带出用户名），改成仓库相对路径：\n" + "\n".join(bad)


def test_no_merge_conflict_markers():
    bad = scan([CONFLICT_MARKER])
    assert not bad, "解冲突漏了，文件里还留着冲突标记：\n" + "\n".join(bad)


def test_patterns_catch_examples():
    """工具先自证：故意写的反例都要被抓到，占位写法不被误报。"""
    assert any(p.search("见 " + "go" + "/foo") for p in INTERNAL)
    assert not any(p.search("algo/foo、cargo/x") for p in INTERNAL)
    assert all(any(p.search(x) for p in PROCESS) for x in ("Chr" + "is 定", "根因调查 a" + "gent", "P" + "M#61"))
    assert not any(p.search(x) for p in PROCESS for x in ("见 AGENTS.md", "Christmas", "pm-check"))
    for x in ("tomur" + "top/maxtext", "见 maxtext" + "#108", "janus" + "#343", "客" + "户决定"):
        assert any(p.search(x) for p in PRIVATE_REFS), x
    for x in ("分布式客" + "户端", "AI-Hypercomputer/maxtext#4625", "tencent/Hy3", "台账 #20"):
        assert not any(p.search(x) for p in PRIVATE_REFS), x
    assert LOCAL_ABS_PATH.search("/" + "home/someone/work/x.yaml")
    assert LOCAL_ABS_PATH.search('"path": "/' + 'Users/someone/x"')
    assert not LOCAL_ABS_PATH.search("models/hunyuan4/x.yaml、~/work/x、$HOME/x")
    assert REAL_BUCKET.search("gs" + "://my-bucket/x")
    assert any(p.search("kueue" + ".x-k8s.io/queue-name: q") for p in SITE_SPECIFIC)
    assert any(p.search("在共享" + " v5p 上跑") for p in SITE_SPECIFIC)
    assert not any(p.search("共享专家、Kueue、JobSet、gke-nodepool") for p in SITE_SPECIFIC)
    for ok in ("gs" + "://<POC 桶>/a", "gs" + "://…/a", "gs" + "://$BUCKET/a", "gs" + "://{a,b}"):
        assert not REAL_BUCKET.search(ok), ok


def test_every_janus_module_has_docstring():
    missing = [
        p.relative_to(ROOT).as_posix()
        for p in sorted((ROOT / "janus").rglob("*.py"))
        if p.stat().st_size and not ast.get_docstring(ast.parse(p.read_text(encoding="utf-8")))
    ]
    assert not missing, "这些模块缺开头的说明（写它做什么、规格在哪）：\n" + "\n".join(missing)
