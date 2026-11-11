"""一次性迁移：已提交结果里的目录指纹从「原文件 sha256」换成内容指纹（content-v1，见 janus/catalog.py）。

对每份 `janus run` 写的 `*.results.json`（catalog.path 是 .yaml / .yml），对 catalog 与 includes 里的每份文件：

1. 在 git 历史里找原文件 sha256 等于结果里记的那一版（跑的时候用的那份目录）；
2. 算那一版的内容指纹，和仓库里现在那份文件的内容指纹比；
3. 相等（中间只改了注释、description 之类的说明文字）才写进结果：
   `{path, hash_algo: content-v1, content_sha256, raw_sha256}`，`raw_sha256` 就是原来记的 `sha256`；schema_version 改成 2。
   不相等（检查真的变了）或历史里找不到那一版：不改这份结果，记进日志的「没迁移」一节。

另外给 `models/*/regression/lkg.yaml` 的每条登记补 `catalog_content_sha256`（R02 改比它）：每个原文件 sha256
在历史里找到那一版、算它的内容指纹。登记的是当时那版目录，不和现在的比；找不到的不补，记进日志。

独立脚本写的结果（catalog.path 是 .py）不动：它们记的是脚本文件本身的 sha256。

  python tools/migrate_catalog_content_hash.py            # 只看：打印日志，不改文件
  python tools/migrate_catalog_content_hash.py --write --log tools/migrate_catalog_content_hash.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from janus.catalog import HASH_ALGO, content_sha256  # noqa: E402


def git(*args: str, text: bool = True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True, text=text).stdout


def history_index() -> dict[str, tuple[str, bytes]]:
    """git 历史里所有 .yaml / .yml 文件的每一版：原文件 sha256 → (blob 号, 内容)。"""
    blobs = {}
    for line in git("rev-list", "--objects", "--all").splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].endswith((".yaml", ".yml")):
            blobs.setdefault(parts[0], parts[1])
    if not blobs:
        return {}
    out = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        input="\n".join(blobs).encode() + b"\n",
        capture_output=True,
        check=True,
    ).stdout
    idx, pos = {}, 0
    while pos < len(out):
        nl = out.index(b"\n", pos)
        sha, kind, size = out[pos:nl].decode().split()
        body = out[nl + 1 : nl + 1 + int(size)]
        pos = nl + 1 + int(size) + 1
        if kind == "blob":
            idx.setdefault(hashlib.sha256(body).hexdigest(), (sha, body))
    return idx


def first_commit_with(blob: str) -> str:
    """最早引入这个 blob 的提交（日志里写「跑的是哪一版」）。"""
    out = git("log", "--all", "--reverse", "--format=%h", f"--find-object={blob}").split()
    return out[0] if out else "?"


def current_file(result_path: Path, catalog_path: str) -> Path | None:
    """同 tests/test_committed_results.py：路径在仓库里就用它（仓库外的不算），否则到同一实践的 catalogs/ 下
    先按文件名、再按结果名（<目录名>.results.json）找。"""
    p = ROOT / catalog_path
    if p.is_file() and p.resolve().is_relative_to(ROOT.resolve()):
        return p
    for alt in (
        result_path.parent.parent / "catalogs" / p.name,
        result_path.parent.parent / "catalogs" / (result_path.name.split(".")[0] + p.suffix),
    ):
        if alt.is_file():
            return alt
    return None


def migrate_result(f: Path, idx: dict, write: bool, log: list[str]) -> str:
    """返回 migrated / already / failed / skipped。"""
    rel = f.relative_to(ROOT).as_posix()
    text = f.read_text()
    d = json.loads(text)
    cat = d.get("catalog") or {}
    if not str(cat.get("path", "")).endswith((".yaml", ".yml")):
        return "skipped"
    if cat.get("hash_algo") == HASH_ALGO:
        return "already"
    entries = [cat, *(cat.get("includes") or [])]
    new_entries, problems, notes = [], [], []
    for e in entries:
        raw = e["sha256"]
        cur = current_file(f, e["path"])
        hit = idx.get(raw)
        if cur is None:
            problems.append(f"{e['path']}：仓库里找不到这份目录")
            continue
        if hit is None:
            problems.append(f"{e['path']}：git 历史里没有原文件 sha256 = {raw[:12]}… 的版本")
            continue
        old_c = content_sha256(hit[1])
        now_c = content_sha256(cur)
        if old_c != now_c:
            problems.append(
                f"{e['path']}：跑的那一版（{first_commit_with(hit[0])}，sha256 {raw[:12]}…）内容指纹 "
                f"{old_c[:12]}… ≠ 现在 {cur.relative_to(ROOT).as_posix()} 的 {now_c[:12]}…（检查真的变了）"
            )
            continue
        same = (
            "原文件也相同"
            if hashlib.sha256(cur.read_bytes()).hexdigest() == raw
            else (f"原文件不同、只差说明文字（跑的那一版 {first_commit_with(hit[0])}）")
        )
        notes.append(f"{Path(e['path']).name}：{same}")
        new_entries.append({"path": e["path"], "hash_algo": HASH_ALGO, "content_sha256": now_c, "raw_sha256": raw})
    if problems:
        log.append(f"- 没迁移 `{rel}`：" + "；".join(problems))
        return "failed"
    new_cat = dict(new_entries[0])
    if cat.get("includes"):
        new_cat["includes"] = new_entries[1:]
    for k, v in cat.items():  # 将来若有别的键，原样留着
        if k not in ("path", "sha256", "includes"):
            new_cat[k] = v
    out = {}
    for k, v in d.items():
        out[k] = 2 if k == "schema_version" else new_cat if k == "catalog" else v
    log.append(f"- 迁移 `{rel}`：" + "；".join(notes))
    if write:
        f.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return "migrated"


LKG_BLOCK = re.compile(r"^(?P<ind>\s*)catalog_sha256:\n(?P<body>(?:(?P=ind)\s+\S+: [0-9a-f]{64}\n)+)", re.M)


def migrate_lkg(p: Path, idx: dict, write: bool, log: list[str]) -> tuple[int, int]:
    """lkg.yaml 每个 catalog_sha256 块后面补一个 catalog_content_sha256 块（按文本插入，注释与排版不动）。"""
    text = p.read_text()
    rel = p.relative_to(ROOT).as_posix()
    if "catalog_content_sha256:" in text:
        log.append(f"- `{rel}`：已有 catalog_content_sha256，跳过")
        return 0, 0
    ok = bad = 0
    pieces, last = [], 0
    for m in LKG_BLOCK.finditer(text):
        ind, body = m.group("ind"), m.group("body")
        lines, missing = [], []
        for ln in body.splitlines():
            name, raw = ln.strip().split(": ")
            hit = idx.get(raw)
            if hit is None:
                missing.append(f"{name} {raw[:12]}…")
                continue
            lines.append(ln.replace(raw, content_sha256(hit[1])))
        pieces.append(text[last : m.end()])
        last = m.end()
        line_no = text.count("\n", 0, m.start()) + 1
        if missing:
            bad += 1
            log.append(f"- `{rel}` 第 {line_no} 行的登记没补：历史里找不到 {missing}")
            continue
        ok += 1
        pieces.append(f"{ind}catalog_content_sha256:\n" + "\n".join(lines) + "\n")
    pieces.append(text[last:])
    if write and ok:
        p.write_text("".join(pieces))
    log.append(f"- `{rel}`：{ok} 条登记补了 catalog_content_sha256" + (f"，{bad} 条没补" if bad else ""))
    return ok, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="真的改文件（不给只打印日志）")
    ap.add_argument("--log", type=Path, help="日志另写一份到这个文件")
    a = ap.parse_args(argv)
    idx = history_index()
    log = [
        f"# 目录指纹迁移到 {HASH_ALGO}",
        "",
        f"仓库提交 {git('rev-parse', '--short', 'HEAD').strip()}；历史里 {len(idx)} 个不同的 YAML 版本。",
        "",
        "## 结果文件",
        "",
    ]
    counts = {"migrated": 0, "already": 0, "failed": 0, "skipped": 0}
    for f in sorted(ROOT.glob("models/**/*.results.json")):
        counts[migrate_result(f, idx, a.write, log)] += 1
    log += [
        "",
        f"迁移 {counts['migrated']}，没迁移 {counts['failed']}，本来就是 {HASH_ALGO} {counts['already']}，"
        f"不是目录写的（独立脚本或没有 catalog）{counts['skipped']}。",
        "",
        "## LKG 登记",
        "",
    ]
    for p in sorted(ROOT.glob("models/*/regression/lkg.yaml")):
        migrate_lkg(p, idx, a.write, log)
    text = "\n".join(log) + "\n"
    print(text, end="")
    if a.log:
        a.log.write_text(text)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
