"""已提交的 results.json 记的目录指纹要与仓库里那份文件现在的对得上。

两种产物都查：
- `janus run` 的结果（schema_version 2）：`catalog.content_sha256` 是目录的内容指纹（`hash_algo: content-v1`，
  见 janus/catalog.py）。只改注释或 description、notes、source、pending_note 这些说明文字，指纹不变；
  改了检查本身（选什么、门槛、噪声对、运行、params……）就用同一批运行目录重跑 `janus run`，把新结果一起提交。
- 独立脚本写的结果：`catalog` 填的是脚本自己的路径与原文件 sha256（见 docs/dev/design/0003-l1-op-replay.md）。
  改了脚本，就用当前脚本重跑它写过的结果。

不然拿结果里的指纹去核对，会对不上。

目录或脚本搬过家时，结果里的 `catalog.path` 还是旧路径（见 docs/dev/CHANGES-PATHS.md）。
目录按文件名到同一实践的 `catalogs/` 下找，找不到再到同一模型各实践的 `catalogs/` 下找唯一同名的；脚本按文件名到 `models/` 下找；仓库外的路径不算。
暂时没法重跑的结果写进 HISTORICAL，并写明原因。
"""

import hashlib
import json

import pytest
from helpers import ROOT  # noqa: E402  仓库根

from janus.catalog import HASH_ALGO, content_sha256

# 暂时没法重跑的结果：键是结果文件相对仓库根的路径，值是原因。第一阶段 5 个 GPU 脚本的结果已于 2026-11-06 用当前脚本重跑，清空。
HISTORICAL: dict[str, str] = {}

RESULTS = sorted(p for p in ROOT.glob("models/**/*.results.json"))


def _catalog_file(result_path, catalog_path):
    """catalog.path 在仓库里就用它（仓库外的绝对路径不算，免得结果只在某台机器上对得上）。不在时（搬过家，或跑的时候用了绝对路径）：
    目录到同一实践的 catalogs/ 下按文件名找，再按结果名找（<目录名>.results.json，跑的时候用了临时副本的）；
    脚本到 models/ 下按文件名找，找到唯一的一个才算。"""
    p = ROOT / catalog_path
    if p.is_file() and p.resolve().is_relative_to(ROOT.resolve()):
        return p
    if p.suffix == ".py":
        cands = sorted(ROOT.glob(f"models/**/{p.name}"))
        return cands[0] if len(cands) == 1 else None
    for alt in (
        result_path.parent.parent / "catalogs" / p.name,  # 搬过家：同名
        result_path.parent.parent / "catalogs" / (result_path.name.split(".")[0] + p.suffix),  # 临时副本：同结果名
    ):
        if alt.is_file():
            return alt
    # 搬过家、目录在同一模型的另一个实践里（如混元 4 按尺寸分实践后，T2 的结果用 01-mini 的目录）：唯一同名才算
    model_dir = next((q for q in result_path.parents if q.parent.name == "models"), None)
    if model_dir is not None:
        cands = sorted((model_dir / "alignments").glob(f"**/catalogs/{p.name}"))
        if len(cands) == 1:
            return cands[0]
    return None


def _with_catalog():
    for f in RESULTS:
        cat = json.loads(f.read_text()).get("catalog") or {}
        if str(cat.get("path", "")).endswith((".yaml", ".yml", ".py")):
            yield f


def _matches(f, e):
    """(对不对得上, 记的值, 现在的值)：content-v1 比内容指纹；脚本写的结果比原文件 sha256。"""
    if e.get("hash_algo") == HASH_ALGO:
        return content_sha256(f) == e["content_sha256"], e["content_sha256"], content_sha256(f)
    got = hashlib.sha256(f.read_bytes()).hexdigest()
    return got == e["sha256"], e["sha256"], got


@pytest.mark.parametrize("result", list(_with_catalog()), ids=lambda f: str(f.relative_to(ROOT)))
def test_catalog_hash_matches_current_file(result):
    rel = str(result.relative_to(ROOT))
    if rel in HISTORICAL:
        pytest.skip(HISTORICAL[rel])
    cat = json.loads(result.read_text())["catalog"]
    entries = [cat] + list(cat.get("includes") or [])
    for e in entries:
        f = _catalog_file(result, e["path"])
        assert f is not None, f"{rel}：找不到 {e['path']}（搬家了登记进 CHANGES-PATHS.md，或写进 HISTORICAL）"
        if f.suffix == ".py":
            assert "sha256" in e, f"{rel}：脚本写的结果要记脚本的 sha256"
        else:
            assert e.get("hash_algo") == HASH_ALGO, (
                f"{rel}：目录写的结果要记内容指纹（hash_algo: {HASH_ALGO}），旧结果用 tools/migrate_catalog_content_hash.py 迁移"
            )
        ok, want, got = _matches(f, e)
        how = "用当前脚本重跑" if f.suffix == ".py" else "检查本身改了，用同一批运行目录重跑 janus run"
        assert ok, (
            f"{rel} 记的 {want[:12]}… ≠ {f.relative_to(ROOT)} 现在的 {got[:12]}…："
            f"{how}并提交新结果（暂时没法重跑的写进 HISTORICAL）"
        )


def test_historical_entries_exist_and_still_differ():
    """HISTORICAL 只放真对不上的：文件没了或已经对上了，就把那一条删掉。"""
    for rel in HISTORICAL:
        f = ROOT / rel
        assert f.is_file(), f"HISTORICAL 里的 {rel} 不存在"
        cat = json.loads(f.read_text())["catalog"]
        cf = _catalog_file(f, cat["path"])
        assert cf is None or not _matches(cf, cat)[0], f"{rel} 已经对上了，从 HISTORICAL 删掉"
