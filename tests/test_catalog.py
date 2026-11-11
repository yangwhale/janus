"""catalog.py：检查项目录的 include / vars / foreach 展开、内容指纹（校验规则的用例在 test_run.py 与 test_validate.py）。"""

import hashlib
import json

import pytest
import yaml

from janus.catalog import HASH_ALGO, content_sha256, load


def test_catalog_include_and_typed_vars(tmp_path):
    (tmp_path / "frag.yaml").write_text(yaml.safe_dump({"checks": [
        {"id": "R01", "type": "run_complete", "level": "regress", "select": {"kind": "metrics"},
         "params": {"steps": "{steps}"}},
        {"id": "R03", "type": "data_fingerprint", "level": "regress", "select": {"kind": "tokens", "steps": "1:{end}"}}]}))
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"schema_version": 1, "model": "m", "stage": "s",
                                                      "runs": {"root": "/x", "test": "t"}, "include": ["frag.yaml"],
                                                      "vars": {"steps": 20, "end": 21}, "checks": []}))
    cat = load(tmp_path / "t.yaml")
    assert cat.checks[0]["params"]["steps"] == 20 and cat.checks[1]["select"]["steps"] == "1:21"
    assert cat.data["_includes"][0]["path"].endswith("frag.yaml")


BASE_CAT = """\
schema_version: 1
model: m
stage: s
description: 一句话说明
runs:
  test: t
  golden: [g]
  noise: [[g, h]]
  root: /x
checks:
  - id: L3.grad
    type: tensor_tolerance
    level: L3
    select: {kind: tensor, step: 1, name: "grad.*"}
    metrics: [rel_l2]
    threshold: {rel_l2: 0.001}
    source: issue 评论
    notes: 备注
  - id: L4.loss
    type: loss_series
    level: L4
    select: {kind: metrics, field: lm_loss, window: "1:100"}
    threshold: pending
    pending_note: 等标定
    params: {source: 规格默认值, window: "1:100"}
"""


def _h(text):
    return content_sha256(text)


@pytest.mark.parametrize("edit", [
    lambda t: t.replace("schema_version: 1\n", "# 新注释：见 docs/reference/judging.md\nschema_version: 1\n"),
    lambda t: t.replace("    metrics: [rel_l2]\n", "    metrics: [rel_l2]   # 行尾注释\n"),
    lambda t: t.replace("description: 一句话说明", "description: 改过的说明"),
    lambda t: t.replace("description: 一句话说明\n", ""),
    lambda t: t.replace("source: issue 评论", "source: 推断：另一种写法"),
    lambda t: t.replace("notes: 备注", "notes: 改过的备注"),
    lambda t: t.replace("pending_note: 等标定", "pending_note: 等第二批运行"),
    lambda t: t.replace("    notes: 备注\n", "    notes: 备注\n    description: 检查项上也能写说明\n"),
    lambda t: t.replace("threshold: {rel_l2: 0.001}", "threshold: {rel_l2: 1.0e-3}"),        # 同一个浮点数的另一种写法
    lambda t: t.replace("select: {kind: tensor, step: 1, name: \"grad.*\"}",
                        "select:\n      name: \"grad.*\"\n      kind: tensor\n      step: 1"),  # 块写法、键换顺序
], ids=["comment", "trailing-comment", "description", "drop-description", "source", "notes", "pending_note",
        "check-description", "float-spelling", "layout"])
def test_content_hash_ignores_comments_and_descriptions(edit):
    edited = edit(BASE_CAT)
    assert edited != BASE_CAT
    assert hashlib.sha256(edited.encode()).hexdigest() != hashlib.sha256(BASE_CAT.encode()).hexdigest()
    assert _h(edited) == _h(BASE_CAT)


@pytest.mark.parametrize("edit", [
    lambda t: t.replace("threshold: {rel_l2: 0.001}", "threshold: {rel_l2: 0.002}"),
    lambda t: t.replace("threshold: pending", "threshold: calibrated"),
    lambda t: t.replace("noise: [[g, h]]", "noise: [[g, h], [g, t]]"),
    lambda t: t.replace("root: /x", "root: /y"),
    lambda t: t.replace("name: \"grad.*\"", "name: \"grad.layers.*\""),
    lambda t: t.replace("params: {source: 规格默认值, window: \"1:100\"}", "params: {source: 规格默认值, window: \"1:200\"}"),
    lambda t: t.replace("params: {source: 规格默认值", "params: {source: 别的出处"),   # params 里的键一律算内容
    lambda t: t.replace("    metrics: [rel_l2]\n", "    metrics: [rel_l2]\n    hard: false\n"),
    lambda t: t.replace("threshold: {rel_l2: 0.001}", "threshold: {rel_l2: 1}"),         # 1 与 1.0 / 0.001 是不同的值
], ids=["threshold", "threshold-kind", "noise", "runs-root", "select", "params", "params-source", "hard", "int-vs-float"])
def test_content_hash_changes_with_check_content(edit):
    edited = edit(BASE_CAT)
    assert edited != BASE_CAT
    assert _h(edited) != _h(BASE_CAT)


def test_content_hash_accepts_path_text_and_bytes(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(BASE_CAT)
    assert content_sha256(p) == content_sha256(BASE_CAT) == content_sha256(BASE_CAT.encode())


def test_includes_record_content_and_raw_hash(tmp_path):
    (tmp_path / "frag.yaml").write_text("# 片段\nchecks:\n  - {id: R01, type: run_complete, level: regress, "
                                         "select: {kind: metrics}, params: {steps: 3}}\n")
    (tmp_path / "t.yaml").write_text(json.dumps({"schema_version": 1, "model": "m", "stage": "s",
                                                 "runs": {"root": "/x", "test": "t"}, "include": ["frag.yaml"],
                                                 "checks": []}))
    inc = load(tmp_path / "t.yaml").data["_includes"][0]
    raw = (tmp_path / "frag.yaml").read_bytes()
    assert inc["hash_algo"] == HASH_ALGO
    assert inc["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert inc["content_sha256"] == content_sha256(raw.decode().replace("# 片段\n", ""))   # 片段的注释也不进指纹


def test_display_path_is_repo_relative(tmp_path):
    """结果里记的目录路径不带本机绝对路径：仓库里的文件写成相对仓库根，相对路径原样。"""
    from janus.catalog import display_path, fingerprint

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    cat = repo / "models" / "m" / "catalogs" / "c.yaml"
    cat.parent.mkdir(parents=True)
    cat.write_text("stage: x\n")
    assert display_path(cat) == "models/m/catalogs/c.yaml"
    assert fingerprint(cat)["path"] == "models/m/catalogs/c.yaml"
    assert display_path("models/m/catalogs/c.yaml") == "models/m/catalogs/c.yaml"
    outside = tmp_path / "loose.yaml"
    outside.write_text("stage: y\n")
    assert display_path(outside) == outside.as_posix()
