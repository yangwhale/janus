"""janus validate：好的目录过、坏的目录报清楚哪条；仓库里提交的全部目录都要过。"""

import yaml
from helpers import ROOT, catalog

from janus import validate


def test_good_and_bad(tmp_path, capsys):
    good = tmp_path / "catalogs" / "good.yaml"
    good.parent.mkdir()
    good.write_text(yaml.safe_dump(catalog(tmp_path), allow_unicode=True))
    bad = tmp_path / "catalogs" / "bad.yaml"
    bad.write_text(yaml.safe_dump({**catalog(tmp_path), "schema_version": 2}, allow_unicode=True))
    assert validate.main([str(good)]) == 0
    assert "4 条检查（L2 2、L4 2）" in capsys.readouterr().out
    assert validate.main([str(good), str(bad)]) == 2
    assert "schema_version 必须是 1" in capsys.readouterr().err
    assert validate.main([str(tmp_path / "missing.yaml")]) == 2
    assert validate.main([]) == 2


def test_all_skips_included_fragments(tmp_path):
    d = tmp_path / "m" / "catalogs"
    d.mkdir(parents=True)
    (d / "common.yaml").write_text(yaml.safe_dump({"checks": catalog(tmp_path)["checks"][:1]}))  # 片段：不是完整目录
    main = {**catalog(tmp_path), "include": ["common.yaml"], "checks": catalog(tmp_path)["checks"][1:]}
    (d / "t0.yaml").write_text(yaml.safe_dump(main, allow_unicode=True))
    assert [p.name for p in validate.collect(tmp_path)] == ["t0.yaml"]
    assert validate.main(["--all", str(tmp_path), "-q"]) == 0


def test_every_committed_catalog_is_valid(capsys):
    """仓库里每份检查项目录都能被 janus run 读入：坏的目录在 CI 就拦下，不等到上机才发现。"""
    paths = validate.collect(ROOT / "models")
    assert paths, "没找到任何目录文件"
    bad = [line for ok, line in map(validate.check, paths) if not ok]
    assert not bad, "\n".join(bad)


def test_hard_default_per_type(tmp_path):
    """不写 hard：ulp_distribution 默认只报不拦（规格「默认只诊断」），其余默认 hard。写了就照写的。"""
    from janus.catalog import load

    cat = catalog(
        tmp_path,
        [
            {
                "id": "L2.ulp",
                "type": "ulp_distribution",
                "level": "L2",
                "select": {"kind": "tensor", "step": 1, "name": "*"},
            },
            {
                "id": "L2.ulp.hard",
                "type": "ulp_distribution",
                "level": "L2",
                "hard": True,
                "select": {"kind": "tensor", "step": 1, "name": "*"},
            },
        ],
    )
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cat, allow_unicode=True))
    hard = {c["id"]: c["hard"] for c in load(p).checks}
    assert hard["L2.ulp"] is False and hard["L2.ulp.hard"] is True and hard["L4.loss"] is True
