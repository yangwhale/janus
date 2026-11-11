"""janus 命令行：子命令分发、版本号单一来源、旧写法兼容。"""

import re
import subprocess
import sys

import pytest
from helpers import ROOT

import janus
from janus import cli


def test_version_single_source():
    """版本只写在 janus/__init__.py；pyproject.toml 不另写死。"""
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-]?(dev|rc|a|b)\d*)?", janus.__version__)
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert not re.search(r'^version\s*=\s*"', text, re.M), "pyproject.toml 里不要再写死 version"


def test_version_flag(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"janus {janus.__version__}"


def test_no_args_prints_usage_and_fails(capsys):
    assert cli.main([]) == 2
    out = capsys.readouterr().err  # 用法错：用法打到 stderr
    for name in cli.COMMANDS:
        assert name in out


def test_unknown_command(capsys):
    assert cli.main(["nope"]) == 2
    assert "nope" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_every_command_has_help(name, capsys):
    """每个子命令都能 --help 且返回 0：模块存在、main 能接 argv。"""
    assert cli.main([name, "--help"]) == 0
    assert capsys.readouterr().out.strip()


def test_run_bad_args_exit_2(capsys):
    assert cli.main(["run"]) == 2  # 缺目录文件：argparse 报错，按约定返回 2
    assert cli.main(["run", "x.yaml", "--bogus"]) == 2


def test_legacy_form_routes_to_run(tmp_path, capsys):
    """旧写法 `python -m janus <目录文件>`：当 run 处理（目录不存在 → 目录不合规，退出码 2）。"""
    assert cli.main([str(tmp_path / "missing.yaml")]) == 2


def test_python_dash_m_janus():
    r = subprocess.run([sys.executable, "-m", "janus", "--version"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and r.stdout.strip() == f"janus {janus.__version__}"


def test_changelog_has_unreleased_or_current_version():
    """改了代码就要记 CHANGELOG：要么有 [Unreleased] 一节，要么有当前版本的一节。"""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in text or f"## [{janus.__version__}]" in text


def test_missing_file_is_exit_2_not_traceback(tmp_path, capsys):
    """子命令读不了文件：统一退出码 2，不抛栈（1 只留给「有硬检查 fail」）。"""
    assert cli.main(["offsets", "--test", f"t={tmp_path / 'no.jsonl'}", "--golden", f"g={tmp_path / 'no.jsonl'}"]) == 2
    assert "读不了文件" in capsys.readouterr().err
    assert cli.main(["fault-matrix", str(tmp_path), "--spec", str(tmp_path / "no.yaml")]) == 2


def test_unexpected_error_is_exit_2(monkeypatch, capsys):
    import janus.validate

    def boom(argv):
        raise RuntimeError("故意的")

    monkeypatch.setattr(janus.validate, "main", boom)
    assert cli.main(["validate", "x.yaml"]) == 2
    err = capsys.readouterr().err
    assert "RuntimeError: 故意的" in err and "内部错误" in err
