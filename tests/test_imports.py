"""判定引擎不依赖训练框架（docs/dev/architecture.md「几条不变式」第 4 条）：导入引擎、跑 janus run 都不拉进 torch / jax，
也不拉进 matplotlib（画图只在 --html 与分析子命令里按需导入）。在子进程里查 sys.modules，不受本进程已导入的模块影响。"""

import json
import subprocess
import sys

from helpers import ROOT

HEAVY = ("torch", "jax", "jaxlib", "flax", "matplotlib")
ENGINE = (
    "janus",
    "janus.cli",
    "janus.run",
    "janus.catalog",
    "janus.store",
    "janus.metrics",
    "janus.checks",
    "janus.validate",
    "janus.report.markdown",
)


def _loaded_after(code: str) -> set[str]:
    probe = code + "\nimport sys, json\nprint(json.dumps(sorted(m.split('.')[0] for m in sys.modules)))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, check=True)
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_engine_modules_do_not_import_heavy_frameworks():
    loaded = _loaded_after("\n".join(f"import {m}" for m in ENGINE))
    assert not loaded & set(HEAVY), f"引擎导入了 {sorted(loaded & set(HEAVY))}"


def test_janus_run_end_to_end_stays_light(tmp_path):
    """真跑一遍 demo：取数、全部检查、写 report.md，都不需要 torch / jax / matplotlib。"""
    code = (
        f"from janus import demo\nfrom janus.run import main\ncat = demo.make_demo({str(tmp_path / 'd')!r})\n"
        f"assert main([str(cat), '--out', {str(tmp_path / 'o')!r}]) == 0"
    )
    loaded = _loaded_after(code)
    assert not loaded & set(HEAVY), f"janus run 导入了 {sorted(loaded & set(HEAVY))}"


def test_probe_catches_a_heavy_import():
    """工具先自证：真导入了 matplotlib 时探针要看得见。"""
    assert "matplotlib" in _loaded_after("import matplotlib")
