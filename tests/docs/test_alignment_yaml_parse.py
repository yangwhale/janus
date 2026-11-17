"""对齐目录下的 YAML（runs.yaml、阶梯、目录）都要能解析；手改出错时 make check 能拦住。"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted(p for p in ROOT.glob("models/*/alignments/**/*.yaml") if ".git" not in p.parts)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_yaml_parses(path: Path) -> None:
    yaml.safe_load(path.read_text())
