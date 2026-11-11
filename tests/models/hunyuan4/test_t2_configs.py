"""混元 4 T2 的 TPU 配置（hy4-t2-b.yml、hy4-t2-d1.yml）与 hy4-mid 的配置（hy4-mid-t1.yml）必须带着 T1 现行配置（hy4-mini-t1.yml）的全部数值与优化器开关：
除模型名外逐键相同；阶段名在 hy4_map 里有，层数与 MTP 层数对得上（B：2 层无 MTP；D1：1 层 + MTP）。"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MT = ROOT / "models" / "hunyuan4" / "frameworks" / "maxtext"
sys.path.insert(0, str(ROOT / "models" / "hunyuan4" / "model"))

from hy4_map import STAGE_PARAMS, STAGES  # noqa: E402


def _load(name):
    return yaml.safe_load((MT / name).read_text())


def test_t2_ymls_carry_t1_switches():
    t1 = _load("hy4-mini-t1.yml")
    assert t1["model_name"] == "hy4-mini"
    for name, model in (("hy4-t2-b.yml", "hy4-preview"), ("hy4-t2-d1.yml", "hy4-preview"), ("hy4-mid-t1.yml", "hy4-mid")):
        t2 = _load(name)
        assert t2["model_name"] == model, name
        skip = {"model_name"}
        rest_t1 = {k: v for k, v in t1.items() if k not in skip}
        rest_t2 = {k: v for k, v in t2.items() if k not in skip}
        assert rest_t2 == rest_t1, name
        assert t2["ihc_stream_sum_unrolled"] is True  # 台账 M28：T1 现行配置里最新打开的开关


def test_t2_stages():
    assert STAGES["B"] == (2, 0)
    assert STAGES["D1"] == (1, 1)
    assert STAGE_PARAMS["B"] == 12_078_550_957
    assert STAGE_PARAMS["D1"] == 12_153_673_625


def test_mid_stage():
    assert STAGES["mid"] == (7, 1)
    assert STAGE_PARAMS["mid"] == 5_189_678_849

