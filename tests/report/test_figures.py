"""report/figures.py：逐级大表的配图只读 results.json / details，不重算；没数据时不出图。"""

import json

from janus.report import figures


def _calibrated(cid, x, n):
    return {
        "id": cid,
        "threshold": {"kind": "calibrated", "noise": {"upper": {"rel_l2": n}}},
        "metrics": {"rel_l2_max": x},
    }


def test_l2_layers_draws_from_results(tmp_path):
    checks = [_calibrated(f"L2.layer{i}.fwd", 1e-4 * (i + 1), 2e-4) for i in range(4)]
    checks += [_calibrated(f"L2.layer{i}.bwd", 0.0, 1e-5) for i in range(4)]  # 实测为 0 也要画得出来（对数轴）
    checks.append({"id": "L4.loss", "threshold": {"kind": "calibrated"}, "metrics": {}})
    out = figures.l2_layers(checks, tmp_path / "l2.png", "t")
    assert out == tmp_path / "l2.png" and out.stat().st_size > 0
    assert figures.l2_layers([checks[-1]], tmp_path / "none.png", "t") is None  # 没有 L2 逐层检查：不出图
    assert not (tmp_path / "none.png").exists()


def test_ulp_buckets(tmp_path):
    c = {"metrics": {"buckets": {"0": 0.2, "1": 0.3, "2-4": 0.5}}}
    assert figures.ulp_buckets(c, tmp_path / "u.png", "t").stat().st_size > 0
    assert figures.ulp_buckets({"metrics": {}}, tmp_path / "x.png", "t") is None


def test_router_margin(tmp_path):
    d = tmp_path / "d.jsonl"
    assert figures.router_margin(d, tmp_path / "m.png", "t") is None  # 没有明细文件
    d.write_text(json.dumps({"layer": 1, "margins_all": [1e-3, 1e-2, 1e-1]}) + "\n")
    assert figures.router_margin(d, tmp_path / "m.png", "t") is None  # 没有翻转：不出图
    d.write_text(json.dumps({"layer": 1, "margins_all": [1e-3, 1e-2, 1e-1], "margins_flipped": [2e-6, 5e-6]}) + "\n")
    assert figures.router_margin(d, tmp_path / "m.png", "t").stat().st_size > 0
