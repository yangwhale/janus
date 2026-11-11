"""janus demo：造出来的运行与目录能直接跑，好的全过、埋错的只在该报的地方报。"""

import json

from janus import demo
from janus.run import main as run_main


def _statuses(out):
    return {c["id"]: c["status"] for c in json.loads((out / "results.json").read_text())["checks"]}


def test_demo_good_and_bad(tmp_path):
    cat = demo.make_demo(tmp_path / "d")
    assert run_main([str(cat), "--out", str(tmp_path / "good")]) == 0
    assert set(_statuses(tmp_path / "good").values()) == {"pass"}

    assert run_main([str(cat), "--out", str(tmp_path / "bad"), "--test", "t-bad"]) == 1
    st = _statuses(tmp_path / "bad")
    failed = {k for k, v in st.items() if v == "fail"}
    assert failed == {"L2.layer1.fwd", "L4.loss"}, st  # 埋错的层与 loss 报警
    assert st["L2.layer0.fwd"] == "pass"  # 上一层不受牵连：逐层定位


def test_demo_refuses_to_overwrite(tmp_path, capsys):
    assert demo.main([str(tmp_path)]) == 0
    assert demo.main([str(tmp_path)]) == 2
    assert "已存在" in capsys.readouterr().err
