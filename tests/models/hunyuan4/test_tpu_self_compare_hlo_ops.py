"""tpu_self_compare.hlo_ops：优化后 HLO 的指令计数要数到 6 元及以上的元组指令（形状里带 /*index=5*/ 注释）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "analysis"))

HLO = """\
ENTRY %main (p: f32[4]) -> f32[4] {
  %a = f32[4]{0:T(128)} all-reduce(%p), channel_id=1, to_apply=%add
  %b = (f32[4]{0:T(128)}, bf16[4]{0:T(128)(2,1)}) all-reduce(%p, %q), channel_id=2, to_apply=%add
  %c = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=5*/f32[1]{0}, bf16[2,3]{1,0:T(8,128)(2,1)}) all-reduce(%p, %p, %p, %p, %p, /*index=5*/%p, %q), channel_id=3, to_apply=%add
  ROOT %d = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=5*/f32[1]{0}) fusion(%p), kind=kLoop, calls=%f
}
"""


def test_counts_tuple_ops_with_index_comments(tmp_path):
    import tpu_self_compare as t

    (tmp_path / "hlo").mkdir()
    (tmp_path / "hlo" / "train_step.after_optimizations.txt").write_text(HLO)
    ops = t.hlo_ops(str(tmp_path))
    assert ops["all-reduce"] == 3
    assert ops["fusion"] == 1
