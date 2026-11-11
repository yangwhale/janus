"""analysis/hlo_mof_dus_scan.py：认出 一次实际排查 的特征（案例 16），不误报只是同在一颗多输出 fusion 里的 DUS。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "analysis"))

import hlo_mof_dus_scan as s  # noqa: E402

# libtpu 0.0.46 上 #71 最小复现的那颗 fusion（删去与判断无关的成员）：DUS 原地写 param_0，同一颗 fusion 里 slice 又读 param_0
BAD = """\
HloModule jit_f, is_scheduled=true

%fused_computation.9 (param_0.20: f32[6,2], param_1.24: f32[2,2], param_2.23: f32[2,2]) -> (f32[6,2], f32[2,2]) {
  %param_0.20 = f32[6,2]{1,0:T(8,128)} parameter(0)
  %param_1.24 = f32[2,2]{1,0:T(2,128)S(1)} parameter(1)
  %constant.35 = s32[]{:T(128)} constant(0)
  %dynamic-update-slice.3 = f32[6,2]{1,0:T(8,128)} dynamic-update-slice(%param_0.20, %param_1.24, %constant.35, %constant.35), metadata={op_name="jit(f)/scatter"}
  %slice.25.clone.1 = f32[2,2]{1,0:T(2,128)} slice(%param_0.20), slice={[0:2], [0:2]}
  %param_2.23 = f32[2,2]{1,0:T(2,128)S(1)} parameter(2)
  %add.11.clone.1 = f32[2,2]{1,0:T(2,128)} add(%slice.25.clone.1, %param_2.23)
  ROOT %tuple.9 = (f32[6,2]{1,0:T(8,128)}, f32[2,2]{1,0:T(2,128)}) tuple(%dynamic-update-slice.3, %add.11.clone.1)
}

ENTRY %main.7 (a3.1: f32[6,2], a4.1: f32[2,2], a6.1: f32[2,2]) -> (f32[6,2], f32[2,2]) {
  %a3.1 = f32[6,2]{1,0:T(8,128)} parameter(0)
  %a4.1 = f32[2,2]{1,0:T(2,128)} parameter(1)
  %a6.1 = f32[2,2]{1,0:T(2,128)} parameter(2)
  ROOT %constant_dynamic-update-slice_fusion = (f32[6,2]{1,0:T(8,128)}, f32[2,2]{1,0:T(2,128)}) fusion(%a3.1, %a4.1, %a6.1), kind=kOutput, calls=%fused_computation.9
}
"""

# 混元 3 训练步里常见的那种：多输出 fusion 里有 DUS，但目标缓冲只被 DUS 自己用；第二个输出是更新值本身
OK = """\
HloModule jit_train_step, is_scheduled=true

%fused_computation.1389 (param_0.6662: bf16[3,2,8,128], param_1.7249: s32[], param_7.809: bf16[2,8,128]) -> (bf16[3,2,8,128], bf16[2,8,128]) {
  %param_0.6662 = bf16[3,2,8,128]{3,2,1,0:T(8,128)(2,1)} parameter(0)
  %param_7.809 = bf16[2,8,128]{2,1,0:T(8,128)(2,1)} parameter(2)
  %reduce_precision.234 = bf16[2,8,128]{2,1,0:T(8,128)(2,1)} reduce-precision(%param_7.809), exponent_bits=8, mantissa_bits=7
  %bitcast.2863 = bf16[1,2,8,128]{3,2,1,0:T(8,128)(2,1)} bitcast(%reduce_precision.234)
  %param_1.7249 = s32[]{:T(128)} parameter(1)
  %constant.2028 = s32[]{:T(128)} constant(0)
  %dynamic-update-slice.384 = bf16[3,2,8,128]{3,2,1,0:T(8,128)(2,1)} dynamic-update-slice(%param_0.6662, %bitcast.2863, %param_1.7249, %constant.2028, %constant.2028, /*index=5*/%constant.2028)
  ROOT %tuple.1765 = (bf16[3,2,8,128]{3,2,1,0:T(8,128)(2,1)}, bf16[2,8,128]{2,1,0:T(8,128)(2,1)}) tuple(%dynamic-update-slice.384, %reduce_precision.234)
}

ENTRY %main (p0: bf16[3,2,8,128], p1: s32[], p2: bf16[2,8,128]) -> (bf16[3,2,8,128], bf16[2,8,128]) {
  %p0 = bf16[3,2,8,128]{3,2,1,0:T(8,128)(2,1)} parameter(0)
  %p1 = s32[]{:T(128)} parameter(1)
  %p2 = bf16[2,8,128]{2,1,0:T(8,128)(2,1)} parameter(2)
  ROOT %fusion.7 = (bf16[3,2,8,128]{3,2,1,0:T(8,128)(2,1)}, bf16[2,8,128]{2,1,0:T(8,128)(2,1)}) fusion(%p0, %p1, %p2), kind=kLoop, calls=%fused_computation.1389
}
"""


def test_flags_issue71_signature():
  r = s.scan_text(BAD)
  assert r["mof_with_dus"] == 1
  assert r["with_71_signature"] == 1
  h = r["hits"][0]
  assert h["fusion"] == "constant_dynamic-update-slice_fusion"
  assert h["target_tiles"] == ["8,128"] and h["update_tiles"] == ["2,128"]
  assert len(h["readers_of_target"]) == 1 and "slice(%param_0.20)" in h["readers_of_target"][0]


def test_dus_without_sibling_read_is_not_flagged():
  r = s.scan_text(OK)
  assert r["mof_with_dus"] == 1
  assert r["with_71_signature"] == 0


def test_single_output_fusion_is_ignored():
  single = BAD.replace(
      "ROOT %tuple.9 = (f32[6,2]{1,0:T(8,128)}, f32[2,2]{1,0:T(2,128)}) tuple(%dynamic-update-slice.3, %add.11.clone.1)",
      "ROOT %add.99 = f32[6,2]{1,0:T(8,128)} add(%dynamic-update-slice.3, %dynamic-update-slice.3)")
  assert s.scan_text(single)["mof_with_dus"] == 0
