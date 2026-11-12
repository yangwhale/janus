"""analysis/elided_rounding_scan.py：认出「声明了 BF16 舍入、同一颗融合里又转回 FP32」的往返（案例 17 第九幕），
不把 reduce_precision 补上的、真舍了的、输入本来就在 BF16 网格上的算进去；LLO 核对只看「整颗只含往返」的 kernel。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "analysis"))

import elided_rounding_scan as s  # noqa: E402

# 源码位置表：帧 1 在 linears.py 第 203 行（反向里线性层的输入梯度 / 权重梯度那一行）
TABLES = """\
FileNames
1 "/deps/maxtext/src/maxtext/layers/linears.py"

FunctionNames
1 "__call__"

FileLocations
1 {file_name_id=1 function_name_id=1 line=203 end_line=203 column=1 end_column=1}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
"""

OP_BWD = 'op_name="jit(train_step)/while/body/eval_jaxpr/transpose(jvp())/while/body/eval_jaxpr/convert_element_type"'


def _hlo(fused: str, entry_out: str = "f32[8,128]") -> str:
    return f"""\
HloModule jit_train_step, is_scheduled=true

{fused}

ENTRY %main.2 (p.1: f32[8,128]) -> {entry_out} {{
  %p.1 = f32[8,128]{{1,0}} parameter(0)
  ROOT %fusion.1 = {entry_out} fusion(%p.1), kind=kLoop, calls=%fused_computation.1
}}

{TABLES}"""


# 最小例子：声明了舍入（convert f32→bf16），同一颗融合里马上转回 f32 去用 → 省掉
ELIDED = _hlo(f"""\
%fused_computation.1 (param_0.1: f32[8,128]) -> f32[8,128] {{
  %param_0.1 = f32[8,128]{{1,0}} parameter(0)
  %multiply.1 = f32[8,128]{{1,0}} multiply(%param_0.1, %param_0.1)
  %convert.1 = bf16[8,128]{{1,0:T(8,128)(2,1)}} convert(%multiply.1), metadata={{{OP_BWD} stack_frame_id=1}}
  %convert.2 = f32[8,128]{{1,0}} convert(%convert.1)
  ROOT %add.1 = f32[8,128]{{1,0}} add(%convert.2, %param_0.1)
}}""")

# 反例：转回 f32 后紧跟舍到 BF16 网格的 reduce_precision → 已补（covered），不算省掉
COVERED = _hlo(f"""\
%fused_computation.1 (param_0.1: f32[8,128]) -> f32[8,128] {{
  %param_0.1 = f32[8,128]{{1,0}} parameter(0)
  %multiply.1 = f32[8,128]{{1,0}} multiply(%param_0.1, %param_0.1)
  %convert.1 = bf16[8,128]{{1,0}} convert(%multiply.1), metadata={{{OP_BWD} stack_frame_id=1}}
  %convert.2 = f32[8,128]{{1,0}} convert(%convert.1)
  %reduce-precision.1 = f32[8,128]{{1,0}} reduce-precision(%convert.2), exponent_bits=8, mantissa_bits=7
  ROOT %add.1 = f32[8,128]{{1,0}} add(%reduce-precision.1, %param_0.1)
}}""")

# 反例：输入已经被 reduce_precision 舍到 BF16 网格上，再来回 convert 是精确的 → 跳过
EXACT = _hlo(f"""\
%fused_computation.1 (param_0.1: f32[8,128]) -> f32[8,128] {{
  %param_0.1 = f32[8,128]{{1,0}} parameter(0)
  %reduce-precision.1 = f32[8,128]{{1,0}} reduce-precision(%param_0.1), exponent_bits=8, mantissa_bits=7
  %convert.1 = bf16[8,128]{{1,0}} convert(%reduce-precision.1), metadata={{{OP_BWD} stack_frame_id=1}}
  %convert.2 = f32[8,128]{{1,0}} convert(%convert.1)
  ROOT %add.1 = f32[8,128]{{1,0}} add(%convert.2, %param_0.1)
}}""")

# 多输出融合：BF16 值一路转回 f32（省掉），另一路作为 tuple ROOT 的元素写出（舍了）→ 这颗 kernel 不算「只含往返」
TUPLE = _hlo(
    f"""\
%fused_computation.1 (param_0.1: f32[8,128]) -> (bf16[8,128], f32[8,128]) {{
  %param_0.1 = f32[8,128]{{1,0}} parameter(0)
  %multiply.1 = f32[8,128]{{1,0}} multiply(%param_0.1, %param_0.1)
  %convert.1 = bf16[8,128]{{1,0}} convert(%multiply.1), metadata={{{OP_BWD} stack_frame_id=1}}
  %convert.2 = f32[8,128]{{1,0}} convert(%convert.1)
  %add.1 = f32[8,128]{{1,0}} add(%convert.2, %param_0.1)
  ROOT %tuple.1 = (bf16[8,128]{{1,0}}, f32[8,128]{{1,0}}) tuple(%convert.1, %add.1)
}}""",
    entry_out="(bf16[8,128], f32[8,128])",
)


def _write(tmp_path, text, packs=None):
    hlo = tmp_path / "train_step.after_optimizations.txt"
    hlo.write_text(text)
    jf = None
    if packs is not None:
        jf = tmp_path / "jf"
        jf.mkdir()
        body = "\n".join(["  vadd.f32 v1, v2, v3"] + ["  vpack.i.bf16 v4, v5, v6"] * packs)
        (jf / "1794-fusion.1-71-final_bundles.txt").write_text(body + "\n")
        # schedule-analysis 那份不算
        (jf / "1794-fusion.1-70-schedule-analysis_final_bundles.txt").write_text("vpack.i.bf16\n" * 99)
    return hlo, jf


def test_elided_round_trip_found_and_llo_confirms(tmp_path):
    hlo, jf = _write(tmp_path, ELIDED, packs=0)
    r = s.scan(hlo, jf, positions=True)
    assert r["llo_check"]["kernels_only_elided"] == 1
    assert r["llo_check"]["with_zero_packs"] == 1
    assert r["n_llo_kernels"] == 1
    (site,) = r["sites"]
    assert site["site"] == "linears.py:203"
    assert site["module"] == "decoder/mlp"  # op_name 里两层 while/body = 层 scan 里
    assert (site["elided_uses"], site["kept_uses"], site["covered_uses"]) == (1, 0, 0)
    assert site["backward"] == 1 and site["forward"] == 0
    assert site["elided_kernels"] == ["fusion.1"] and site["elided_kernel_packs"] == [0]
    assert r["elided_uses_by_class"] == {"weight_grad": {"scan": 1}}  # 形状里没有序列维：按权重梯度归类
    (p,) = r["positions"]
    assert p["kernel"] == "fusion.1" and p["computation"] == "fused_computation.1"
    assert p["direction"] == "backward" and p["kernel_packs"] == 0 and p["kernel_only_elided"] is True


def test_llo_with_packs_is_reported(tmp_path):
    """规律不成立时（例如关了 excess precision），只含往返的 kernel 里有打包指令，要报出来。"""
    hlo, jf = _write(tmp_path, ELIDED, packs=3)
    r = s.scan(hlo, jf)
    assert r["llo_check"]["with_zero_packs"] == 0
    assert [list(t) for t in r["llo_check"]["nonzero_examples"]] == [["fusion.1", 3]]


def test_reduce_precision_after_round_trip_is_covered(tmp_path):
    hlo, jf = _write(tmp_path, COVERED, packs=4)
    r = s.scan(hlo, jf, positions=True)
    assert r["sites"] == []  # 没有省掉的
    assert r["elided_uses_by_class"] == {}
    assert r["llo_check"]["kernels_only_elided"] == 0
    (p,) = r["positions"]
    assert (p["elided_uses"], p["covered_uses"]) == (0, 1)


def test_input_already_on_bf16_grid_is_skipped(tmp_path):
    hlo, _ = _write(tmp_path, EXACT)
    r = s.scan(hlo, None, positions=True)
    assert r["skipped_exact_converts"] == 1
    assert r["sites"] == [] and r["positions"] == []


def test_tuple_root_counts_as_kept(tmp_path):
    hlo, jf = _write(tmp_path, TUPLE, packs=2)
    r = s.scan(hlo, jf)
    (site,) = r["sites"]
    assert (site["elided_uses"], site["kept_uses"]) == (1, 1)
    assert r["llo_check"]["kernels_only_elided"] == 0  # 打包指令来自写出的 BF16，不是省掉的那一路


def test_no_stack_tables(tmp_path):
    """没有栈帧表（别的框架、或 dump 时去掉）也能扫，源码位置记「?」。"""
    hlo, _ = _write(tmp_path, ELIDED.split("FileNames")[0])
    r = s.scan(hlo, None)
    (site,) = r["sites"]
    assert site["site"] != "linears.py:203"
    assert site["elided_uses"] == 1
    assert r["llo_check"]["kernels_only_elided"] == 0  # 没给 LLO
