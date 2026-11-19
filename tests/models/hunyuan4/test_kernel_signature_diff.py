"""analysis/kernel_signature_diff.py：按 kernel 名比 tpu_custom_call 的结果类型，认出 kernel 内部缓冲 dtype 变窄
（tokamax 0.0.17 原生行为下 splash dkv 的 dq 部分和从 f32 变 bf16 三缓冲），不把编号后缀、布局注记当成变化。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "analysis"))

import kernel_signature_diff as k  # noqa: E402

TGT = 'custom_call_target="tpu_custom_call", backend_config="x"'


def _hlo(dkv_out: str, gmm_suffix: str = "14") -> str:
    return f"""\
ENTRY %main {{
  %splash_mha_dkv_segmented_no_residuals.37 = ({dkv_out}, bf16[16,16384,128]{{2,1,0:T(16,128)(2,1)}}) custom-call(%a, %b), {TGT}
  %gmm.{gmm_suffix} = bf16[32768,1536]{{1,0:T(8,128)(2,1)}} custom-call(%c), {TGT}
  %other = f32[8]{{0}} custom-call(%d), custom_call_target="Sharding"
}}
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_same_signature_ignores_suffix_and_layout(tmp_path):
    a = _write(tmp_path, "a.txt", _hlo("f32[32,16,2048,128]{3,2,1,0:T(8,128)}"))
    b = _write(tmp_path, "b.txt", _hlo("f32[32,16,2048,128]{3,2,1,0}", gmm_suffix="99"))
    res = k.compare(k.signatures(a), k.signatures(b))
    assert res["changed"] == [] and res["kernels_base"] == 2 and not res["narrowed"]
    assert k.main(["--base", a, "--test", b]) == 0


def test_dq_buffer_narrowing_is_flagged(tmp_path):
    a = _write(tmp_path, "a.txt", _hlo("f32[32,16,2048,128]{3,2,1,0}"))
    b = _write(tmp_path, "b.txt", _hlo("bf16[3,16,2048,128]{3,2,1,0}"))
    res = k.compare(k.signatures(a), k.signatures(b))
    assert res["narrowed"]
    (row,) = res["changed"]
    assert row["kernel"] == "splash_mha_dkv_segmented_no_residuals"
    assert row["narrowed"] == ["结果 0: f32 → bf16"]
    assert k.main(["--base", a, "--test", b]) == 1


def test_shape_only_change_is_not_narrowing(tmp_path):
    a = _write(tmp_path, "a.txt", _hlo("f32[32,16,2048,128]{3,2,1,0}"))
    b = _write(tmp_path, "b.txt", _hlo("f32[16,16,2048,128]{3,2,1,0}"))
    res = k.compare(k.signatures(a), k.signatures(b))
    assert len(res["changed"]) == 1 and not res["narrowed"]


def test_non_tpu_custom_calls_are_ignored(tmp_path):
    a = _write(tmp_path, "a.txt", _hlo("f32[4,8]{1,0}"))
    sig = k.signatures(a)
    assert all(name != "other" for name, _ in sig)
