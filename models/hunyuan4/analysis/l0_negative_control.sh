#!/bin/bash
# L0 负对照（hy4-mini）：不做 RoPE 列去交错转出来的检查点，L0 的三道核对都必须判不一致。
#   1) 同一个 HF 目录按 checkpoint_rope_interleave=false 转换（只有这一处和正式转换不同）→ <work>/neg-ckpt
#   2) verify_ckpt.py maxtext：按对齐配置（去交错）反向取回，对 tensor_sha256.json —— 期望 FAIL
#   3) verify_ckpt.py rope：去交错三条核对 —— 期望 FAIL
#   4) l0_init_vs_gpu_mcore.py：换成 GPU 名字对 GPU 初始参数清单 —— 期望 FAIL（且只在带 RoPE 列的张量上不同）
# 再用 l0_negative_control_summary.py 把四步的结果合成一份结果 json。
# 用法：bash l0_negative_control.sh <HF 目录> <TPU 运行目录> <GPU 运行目录> <work> <结果 json>
#   需要 MaxText 的 CPU 环境（PY 指向它的 python）；单进程内存约 10 GB
set -uo pipefail
HF=$1; TPU=$2; GPU=$3; W=$4; OUT=$5
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd); M=$HERE/../model
export JAX_PLATFORMS=cpu PYTHONPATH=$(cd "$HERE/../../.." && pwd); unset HY4_YML
mkdir -p "$W"
cd "$M"
[ -d "$W/neg-ckpt/0/items" ] || $PY convert_hf_to_maxtext.py --hf "$HF" --stage mini --out "$W/neg-ckpt" model_name=hy4-mini checkpoint_rope_interleave=false > "$W/convert.log" 2>&1
echo "convert rc=$?"
$PY verify_ckpt.py maxtext --ckpt "$W/neg-ckpt/0/items" --hf "$HF" --stage mini --janus-run "$W/runs/hy4-mini-maxtext-ckpt-neg" model_name=hy4-mini > "$W/maxtext.log" 2>&1
echo "maxtext rc=$?（期望 1）"
$PY verify_ckpt.py rope --ckpt "$W/neg-ckpt/0/items" --hf "$HF" --stage mini --out "$W/rope" model_name=hy4-mini > "$W/rope.log" 2>&1
echo "rope rc=$?（期望 1）"
$PY "$HERE/l0_init_vs_gpu_mcore.py" --ckpt "$W/neg-ckpt/0/items" --hf "$HF" --stage mini --tpu-run "$TPU" --gpu-run "$GPU" --out "$W/init_vs_gpu.json" model_name=hy4-mini > "$W/init.log" 2>&1
echo "init_vs_gpu rc=$?"
$PY "$HERE/l0_negative_control_summary.py" --work "$W" --hf "$HF" --out "$OUT"
