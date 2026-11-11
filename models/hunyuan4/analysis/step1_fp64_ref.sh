#!/bin/bash
# 第 1 步整网梯度的 CPU FP64 参照与三方比较（L3，PROTOCOL 变更记录 2026-11-08）。
# 一个尺寸一遍：先算参照（FP64 全批、FP32 自检、灌 GPU 选择的第一行），再逐代比较。同一时间只跑一个重活。
#
# 用法（变量按本机填；路径都是占位）：
#   HF=<共同的 HF 目录>  TOKENS=<tokens.npy>  OUT=<参照输出目录>  MLM=<Megatron-Core 0.20.0 源码>  PY=<python>
#   GPU_GRADS="g2s-a=<…> g2s-b=<…> g2s-c=<…>"     # export_hf_grads.py 导出的 GPU 第 1 步梯度
#   GPU_L1=<GPU l1 抓数 step_1_fwd.safetensors>    GPU_DUMP=<GPU 第 1 步全量抓数的 step_1 目录>  GPU_METRICS=<其 metrics.jsonl>
#   TPU="t1zdiag2=<diag_grads.safetensors>"  TPU_KD=<同一运行的 diag_kd_inputs.safetensors>  TPU_DIAG=<diag_step1.json>
#   OLD_REF="t1diag18b=<diag_grads.safetensors>"  RESULT=<结果文件>
#   bash step1_fp64_ref.sh ref      # 参照（hy4-mini：FP64 约 18 分钟、FP32 约 10 分钟，40 核；峰值 60 GB）
#   bash step1_fp64_ref.sh compare  # 比较（每代一次，几分钟）
# hy4-mid：ref 加 EXTRA=--recompute（64 头的注意力分数要逐层重算）。T2 截层的 FP64 参数加梯度约 194 GB，
# 加上激活预计超过单进程 250 GB（估算，没有试）。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$MLM:$HERE/../../..
LIMIT=${LIMIT:-250G}
run() { systemd-run --user --scope -q -p MemoryMax=$LIMIT -p MemorySwapMax=0 "$@"; }
case "$1" in
  ref)
    cd "$HERE/../frameworks/megatron"
    run $PY step1_ref.py --hf "$HF" --tokens "$TOKENS" --dtype float64 --out "$OUT/fp64" --threads 40 --audit ${EXTRA:-}
    run $PY step1_ref.py --hf "$HF" --tokens "$TOKENS" --dtype float32 --out "$OUT/fp32" --threads 40 --audit ${EXTRA:-}
    run $PY step1_ref.py --hf "$HF" --tokens "$TOKENS" --dtype float64 --out "$OUT/fp64/inject" --threads 40 \
      --inject-l1 "$GPU_L1" --inject-row 0 --detail-rows 0 ${EXTRA:-}
    ;;
  compare)
    $PY "$HERE/step1_vs_fp64.py" --ref "$OUT/fp64" --gpu $GPU_GRADS --tpu "$TPU" --old-ref "$OLD_REF" \
      --cpu32 "$OUT/fp32" --gpu-l1 "$GPU_L1" --gpu-route-dump "$GPU_DUMP" --tpu-kd-inputs "${TPU_KD:-}" \
      --gpu-meta "$GPU_METRICS" --tpu-diag "${TPU_DIAG:-}" --out "$RESULT"
    ;;
esac
