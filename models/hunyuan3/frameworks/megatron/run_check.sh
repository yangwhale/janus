#!/bin/bash
# 用法：run_check.sh <运行名> <步数> <det|nd> [train_hy3.py 的其他参数]
# 第一阶段的运行（d-a、s-* 等）用的是 Pod 上的 /data/r.sh；本脚本参数与它相同（同一份权重、数据、batch、序列、学习率、种子、EP8），
# 区别只有：训练脚本与 janus 都取本检出（PYTHONPATH 前置本检出），不碰 Pod 上的共享目录。
# 环境变量：
#   RUNS   janus 运行目录的父目录，默认 /data/janus-runs2
#   TF32   off → 关 TF32：去掉 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE、设 NVIDIA_TF32_OVERRIDE=0，并给训练脚本 --no-tf32
#          （torch 开关管不到 TE 的 FP32 GEMM，必须靠 NVIDIA_TF32_OVERRIDE，见 models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-tf32.report.md）
# GPU 共享时由调用方包一层：flock /data/gpu.lock run_check.sh ...
export PYTHONPATH=${PYTHONPATH:-}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../../.." && pwd)
source /data/env.sh
export PYTHONPATH=$SRC:$PYTHONPATH
N=$1; S=$2; M=$3; shift 3
RUNS=${RUNS:-/data/janus-runs2}
EXTRA=()
if [ "$M" = nd ]; then unset NVTE_ALLOW_NONDETERMINISTIC_ALGO CUBLAS_WORKSPACE_CONFIG NCCL_ALGO; else EXTRA+=(--deterministic); fi
if [ "${TF32:-on}" = off ]; then unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; export NVIDIA_TF32_OVERRIDE=0; EXTRA+=(--no-tf32); fi
rm -rf /data/tlogs/$N /data/runs/$N "$RUNS/$N"
cd "$HERE" && torchrun --nproc-per-node 8 --log-dir /data/tlogs/$N --redirects 3 train_hy3.py --hf /data/hy3-4l --out /data/runs/$N \
  --steps $S --load /data/hy3-4l-mcore --gbs 16 --mbs 1 --seq 4096 --ep 8 --lr 1e-05 --seed 1234 --data /data/ds/fw_hy3_text_document \
  --janus-run "$RUNS/$N" --side golden-$N "${EXTRA[@]}" "$@" > /data/tlogs/$N.out 2>&1
echo "RUN_DONE $N rc=$?" >> /data/tlogs/$N.out
