#!/bin/bash
# 用法：run.sh <运行名> <步数> [train_hy3.py 的其他参数]
#
# 两种设置（docs/reference/environment.md「两种设置」）：
#   参数里有 --deterministic → 确定性设置，并设三个环境变量
#       NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring
#   没有 --deterministic     → 默认设置，三个变量都 unset（与第一阶段默认设置的运行一致）
#   JANUS_DET_ENV=1 / 0 强制设 / 不设这三个变量（如复现「确定性但没设三个变量」的 d-a、f-* 等）
#
# 路径可用环境变量改，默认值是第一阶段的布局：
#   JANUS_ENV    先 source 的环境脚本                    默认 /data/env.sh
#   JANUS_WORK   训练脚本所在目录（从这里起 train_hy3.py） 默认 /data/jw
#   JANUS_HF     截好的 HF 目录（--hf）                   默认 /data/hy3-4l；参数里再给 --hf 也会覆盖它
#   JANUS_RUNS   TensorBoard 输出根（--out <根>/<运行名>）  默认 /data/runs
#   JANUS_TLOGS  torchrun 日志根                          默认 /data/tlogs
# 每次启动先 rm -rf <JANUS_RUNS>/<运行名> 与 <JANUS_TLOGS>/<运行名>。
# 最后一行打印 RUN_DONE <运行名> rc=<torchrun 退出码>，脚本以同一个退出码退出。
if [ $# -lt 2 ] || [ -z "$1" ]; then
  echo "用法：run.sh <运行名> <步数> [train_hy3.py 的其他参数]" >&2
  exit 2
fi
JANUS_ENV=${JANUS_ENV:-/data/env.sh}
JANUS_WORK=${JANUS_WORK:-/data/jw}
JANUS_HF=${JANUS_HF:-/data/hy3-4l}
JANUS_RUNS=${JANUS_RUNS:-/data/runs}
JANUS_TLOGS=${JANUS_TLOGS:-/data/tlogs}
source "$JANUS_ENV" || { echo "source $JANUS_ENV 失败" >&2; exit 2; }

N=$1; S=$2; shift 2
det_env=${JANUS_DET_ENV:-auto}
if [ "$det_env" = auto ]; then
  det_env=0
  for x in "$@"; do [ "$x" = --deterministic ] && det_env=1; done
fi
case $det_env in
  1) export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring ;;
  0) unset NVTE_ALLOW_NONDETERMINISTIC_ALGO CUBLAS_WORKSPACE_CONFIG NCCL_ALGO ;;
  *) echo "JANUS_DET_ENV 只能是 auto、1、0，收到 $JANUS_DET_ENV" >&2; exit 2 ;;
esac

rm -rf "$JANUS_TLOGS/$N" "$JANUS_RUNS/$N"
cd "$JANUS_WORK" || { echo "进不了 $JANUS_WORK" >&2; exit 2; }
torchrun --nproc-per-node 8 --log-dir "$JANUS_TLOGS/$N" --redirects 3 train_hy3.py \
  --hf "$JANUS_HF" --out "$JANUS_RUNS/$N" --steps "$S" "$@"
rc=$?
echo "RUN_DONE $N rc=$rc"
exit $rc
