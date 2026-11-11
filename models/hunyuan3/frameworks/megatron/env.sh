# 混元 3 GPU 侧的环境（run.sh、prep_data.sh 都先 source 它）。路径可用环境变量改，默认值是第一阶段的布局：
#   JANUS_SRC         janus 仓库检出（加进 PYTHONPATH）     默认 /data/janus-src
#   JANUS_TRITON_CACHE  Triton 缓存（设成 TRITON_CACHE_DIR）  默认 /data/triton_cache
export NCCL_ENV_PLUGIN=none NCCL_NET_PLUGIN=none NCCL_PROFILER_PLUGIN=none NCCL_TUNER_PLUGIN=none
unset NCCL_TUNER_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export JANUS_SRC=${JANUS_SRC:-/data/janus-src}
export PYTHONWARNINGS=ignore TRITON_CACHE_DIR=${JANUS_TRITON_CACHE:-/data/triton_cache}
export PYTHONPATH=$JANUS_SRC${PYTHONPATH:+:$PYTHONPATH}
