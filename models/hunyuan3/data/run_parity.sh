#!/bin/bash
# TPU 原生数据加载器对拍，一条命令从头跑完（全程 CPU）。在 GPU Pod 里跑：那里有数据、分词器、GPU golden 和 Megatron-Bridge。
#
#   bash models/hunyuan3/data/run_parity.sh [工作目录]
#
# 可用环境变量改路径：DATA RUN HF MEGATRON_CACHE MEGATRON_PY MAXTEXT_COMMIT VENV
# 产出：$W/parity/results.json、report.md；中间产物（MaxText dump、两边的参照）都在 $W 下。
set -euo pipefail

W=${1:-/tmp/g62}
J=$(cd "$(dirname "$0")/../../.." && pwd)                          # janus 仓库根
DATA=${DATA:-/data/ds/fw_hy3_text_document}                          # Megatron .bin/.idx 前缀
RUN=${RUN:-/data/janus-runs2/g1k-a}                                  # GPU golden（--save-batches）
HF=${HF:-/data/hy3-4l}                                               # 分词器（只给 Megatron 侧取 eod / pad id）
MEGATRON_CACHE=${MEGATRON_CACHE:-$DATA/cache/GPTDataset_indices}     # GPU 训练时 Megatron 写的索引缓存
MEGATRON_PY=${MEGATRON_PY:-/opt/venv/bin/python}
MAXTEXT_COMMIT=${MAXTEXT_COMMIT:-ce608b918c4fdfbfdb62f5a7976d2e15e4fdd720}
VENV=${VENV:-$W/venv}
S=$J/models/hunyuan3/data

export CUDA_VISIBLE_DEVICES=                                         # 不用 GPU
export JAX_PLATFORMS=cpu
mkdir -p "$W"

# 1. MaxText（固定 commit）装进单独的 venv；不碰系统 Python 和 Pod 里已有的环境
if [ ! -x "$VENV/bin/python" ]; then
  [ -d "$W/maxtext" ] || git clone -q https://github.com/AI-Hypercomputer/maxtext.git "$W/maxtext"
  git -C "$W/maxtext" checkout -q "$MAXTEXT_COMMIT"
  env -u VIRTUAL_ENV uv venv -q --python /usr/bin/python3.12 "$VENV"
  (cd "$W/maxtext" && VIRTUAL_ENV="$VENV" uv pip install -q -e ".[tpu]" --resolution=lowest)
fi
PY=$VENV/bin/python

# 2. MaxText 走训练入口读 1000 步（8 个 CPU 假设备，per_device_batch_size=1、gradient_accumulation_steps=2）
D="--data $DATA --steps 1000"
$PY "$S/maxtext_dump.py" $D --npy-dir "$W/npy" --out "$W/mt-w1" --workers 1      # 第一次：索引缓存未命中，现建
$PY "$S/maxtext_dump.py" $D --npy-dir "$W/npy" --out "$W/mt-w0" --workers 0      # 之后：命中缓存
$PY "$S/maxtext_dump.py" $D --npy-dir "$W/npy" --out "$W/mt-w8" --workers 8
for h in 2 4; do
  $PY "$S/maxtext_dump.py" $D --npy-dir "$W/npy" --out "$W/mt-h$h" --emulate-hosts $h
done
# 反例：MaxText 的默认 reset_attention_mask / packing、不设 split、默认种子 0（只存前 5 步）
$PY "$S/maxtext_dump.py" $D --dump-steps 5 --npy-dir "$W/npy-ctrl-reset" --out "$W/ctrl-reset" --reset-attention-mask --packing-divisor 25
$PY "$S/maxtext_dump.py" $D --dump-steps 5 --npy-dir "$W/npy-ctrl-nosplit" --out "$W/ctrl-nosplit" --split ""
$PY "$S/maxtext_dump.py" $D --dump-steps 5 --npy-dir "$W/npy-ctrl-seed0" --out "$W/ctrl-seed0" --seed 0

# 3. Megatron 侧参照（Megatron-Bridge 自己的 GPTDataset，CPU）：跨 epoch 的索引与样本、合成边界用例
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
for n in 16000 356736 400000; do
  PYTHONWARNINGS=ignore $MEGATRON_PY "$S/megatron_reference.py" epochs --data "$DATA" --hf "$HF" --num-samples $n \
    --cache "$W/meg-cache" --out "$W/meg-ep-$n"
done
PYTHONWARNINGS=ignore $MEGATRON_PY "$S/megatron_reference.py" synthetic --hf "$HF" --work "$W/syn" --out "$W/meg-syn"

# 4. MaxText 侧同样的参照
for n in 16000 356736 400000; do
  $PY "$S/maxtext_reference.py" epochs --ref "$W/meg-ep-$n" --data "$DATA" --npy-dir "$W/npy-ref" --out "$W/mt-ep-$n"
done
$PY "$S/maxtext_reference.py" synthetic --ref "$W/meg-syn" --npy-dir "$W/npy-ref" --out "$W/mt-syn"

# 5. 对拍（pad id 取 Megatron 分词器给的值）
PAD=$($PY -c "import json; print(json.load(open('$W/meg-syn/ref.json'))['pad'])")
$PY "$S/native_loader_parity.py" --mt "$W/mt-w1" --gpu-run "$RUN" --megatron-index-dir "$MEGATRON_CACHE" \
  --out "$W/parity" --pad-id "$PAD" --count-pad-in "$DATA.bin" \
  --mt-variant "$W/mt-w0" --mt-variant "$W/mt-w8" --mt-hosts "$W/mt-h2" --mt-hosts "$W/mt-h4" \
  --epochs-ref "$W/meg-ep-16000:$W/mt-ep-16000" --epochs-ref "$W/meg-ep-356736:$W/mt-ep-356736" \
  --epochs-ref "$W/meg-ep-400000:$W/mt-ep-400000" --eod-ref "$W/meg-syn:$W/mt-syn" \
  --negative "MaxText 默认的 reset_attention_mask 与 packing=$W/ctrl-reset" --negative "不设 mmap_npy_split=$W/ctrl-nosplit" \
  --negative "MaxText 默认种子 0=$W/ctrl-seed0"
