#!/bin/bash
# 下载 fineweb-edu 的一个分片 → jsonl → 用混元 3 分词器跑 Megatron 的 preprocess_data.py。
# 路径可用环境变量改，默认值是第一阶段的布局：
#   JANUS_ENV    先 source 的环境脚本                 默认 /data/env.sh
#   JANUS_DS     数据目录（不存在就建）                默认 /data/ds
#   JANUS_HF     分词器所在的 HF 目录                  默认 /data/hy3-4l
#   MEGATRON_LM  Megatron-LM 源码（找 tools/preprocess_data.py） 默认 /opt/Megatron-Bridge/3rdparty/Megatron-LM
set -eo pipefail
JANUS_ENV=${JANUS_ENV:-/data/env.sh}
JANUS_DS=${JANUS_DS:-/data/ds}
JANUS_HF=${JANUS_HF:-/data/hy3-4l}
MEGATRON_LM=${MEGATRON_LM:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}
source "$JANUS_ENV"
mkdir -p "$JANUS_DS"
cd "$JANUS_DS"
curl -sfL --retry 5 -o fw.parquet https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet || { echo DL_FAIL; exit 1; }
python - <<PY
import pyarrow.parquet as pq, json
t = pq.read_table("fw.parquet", columns=["text"])
n = 0
with open("fw.jsonl","w") as f:
    for batch in t.to_batches():
        for x in batch.column(0).to_pylist():
            f.write(json.dumps({"text": x}) + "\n"); n += 1
print("docs", n)
PY
sha256sum fw.parquet > fw.parquet.sha256
python "$MEGATRON_LM/tools/preprocess_data.py" --input fw.jsonl --output-prefix fw_hy3   --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$JANUS_HF" --append-eod --workers 64 --json-keys text
ls -la "$JANUS_DS"
echo PREP_DONE
