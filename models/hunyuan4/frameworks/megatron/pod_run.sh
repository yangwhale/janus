#!/bin/bash
# 混元 4 Megatron 版在 GPU Pod 里的整条流程（不依赖进 Pod 执行命令）：
#   从桶拉代码、截好的 HF、数据 → L0 核张量 sha256 → [Bridge 自检] → [GPU 等价性检查] → 依次训练 RUNS → 每步结果传桶。
# 环境变量：
#   BKT / PFX   桶名与桶内前缀（站点配置给），结果写到 $PFX/megatron/$JOB/
#   CODE        桶内代码包（janus 仓库的 tar.gz）
#   HF_FILES    要拉的 HF 文件（空格分隔），在 $PFX/hf/$HFDIR/ 下（HFDIR 默认 hy4-4l）
#   NPROC       每台机器每个训练用几张卡（默认 8；hy4-mini 用 1–4）
#   NNODES      机器数（默认 1）。多机时每台跑同一个脚本：NODE_RANK 取 JOB_COMPLETION_INDEX（Indexed Job / JobSet），
#               MASTER_ADDR 是 0 号机的地址（JobSet 的 <名字>-<副本组>-0-0.<名字>），MASTER_PORT 默认 29500；
#               机间走 GPUDirect RDMA：Pod 要挂 RDMA 网卡（见 README「多机」），NCCL 用镜像自带的 GCP 插件（gIB）
#   BRIDGE      非空时跑 bridge_check.py，值是截几层（如 2）
#   EQUIV       空格分隔的「层数:模式」，模式里含 bwd 则反向、含 mtp 则带 MTP、含 replay 则灌 HF 的 top-k（如 fwdbwdmtpreplay），如 "1:fwdbwd 3:fwd"；HF 在 cuda:1、Megatron（TE）在 cuda:0
#   RUNS        空格分隔的「运行名:层数:步数:附加参数（逗号分隔）」，如 "mg-a:1:10: mg-c:3:10:--recompute"
#   JOB         作业名（结果目录名）
set -o pipefail
mkdir -p /data/out
exec > >(tee -a /data/out/pod.log) 2>&1
tok() { curl -s --max-time 30 --retry 5 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])'; }
enc() { python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
gput() { curl -sf --retry 3 -X POST -H "Authorization: Bearer $(tok)" -H "Content-Type: application/octet-stream" -T "$1" "https://storage.googleapis.com/upload/storage/v1/b/$BKT/o?uploadType=media&name=$(enc "$2")" > /dev/null && echo "PUT $2" || echo "PUT_FAIL $2"; }
export -f tok enc gput
export BKT
OUT=$PFX/megatron/$JOB
up() { gput "$1" "$OUT/$2"; }

NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}; MASTER_PORT=${MASTER_PORT:-29500}
NSFX=""; [ "$NNODES" -gt 1 ] && NSFX="-node$NODE_RANK"   # 多机时每台的日志分开传
# NCCL：单机关掉镜像里的 GCP 插件（它要 RDMA 网卡，单机 Pod 没挂；同混元 3 env.sh）；多机保留它，机间走 gIB。
# 两种都让镜像自带的 NCCL 2.31 先被找到（节点挂的旧版在 /usr/local/nvidia/lib64），与单机 golden 同一份 NCCL。
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
if [ "$NNODES" -eq 1 ]; then
  export NCCL_ENV_PLUGIN=none NCCL_NET_PLUGIN=none NCCL_PROFILER_PLUGIN=none NCCL_TUNER_PLUGIN=none
  unset NCCL_TUNER_CONFIG_PATH
fi
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONWARNINGS=ignore NVIDIA_TF32_OVERRIDE=0
# 确定性运行（附加参数里有 --deterministic）要的三个环境变量由训练入口补（megatron.training.determinism）

echo "POD_START $(date -u +%FT%TZ) $(hostname) JOB=$JOB NNODES=$NNODES NODE_RANK=$NODE_RANK"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
python3 - <<'PY'
import torch, megatron.core, transformer_engine
try:
    import megatron.bridge as mb; mbv = getattr(mb, "__version__", "?")
except Exception as e:  # noqa: BLE001
    mbv = f"import failed: {e}"
import transformers
try:
    import emerging_optimizers as eo; eov = getattr(eo, "__version__", "?")
except Exception as e:  # noqa: BLE001
    eov = f"import failed: {e}"
print("EMERGING_OPTIMIZERS", eov, flush=True)
print("VERSIONS torch", torch.__version__, "mcore", megatron.core.__version__, megatron.core.__file__,
      "te", transformer_engine.__version__, "bridge", mbv, "transformers", transformers.__version__, flush=True)
PY
python3 - <<'PY' || true
# QK-Clip：TE 注意力返回的 max logit 是缩放前还是缩放后（定义要和 hy4_mcore 的 sparse_sink_attention 一致）
import inspect, re, transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention as m
src = inspect.getsource(m)
for i, l in enumerate(src.splitlines()):
    if re.search(r"max_logit", l):
        print("TE_MAXLOGIT", i, l.strip()[:200])
PY
mkdir -p /data/janus-src /data/hf /data/ds
# gcloud CLI（镜像里没有，几秒装好）；两处让下载慢到几百秒的原因（GLM-5.2 一侧实测）：
# Pod 没有 IPv6 路由但 DNS 把 IPv6 排前面，每条连接先等约 4.2 s；gcloud 自带 Python 没有 C 版 crc32c。
t0=$(date +%s)
if ! command -v gcloud >/dev/null; then
  curl -sfL --retry 5 -o /data/gcloud.tgz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
    && tar -xzf /data/gcloud.tgz -C /data && export PATH=/data/google-cloud-sdk/bin:$PATH || { echo GCLOUD_INSTALL_FAIL; exit 1; }
fi
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
grep -q '^precedence ::ffff:0:0/96' /etc/gai.conf 2>/dev/null || echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf
GPY=$(dirname "$(dirname "$(command -v gcloud)")")/platform/bundledpythonunix/bin/python3
[ -x "$GPY" ] && "$GPY" -m pip install -q --disable-pip-version-check google-crc32c >/dev/null 2>&1
echo "GCLOUD_READY seconds=$(( $(date +%s) - t0 )) crc32c=$("$GPY" -c 'import google_crc32c; print(google_crc32c.implementation)' 2>&1 | tail -1)"
echo "CODE_GET $CODE"
gcloud storage cp -q "gs://$BKT/$CODE" /data/code.tgz && tar -xzf /data/code.tgz -C /data/janus-src || { echo CODE_FAIL; exit 1; }
echo "CODE_OK $(ls /data/janus-src | head -3 | tr '\n' ' ')"
python3 -m pip install -q --no-deps -e /data/janus-src 2>&1 | tail -1   # janus 包（训练入口用 janus.dump.megatron 写运行目录）
G=/data/janus-src/models/hunyuan4/frameworks/megatron
t0=$(date +%s)
srcs=""; for f in $HF_FILES; do srcs="$srcs gs://$BKT/$PFX/hf/${HFDIR:-hy4-4l}/$f"; done
gcloud storage cp -q $srcs /data/hf/ && gcloud storage cp -q "gs://$BKT/$PFX/data/${TOKFILE:-tokens.npy}" /data/ds/tokens.npy || { echo DL_FAIL; exit 1; }
echo "DL_SECONDS $(( $(date +%s) - t0 ))"; du -sh /data/hf /data/ds
# L0：拉下来的每个张量按原始字节对 tensor_sha256.json 逐个核（只核 index 里指向已下载文件的张量）
( cd $G && python3 - <<'PY' ) || { echo L0_HF_SHA_FAIL; exit 1; }
import json, os
from hy4_weights import Safetensors, tensor_sha_ok
st = Safetensors("/data/hf")
names = [k for k, f in st.where.items() if os.path.exists(os.path.join("/data/hf", f))]
print("L0_HF_SHA_OK tensors", tensor_sha_ok("/data/hf", names), flush=True)
PY

# 索引器 KD 对 MaxText 定义：KD="4:mtp 4:mtp+dense 4:mtp+neg"（层数:选项）
for spec in $KD; do
  L=${spec%%:*}; mode=${spec#*:}; kx="--te"
  case $mode in *mtp*) kx="$kx --mtp";; esac
  case $mode in *dense*) kx="$kx --dense";; esac
  case $mode in *neg*) kx="$kx --neg-sum";; esac
  tag=L${L}_${mode//+/_}
  echo "KD_START $spec $(date -u +%FT%TZ)"
  ( cd $G && CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 indexer_kd_check.py --hf /data/hf --layers $L --seq 4096 \
      --tokens /data/ds/tokens.npy $kx --json /data/out/kd_$tag.json ) > /data/out/kd_$tag.log 2>&1
  echo "KD_END $spec rc=$?"; grep -E "^(KD|PASS|FAIL)|Error" /data/out/kd_$tag.log | cut -c1-400
  up /data/out/kd_$tag.log kd_$tag.log; [ -f /data/out/kd_$tag.json ] && up /data/out/kd_$tag.json kd_$tag.json
done

if [ -n "$BRIDGE" ]; then
  echo "BRIDGE_START layers=$BRIDGE $(date -u +%FT%TZ)"
  ( cd $G && CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 bridge_check.py --hf /data/hf --layers $BRIDGE --work /data/bridge-hf \
      --json /data/out/bridge_check.json ) > /data/out/bridge_check.log 2>&1; echo "BRIDGE_END rc=$?"
  grep -E "BRIDGE_CHECK|Error|error" /data/out/bridge_check.log | tail -5
  up /data/out/bridge_check.log bridge_check.log; [ -f /data/out/bridge_check.json ] && up /data/out/bridge_check.json bridge_check.json
fi

if [ -n "$EQUIV" ]; then
  # HF 参照要 transformers 5.22.1（与 HF golden 一侧同版本、同 modeling_hy_v4.py）；训练不用 transformers
  python3 -m pip install -q "transformers==5.22.1" 2>&1 | tail -1
  python3 - <<'PY' || echo EQUIV_TF_VERSION_FAIL
import hashlib, transformers, transformers.models.hy_v4.modeling_hy_v4 as m
h = hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()
print("EQUIV_TF", transformers.__version__, h, flush=True)
assert transformers.__version__ == "5.22.1" and h.startswith("e27ec64d")
PY
  for spec in $EQUIV; do
    L=${spec%%:*}; mode=${spec#*:}; bw=""; EQUIV_EXTRA=""
    case $mode in *bwd*) bw="--backward";; esac
    case $mode in *mtp*) EQUIV_EXTRA="--mtp";; esac
    case $mode in *replay*) EQUIV_EXTRA="$EQUIV_EXTRA --replay-topk";; esac
    echo "EQUIV_START L=$L $mode $(date -u +%FT%TZ)"
    ( cd $G && timeout 5400 python3 equiv_hf.py --hf /data/hf --layers $L --seq 4096 --tokens /data/ds/tokens.npy \
        --device cuda:0 --hf-device cuda:1 --te $bw $EQUIV_EXTRA --json /data/out/equiv_L${L}_$mode.json ) > /data/out/equiv_L${L}_$mode.log 2>&1
    echo "EQUIV_END L=$L rc=$?"; grep -E "^(LAYER|LOGITS|MTP|LOSS|GRAD_WORST|PASS|FAIL)" /data/out/equiv_L${L}_$mode.log
    up /data/out/equiv_L${L}_$mode.log equiv_L${L}_$mode.log
    [ -f /data/out/equiv_L${L}_$mode.json ] && up /data/out/equiv_L${L}_$mode.json equiv_L${L}_$mode.json
  done
fi

if [ "$NNODES" -gt 1 ]; then
  # 等 0 号机的地址能解析（JobSet 的无头服务在 Pod 起来后才有记录）
  for i in $(seq 1 120); do getent hosts "$MASTER_ADDR" >/dev/null && break; sleep 5; done
  getent hosts "$MASTER_ADDR" || { echo MASTER_DNS_FAIL; exit 1; }
  DIST="--nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR"
fi
port=$MASTER_PORT
for spec in $RUNS; do
  IFS=: read -r name L steps extra <<< "$spec"; extra=${extra//,/ }
  echo "RUN_START $name L=$L steps=$steps extra=$extra $(date -u +%FT%TZ)"
  t0=$(date +%s)
  # --resume 的相对路径按 /data/runs 解析（如 --resume=ck-c/ckpt/step_10）
  extra=$(echo "$extra" | sed -E 's#--resume=([^/ ][^ ]*)#--resume=/data/runs/\1#')
  ( cd $G && torchrun $DIST --master_port $port --nproc_per_node ${NPROC:-8} train_hy4mg.py --hf /data/hf --tokens /data/ds/tokens.npy \
      --layers $L --steps $steps --name $name --out /data/runs --code "$CODE" --image "$IMAGE" $extra ) > /data/out/$name.log 2>&1
  rc=$?; port=$((port + 1))   # 每个运行换一个端口，免得上一个运行的端口还没释放
  echo "RUN_END $name rc=$rc seconds=$(( $(date +%s) - t0 ))"
  grep -E "^(META|STEP|LOAD_CHECK|PARAM_NORM_STEP0|MUON_PARTITION|CKPT_SAVED|CKPT_LOADED) " /data/out/$name.log; grep -E "^\[rank[0-9]+\]:.*(Error|error)|OOM" /data/out/$name.log | tail -5
  for f in meta.json metrics.jsonl load_check.json optimizer_partition.json DONE; do [ -f /data/runs/$name/$f ] && up /data/runs/$name/$f runs/$name/$f; done
  # data/：每 rank 的输入指纹（rank<r>.jsonl）与 --save-batches 的 batch（batches/rank<r>.{bin,jsonl}，TPU 回放用）；多机时每台传自己那些 rank 的
  ( cd /data/runs/$name && find data params_sha state_sha step_* modules.txt kd_*.pt qk_clip l1 choices -type f 2>/dev/null ) | while read -r f; do up /data/runs/$name/$f runs/$name/$f; done | grep -c PUT_FAIL
  up /data/out/$name.log runs/$name/train$NSFX.log
done
up /data/out/pod.log pod$NSFX.log
echo POD_DONE
