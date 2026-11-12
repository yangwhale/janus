#!/bin/bash
# 提交一次 TPU 训练（JobSet）：每个 TPU 主机拉代码包与 golden 数据，跑 train_hy3.py；进程 0 写运行目录，提交机取回后传到 GCS。
#
# 用法（在能 kubectl 到 TPU 集群的机器上）：
#   source <(bash stage.sh <maxtext 克隆>)      # 先打包上传代码，导出 MAXTEXT_PKG 等变量（代码不变就不用重跑）
#   PLATFORM=v5p bash run.sh <运行名> <步数> [train_hy3.py 参数…]
#   例：PLATFORM=v5p bash run.sh t-a 1000 --dump-steps 1,1000 --full
#       PLATFORM=v5p bash run.sh t-nohooks 1000 --no-hooks --dump-steps 1,1000
#   DRYRUN=1 只打印 JobSet，不提交。
#
# 可替换的环境变量（默认值见下）：
#   BUCKET       POC 桶（必填，gs://<POC 桶>）：下面几个路径的默认值都在它下面
#   PLATFORM     v5p（2x2x2，2 台主机）| v7x（2x2x1，1 台主机）；两者都是 8 个 JAX 设备
#   CONTEXT      TPU 集群的 kubectl 上下文（默认提交方式下必填）
#   NODEPOOL     节点池名（默认提交方式下必填）
#   IMAGE        MaxText 官方镜像 tpu_pre_training 0.2.5，按 digest 钉死
#   GOLDEN       golden 运行目录（GCS）：只拉 meta.json 与 data/（约 1.8 GB）
#   CKPT         convert_hf_to_maxtext.py 的产物（…/0/items）
#   HF_MANIFEST  HF 目录的 tensor_sha256.json（meta.checkpoint.manifest_sha256，与 golden 同一份）
#   OUT          运行目录上传到 $OUT/<运行名>（由提交机上传，提交机的账号要有写权限；Pod 只需要读）
#   HOLD         写运行目录的 Pod（JAX 进程 0 所在的那个）训练结束后最多等提交机取走的秒数（默认 3600），取走即退出
#   NO_WAIT=1    只提交不等；之后用 COLLECT_ONLY=1 bash run.sh <运行名> 0 取回并上传
#   SCRIPT       默认 train_hy3.py（训练）；layer_replay.py = 逐层隔离回放（参数：--layers、--grad 等，<步数> 不用）
#   GOLDEN_STEP1=1  另拉 golden 第 1 步的输入与逐层激活 / 激活梯度（layer_replay.py 要，约 6 GB）
#   XLA_DUMP=1   把编译后的 HLO（名字含 train_step 的模块）存到 Pod 的 /data/xla，取回时传到 $OUT/xla/<运行名>/（不放进运行目录）
#   LIBTPU_INIT_ARGS  默认空：对齐阶段先不加性能类 libtpu 参数（加了要先过「同程序只改这一个开关」的验证，见 docs/reference/environment.md）
#   XLA_FLAGS    透传给 Pod（默认空）；XLA_DUMP=1 时 dump 参数拼在后面。例：XLA_FLAGS=--xla_allow_excess_precision=false
#                是诊断臂（改舍入位置，数学级），不当噪声臂
#   TRAIN_DIR    Pod 里训练脚本所在目录（相对 janus 包根，默认 models/hunyuan3/frameworks/maxtext）。重跑目录重排前的旧提交时
#                改成当时的位置（2026-11-04 之前是 models/hunyuan3/tpu）
#   DATA_BUCKET  Pod 读的桶（gs://…，例如与 TPU 同区域的一份副本）：GOLDEN、CKPT、HF_MANIFEST 的默认值改到它下面；
#                OUT 仍在 BUCKET 下。不给时 DATA_BUCKET=BUCKET
#   NAMESPACE    默认 default
#   PIP_WHEEL    gs://…/x.whl：Pod 里先 pip install --no-deps（镜像自带的包比 MaxText 要求的旧时用，例如 tokamax）
#
# 站点配置：默认提交到自己的节点池（CONTEXT + NODEPOOL）。集群特有的提交设置（队列标签、节点选择、配额检查等）放在
# 仓库外的站点配置文件里，通过 JANUS_SITE_PROFILE=<文件路径> 指定（接口见 profile.sh 开头与 README「站点配置」）。
# 只换提交方式，Pod 里跑的命令、镜像、拓扑不变。站点配置的提交前检查约定认 ALLOW_QUEUE=1 跳过
set -euo pipefail
RUN=${1:?用法: run.sh <运行名> <步数> [train_hy3.py 参数…]}; STEPS=${2:?缺步数}; shift 2
PLATFORM=${PLATFORM:-v5p}
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=profile.sh
. "$HERE/profile.sh"
IMAGE=${IMAGE:-us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training@sha256:b152355a1425c28e99841c014703495059705de1733df8ac837ba43b4fe52fcf}
B=${BUCKET:?BUCKET=gs://<POC 桶>}/janus/hunyuan3-4l
DB=${DATA_BUCKET:-$BUCKET}/janus/hunyuan3-4l
GOLDEN=${GOLDEN:-$DB/golden/g1k-a}
CKPT=${CKPT:-$DB/maxtext/hy3-4l-maxtext/0/items}
HF_MANIFEST=${HF_MANIFEST:-$DB/hf/hy3-4l/tensor_sha256.json}
TRAIN_DIR=${TRAIN_DIR:-models/hunyuan3/frameworks/maxtext}
OUT=${OUT:-$B/tpu}
HOLD=${HOLD:-3600}
SCRIPT=${SCRIPT:-train_hy3.py}
case "$SCRIPT" in train_hy3.py|layer_replay.py) ;; *) echo "SCRIPT 只能是 train_hy3.py 或 layer_replay.py"; exit 1;; esac
if [ "$SCRIPT" = train_hy3.py ]; then
  PYARGS="--hf-manifest /data/tensor_sha256.json --steps $STEPS --janus-run /data/runs/$RUN"
else
  PYARGS="--out /data/runs/$RUN"
fi
STEP1=""
if [ -n "${GOLDEN_STEP1:-}" ]; then
  STEP1="mkdir -p /data/golden/\$(basename $GOLDEN)/step_1 && for f in index.json input final_norm layers.0 layers.1 layers.2 layers.3 layers.0.bwd layers.1.bwd layers.2.bwd layers.3.bwd buffer.layers.1 buffer.layers.2 buffer.layers.3; do case \$f in *.json) n=\$f;; *) n=\$f.safetensors;; esac; gcloud storage cp $GOLDEN/step_1/\$n /data/golden/\$(basename $GOLDEN)/step_1/; done && touch /data/golden/\$(basename $GOLDEN)/DONE"
fi
XF="${XLA_FLAGS:-}"
[ -n "${XLA_DUMP:-}" ] && XF="$XF --xla_dump_to=/data/xla --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*"
case "$XF" in *"'"*) echo "XLA_FLAGS 里不能有单引号"; exit 1;; esac
XLAENV=""
PIPLINE=""
# PIP_WHEEL=gs://…/x.whl：Pod 里先 pip install --no-deps 这个 wheel（例如镜像自带的 tokamax 比 MaxText 要求的旧）
[ -n "${PIP_WHEEL:-}" ] && PIPLINE="gcloud storage cp $PIP_WHEEL /tmp/ && pip install -q --no-deps /tmp/$(basename $PIP_WHEEL)"
[ -n "${XF// /}" ] && XLAENV="export XLA_FLAGS='${XF# }'"
: "${MAXTEXT_PKG:?先 source <(bash stage.sh …)}" "${MAXTEXT_PKG_SHA256:?}" "${MAXTEXT_COMMIT:?}"
: "${JANUS_PKG:?}" "${JANUS_PKG_SHA256:?}" "${JANUS_COMMIT:?}"
case "$PLATFORM" in
  v5p) NODES=2; TOPO=2x2x2; ACCEL=tpu-v5p-slice;;
  v7x) NODES=1; TOPO=2x2x1; ACCEL=tpu7x;;
  *) echo "PLATFORM 只能是 v5p 或 v7x"; exit 1;;
esac
profile_resolve   # profile.sh（可被站点配置覆盖）：定 CONTEXT / POOL / COLLECT，生成 JobSet 里随站点变化的几段
for a in "$@"; do case "$a" in *"'"*) echo "参数里不能有单引号：$a"; exit 1;; esac; done
NAME=janus-$(echo "$RUN" | tr '[:upper:]_.' '[:lower:]--' | cut -c1-40 | sed 's/-*$//')
G=$(basename "$GOLDEN")
ARGS="$*"

# 运行目录只在 JAX 进程 0 上写；进程 0 不一定在 TPU_WORKER_ID / JOB_COMPLETION_INDEX 为 0 的 Pod 上
#（v5p t-a 实测进程 0 在第 1 号 Pod），所以按「本机有没有运行目录」决定由哪个 Pod 交出结果。两种交法（COLLECT，profile_resolve 定）：
#   exec：不从 Pod 里上传。hostNetwork 下 Pod 的身份是节点的服务账号（Workload Identity 不生效），自己的节点池只有只读范围。
#         改由提交机上的 run.sh 用 kubectl exec tar 取走运行目录后，在提交机上传；取完它会建 /data/collected
#   gcs ：用于没有 pods/exec 权限、节点服务账号能写 $OUT 的集群（由站点配置选）。Pod 自己把运行目录、HLO、日志传到 $OUT，
#         打出 JANUS_READY 就退出（不占着切片等人取）；提交机从 $OUT 拉回本地
if [ "$COLLECT" = gcs ]; then
  HANDOFF="                if [ -d /data/runs/$RUN ]; then
                  up=0
                  gcloud storage rsync -r -q /data/runs/$RUN $OUT/$RUN || up=1
                  if [ -d /data/xla ]; then gcloud storage rsync -r -q /data/xla $OUT/xla/$RUN || up=1; fi
                  gcloud storage cp -q /data/train.log $OUT/logs/$RUN.train_w0.log || true
                  echo \"JANUS_READY rc=\$rc upload=\$up\"
                fi"
else
  HANDOFF="                if [ -d /data/runs/$RUN ]; then
                  echo \"JANUS_READY rc=\$rc\"
                  for _ in \$(seq $HOLD); do [ -e /data/collected ] && break; sleep 1; done
                fi"
fi
M=$(mktemp); trap 'rm -f "$M"' EXIT
cat > "$M" <<YAML
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: $NAME
  namespace: $NAMESPACE
$JS_META
spec:
  ttlSecondsAfterFinished: 7200
  failurePolicy: {maxRestarts: 0}
  replicatedJobs:
  - name: w
    replicas: 1
    template:
      spec:
        parallelism: $NODES
        completions: $NODES
        backoffLimit: 0
        template:
$POD_META
          spec:
            restartPolicy: Never
            # 必须 hostNetwork：不开时 jax.distributed 初始化卡住（TPU backend initialization 超过 60 s 不返回）
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            nodeSelector:
$NODE_SELECTOR
              cloud.google.com/gke-tpu-accelerator: $ACCEL
              cloud.google.com/gke-tpu-topology: $TOPO
            tolerations:
            - {key: google.com/tpu, operator: Exists, effect: NoSchedule}
            - {key: cloud.google.com/gke-spot, operator: Exists, effect: NoSchedule}
$TOLERATIONS
            containers:
            - name: jax
              image: $IMAGE
              command: ["bash", "-c"]
              args:
              - |
                set -e
                ulimit -c 0
                gcloud storage cp $MAXTEXT_PKG /tmp/mt.tgz
                echo '$MAXTEXT_PKG_SHA256  /tmp/mt.tgz' | sha256sum -c -
                cd /deps && rm -rf src/maxtext && tar --no-same-owner -xzf /tmp/mt.tgz
                gcloud storage cp $JANUS_PKG /tmp/janus.tgz
                echo '$JANUS_PKG_SHA256  /tmp/janus.tgz' | sha256sum -c -
                mkdir -p /janus && tar --no-same-owner -xzf /tmp/janus.tgz -C /janus
                $PIPLINE
                mkdir -p /data/golden/$G /data/runs
                gcloud storage cp $GOLDEN/meta.json /data/golden/$G/
                gcloud storage cp -r $GOLDEN/data /data/golden/$G/
                gcloud storage cp $HF_MANIFEST /data/tensor_sha256.json
                $STEP1
                $XLAENV
                export PYTHONPATH=/janus:/deps/src JAX_PLATFORMS=tpu,cpu TPU_STDERR_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0
                export LIBTPU_INIT_ARGS='${LIBTPU_INIT_ARGS:-}'
                export MAXTEXT_COMMIT=$MAXTEXT_COMMIT JANUS_COMMIT=$JANUS_COMMIT JANUS_IMAGE=$IMAGE TPU_TOPOLOGY=$TOPO
                set +e
                cd /janus && python3 $TRAIN_DIR/$SCRIPT --load $CKPT --golden /data/golden/$G \\
                  $PYARGS $ARGS 2>&1 | tee /data/train.log
                rc=\${PIPESTATUS[0]}
$HANDOFF
                echo "JANUS_EXIT rc=\$rc"
                exit \$rc
              resources:
                limits: {google.com/tpu: 4}
                requests: {google.com/tpu: 4}
YAML
sed -i '/^$/d' "$M"   # 随站点变化的几段为空时，删掉留下的空行
collect() {
  # 等写运行目录的 Pod（JAX 进程 0 所在）打出 JANUS_READY（训练已结束，不论成败），再取回运行目录与日志：
  #   exec：用 kubectl exec tar 取回，在提交机上传，再通知它退出
  #   gcs ：Pod 已自己传到 $OUT（JANUS_READY 带 upload=0），提交机从 $OUT 拉回本地。Pod 打完就退出，所以已结束的 Pod 的日志也要看
  # 所有 Pod 都结束了还没有 JANUS_READY 时直接报错。
  local K="kubectl --context $CONTEXT -n $NAMESPACE" pod="" line="" local_dir l
  echo "等训练结束（写运行目录的那个 Pod 打出 JANUS_READY）…"
  while true; do
    local pods ended=0 total=0
    pods=$($K get pods -l "jobset.sigs.k8s.io/jobset-name=$NAME" -o jsonpath='{range .items[*]}{.metadata.name} {.status.phase}{"\n"}{end}' 2>/dev/null || true)
    while read -r p phase; do
      [ -z "$p" ] && continue
      total=$((total + 1))
      case "$phase" in Succeeded|Failed) ended=$((ended + 1));; esac
      if [ -z "$line" ] && { [ "$phase" = Running ] || { [ "$COLLECT" = gcs ] && [ "$phase" != Pending ]; }; }; then
        l=$($K logs "$p" 2>/dev/null | grep -a '^JANUS_READY' | tail -1 || true)
        [ -n "$l" ] && { pod=$p; line=$l; }
      fi
    done <<< "$pods"
    [ -n "$line" ] && break
    if [ "$total" -gt 0 ] && [ "$ended" = "$total" ]; then
      echo "✗ 所有 Pod 都已结束，没有 Pod 打出 JANUS_READY（训练在写运行目录之前就失败了），看日志：$K logs -l jobset.sigs.k8s.io/jobset-name=$NAME"
      return 1
    fi
    sleep 30
  done
  echo "$pod: $line"
  local_dir=${LOCAL_RUNS:-$HOME/janus-runs}
  mkdir -p "$local_dir/logs"
  if [ "$COLLECT" = gcs ]; then
    case "$line" in *upload=0*) ;; *) echo "⚠ Pod 里上传 $OUT/$RUN 没有全部成功（$line），看 Pod 日志：$K logs $pod";; esac
    if gcloud storage ls "$OUT/$RUN/" >/dev/null 2>&1; then
      rm -rf "${local_dir:?}/$RUN"; mkdir -p "$local_dir/$RUN"
      gcloud storage rsync -r -q "$OUT/$RUN" "$local_dir/$RUN"
      [ -f "$local_dir/$RUN/DONE" ] || echo "⚠ $RUN 没有 DONE（训练没写完）"
      echo "运行目录 → $OUT/$RUN（Pod 上传；本地副本 $local_dir/$RUN）"
    else
      echo "⚠ $OUT/$RUN 不存在（训练在写运行目录之前就失败了，或上传失败）"
    fi
    if gcloud storage ls "$OUT/xla/$RUN/" >/dev/null 2>&1; then
      rm -rf "${local_dir:?}/xla/$RUN" && mkdir -p "$local_dir/xla/$RUN"
      gcloud storage rsync -r -q "$OUT/xla/$RUN" "$local_dir/xla/$RUN" && echo "HLO → $OUT/xla/$RUN"
    fi
    gcloud storage cp -q "$OUT/logs/$RUN.train_w0.log" "$local_dir/logs/" 2>/dev/null || $K logs "$pod" > "$local_dir/logs/$RUN.train_w0.log" 2>/dev/null || true
  else
    if $K exec "$pod" -- test -d "/data/runs/$RUN"; then
      rm -rf "${local_dir:?}/$RUN"
      $K exec "$pod" -- tar -C /data/runs -cf - "$RUN" | tar -C "$local_dir" -xf -
      [ -f "$local_dir/$RUN/DONE" ] || echo "⚠ $RUN 没有 DONE（训练没写完）"
      gcloud storage cp -r -q "$local_dir/$RUN" "$OUT/"
      echo "运行目录 → $OUT/$RUN（本地副本 $local_dir/$RUN）"
    else
      echo "⚠ Pod 里没有 /data/runs/$RUN（训练在写运行目录之前就失败了）"
    fi
    if $K exec "$pod" -- test -d /data/xla; then
      rm -rf "${local_dir:?}/xla/$RUN" && mkdir -p "$local_dir/xla/$RUN"
      $K exec "$pod" -- tar -C /data/xla -cf - . | tar -C "$local_dir/xla/$RUN" -xf -
      gcloud storage cp -r -q "$local_dir/xla/$RUN" "$OUT/xla/" && echo "HLO → $OUT/xla/$RUN"
    fi
    $K exec "$pod" -- cat /data/train.log > "$local_dir/logs/$RUN.train_w0.log" 2>/dev/null || true
    gcloud storage cp -q "$local_dir/logs/$RUN.train_w0.log" "$OUT/logs/" || true
    $K exec "$pod" -- touch /data/collected || true
  fi
  echo "日志 → $OUT/logs/$RUN.train_w0.log；JobSet 跑完后删：kubectl --context $CONTEXT -n $NAMESPACE delete jobset $NAME"
  local rc=${line#JANUS_READY rc=}; rc=${rc%% *}
  [ "$rc" = 0 ]
}
if [ -n "${DRYRUN:-}" ]; then cat "$M"; exit 0; fi
if [ -n "${COLLECT_ONLY:-}" ]; then collect; exit $?; fi
kubectl --context "$CONTEXT" get jobset "$NAME" -n "$NAMESPACE" >/dev/null 2>&1 && { echo "✗ JobSet $NAME 已存在：换运行名或先删掉"; exit 1; }
if gcloud storage ls "$OUT/$RUN/" >/dev/null 2>&1; then echo "✗ $OUT/$RUN 已有产物：运行目录写完即只读，换运行名"; exit 1; fi
profile_admit     # 提交前检查（默认不查；站点配置可以加，例如配额与排队）
kubectl --context "$CONTEXT" apply -f "$M"
echo "已提交 $NAME（$SITE_LABEL，$PLATFORM $TOPO，$NODES 台）→ $OUT/$RUN"
echo "看日志：kubectl --context $CONTEXT -n $NAMESPACE logs -f -l jobset.sigs.k8s.io/jobset-name=$NAME --max-log-requests 4"
if [ -n "${NO_WAIT:-}" ]; then echo "NO_WAIT：跑完用 COLLECT_ONLY=1 bash run.sh $RUN 0 取回"; exit 0; fi
collect
