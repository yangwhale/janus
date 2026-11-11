#!/bin/bash
# 提交一次 TPU 训练（JobSet）：每个 TPU 主机拉代码包、golden 数据与 HF 配置，跑 train_hy4.py；进程 0 写运行目录，取回后传到 GCS。
# 从混元 3 的 run.sh 改来（同一套提交、取回、站点配置接口，见 ../../../hunyuan3/frameworks/maxtext/run.sh 的说明）。
#
# 用法（在能 kubectl 到 TPU 集群的机器上）：
#   source <(bash stage.sh <maxtext 克隆>)      # 先打包上传代码，导出 MAXTEXT_PKG 等变量（代码不变就不用重跑）
#   STAGE=A GOLDEN=gs://<桶>/janus/hunyuan4/golden/ga-a PLATFORM=v5p bash run.sh <运行名> <步数> [train_hy4.py 参数…]
#   例：STAGE=A … bash run.sh t-a 300 --dump-steps 1,300 --full
#       STAGE=A … bash run.sh t-a-aot 1 --aot-only         # 真实切片上只编译、记编译期显存分析
#   DRYRUN=1 只打印 JobSet，不提交。
#
# 和混元 3 不同的环境变量：
#   STAGE        阶段 A / B / C / D / D1 / 4L / mini（必填；决定层数、MTP 与默认检查点；T2 用 B、D1）
#   YML          对齐 yml 文件名（本目录下；默认 hy4-4layers.yml，T1 用 hy4-mini-t1.yml，T2 用 hy4-t2-b.yml / hy4-t2-d1.yml）
#   GOLDEN       golden 运行目录（GCS，必填）：只拉 meta.json 与 data/
#   CKPT         默认 $DB/maxtext/hy4-<stage>-maxtext/0/items（convert_hf_to_maxtext.py --stage 的产物）
#   HF_DIR       截好的 HF 目录（默认 $DB/hf/hy4-4l）：拉它的 config.json 与 tensor_sha256.json
#   SCRIPT       train_hy4.py（默认，训练）| op_replay.py（L1 逐算子隔离回放；GOLDEN 指 ga1-*，<步数> 不用）
#                | 本目录下其余检查脚本（如 elementwise_ulp.py）：只传 --stage 与 --out /data/runs/<运行名>/<脚本名>.json，<步数> 不用
#   FP64_NPZ     op_replay.py 的 FP64 参照（GPU 侧 fp64_ops.py 存下的输出，gs://…）；给了才算 R
# 其余（BUCKET、DATA_BUCKET、PLATFORM、IMAGE、OUT、HOLD、NO_WAIT、COLLECT_ONLY、XLA_DUMP、XLA_FLAGS、LIBTPU_INIT_ARGS、
# NAMESPACE、JANUS_SITE_PROFILE）同混元 3。
# PIP_EXTRA="libtpu==<版本> …" Pod 里先 pip install（--no-deps）再训练，用来换 libtpu / jax 版本。
# XLA_DUMP=1 训练时另把训练步的优化后 HLO 拷进运行目录 hlo/，指纹记在 meta.json 的 hlo（train_hy4.py --hlo-dir）。
set -euo pipefail
RUN=${1:?用法: run.sh <运行名> <步数> [train_hy4.py 参数…]}; STEPS=${2:?缺步数}; shift 2
PLATFORM=${PLATFORM:-v5p}
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=profile.sh
. "$HERE/profile.sh"
IMAGE=${IMAGE:-us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training@sha256:b152355a1425c28e99841c014703495059705de1733df8ac837ba43b4fe52fcf}
STAGE=${STAGE:?STAGE=A|B|C|D|D1|4L|mini|mid}
STG=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
B=${BUCKET:?BUCKET=gs://<POC 桶>}/janus/hunyuan4
DB=${DATA_BUCKET:-$BUCKET}/janus/hunyuan4
GOLDEN=${GOLDEN:?GOLDEN=<golden 运行目录，gs://…>}
CKPT=${CKPT:-$DB/maxtext/hy4-$STG-maxtext/0/items}
HF_DIR=${HF_DIR:-$DB/hf/hy4-4l}
TRAIN_DIR=${TRAIN_DIR:-models/hunyuan4/frameworks/maxtext}
OUT=${OUT:-$B/tpu}
HOLD=${HOLD:-3600}
SCRIPT=${SCRIPT:-train_hy4.py}
STEP1=""
case "$SCRIPT" in
  train_hy4.py) PYARGS="--stage $STAGE --hf-config /data/config.json --hf-manifest /data/tensor_sha256.json --steps $STEPS --janus-run /data/runs/$RUN";;
  op_replay.py) PYARGS="--stage $STAGE --out /data/runs/$RUN/op_replay.json${FP64_NPZ:+ --fp64-npz /data/fp64_ops.npz}"
                # 逐算子回放要 golden 第 1 步的前向抓数（ga1-*：step_1/index.json + fwd.safetensors，约 4 GB）
                STEP1="mkdir -p /data/runs/$RUN /data/golden/\$(basename $GOLDEN)/step_1 && gcloud storage cp $GOLDEN/step_1/index.json $GOLDEN/step_1/fwd.safetensors /data/golden/\$(basename $GOLDEN)/step_1/ && touch /data/golden/\$(basename $GOLDEN)/DONE${FP64_NPZ:+ && gcloud storage cp $FP64_NPZ /data/fp64_ops.npz}";;
  *.py) # 其余本目录下的脚本（elementwise_ulp.py、optim_replay.py、layer_replay.py…）：只给 --stage 与 --out，输入由脚本自己取
        # （脚本用 parse_known_args，忽略下面统一附带的 --load / --golden）。结果写 /data/runs/$RUN/<脚本名>.json
        [ -f "$HERE/$SCRIPT" ] || { echo "SCRIPT=$SCRIPT 不在 $HERE"; exit 1; }
        case "$SCRIPT" in */*|*"'"*) echo "SCRIPT 只能是本目录下的文件名"; exit 1;; esac
        PYARGS="--stage $STAGE --out /data/runs/$RUN/${SCRIPT%.py}.json"
        STEP1=":";;   # 运行目录由脚本只在 JAX 进程 0 上建（交结果的 Pod 按「有没有运行目录」认）
  *) echo "SCRIPT 只能是本目录下的 .py（train_hy4.py、op_replay.py 或其余检查脚本）"; exit 1;;
esac
XF="${XLA_FLAGS:-}"
[ -n "${XLA_DUMP:-}" ] && XF="$XF --xla_dump_to=/data/xla --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*"
# 训练时把训练步的优化后 HLO 拷进运行目录 hlo/ 并在 meta.json 记指纹（train_hy4.py --hlo-dir），随运行目录一起取回
[ -n "${XLA_DUMP:-}" ] && [ "$SCRIPT" = train_hy4.py ] && PYARGS="$PYARGS --hlo-dir /data/xla"
case "$XF" in *"'"*) echo "XLA_FLAGS 里不能有单引号"; exit 1;; esac
XLAENV=""
[ -n "${XF// /}" ] && XLAENV="export XLA_FLAGS='${XF# }'"
PIPCMD=""   # PIP_EXTRA="<包==版本> …"：Pod 里先 pip install 这些包（换 libtpu / jax 版本用），装完记下 pip freeze 的相关行
case "${PIP_EXTRA:-}" in *"'"*) echo "PIP_EXTRA 里不能有单引号"; exit 1;; esac
[ -n "${PIP_EXTRA:-}" ] && PIPCMD="pip install -q --no-deps ${PIP_EXTRA} && (pip freeze | grep -iE '^(jax|jaxlib|libtpu)' || true)"
YMLENV=""   # YML=<本目录下的对齐 yml 文件名>（如 hy4-mini-t1.yml）：Pod 里设 HY4_YML，mt_common 改读它；不给就用 hy4-4layers.yml
[ -n "${YML:-}" ] && YMLENV="export HY4_YML=/janus/$TRAIN_DIR/$YML"
: "${MAXTEXT_PKG:?先 source <(bash stage.sh …)}" "${MAXTEXT_PKG_SHA256:?}" "${MAXTEXT_COMMIT:?}"
: "${JANUS_PKG:?}" "${JANUS_PKG_SHA256:?}" "${JANUS_COMMIT:?}"
case "$PLATFORM" in
  v5p) NODES=2; TOPO=2x2x2; ACCEL=tpu-v5p-slice;;
  v5p-16) NODES=4; TOPO=2x2x4; ACCEL=tpu-v5p-slice;;   # 16 个设备：全局批 16 时每设备微批 1、累积 1
  v7x) NODES=1; TOPO=2x2x1; ACCEL=tpu7x;;
  v7x-16) NODES=2; TOPO=2x2x2; ACCEL=tpu7x;;   # 8 片 16 个设备
  *) echo "PLATFORM 只能是 v5p、v5p-16、v7x 或 v7x-16"; exit 1;;
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
                mkdir -p /data/golden/$G /data/runs
                gcloud storage cp $GOLDEN/meta.json /data/golden/$G/
                gcloud storage cp -r $GOLDEN/data /data/golden/$G/
                gcloud storage cp $HF_DIR/tensor_sha256.json /data/tensor_sha256.json
                gcloud storage cp $HF_DIR/config.json /data/config.json
                $STEP1
                $PIPCMD
                $XLAENV
                $YMLENV
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
# 并行分块取回大文件：每块 256 MiB，FETCH_STREAMS 条 kubectl exec 流同时跑。每块先落成单独的临时文件，
# 核对这一块的字节数（kubectl exec 的流偶尔会被截断，截断的块只核总大小查不出来，会在拼好的文件里留下整段 0），
# 不对就重取这一块（最多 3 次）；拼好后再对 Pod 里的原文件核 sha256，不一致就整体失败（调用方退回 tar）
fetch_big() {
  local pod=$1 dst=$2; shift 2
  local f size nblk tmp want_sha got_sha
  for f in "$@"; do
    size=$($K exec "$pod" -- stat -c %s "/data/runs/$f")
    nblk=$(( (size + 268435455) / 268435456 ))
    tmp="$dst/.fetch.$$"; rm -rf "$tmp"; mkdir -p "$tmp" "$dst/$(dirname "$f")"
    seq 0 $((nblk - 1)) | xargs -P "${FETCH_STREAMS:-8}" -I{} sh -c '
      i=$1; want=$(( $3 - i * 268435456 )); [ $want -gt 268435456 ] && want=268435456
      for t in 1 2 3; do
        '"$K"' exec '"$pod"' -- dd if=/data/runs/'"$f"' bs=256M skip=$i count=1 iflag=fullblock status=none > '"$tmp"'/$i
        [ "$(stat -c %s '"$tmp"'/$i)" = "$want" ] && exit 0
      done
      echo "⚠ 第 $i 块取回 3 次都不完整" >&2; exit 1' _ {} "$f" "$size" || { rm -rf "$tmp"; return 1; }
    : > "$dst/$f"
    for i in $(seq 0 $((nblk - 1))); do cat "$tmp/$i" >> "$dst/$f"; done
    rm -rf "$tmp"
    want_sha=$($K exec "$pod" -- sha256sum "/data/runs/$f" | cut -d' ' -f1)
    got_sha=$(sha256sum "$dst/$f" | cut -d' ' -f1)
    [ "$want_sha" = "$got_sha" ] || { echo "⚠ $f 的 sha256 不符（Pod $want_sha，本地 $got_sha）"; return 1; }
  done
}

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
      # 大的运行目录（第 1 步诊断的梯度上 GB）取回时连接偶尔会断，tar 报 Unexpected EOF：最多重试 3 次，
      # 每次先删掉不完整的本地副本；以 Pod 里各文件的字节数核对
      for try in 1 2 3; do
        rm -rf "${local_dir:?}/$RUN"
        # 大于 2 GiB 的文件（第 1 步诊断的梯度，T2 上 48 GB）单条 kubectl exec 流只有约 20 MB/s，取回要 40 分钟、
        # 期间 Pod 一直占着 TPU：这些文件按块用 FETCH_STREAMS（默认 8）条 exec 流并行取，其余照旧用 tar
        # 第 1 次用并行分块；失败了第 2、3 次退回整目录 tar（不依赖 Pod 里的 dd）
        # FETCH_BIG_MIN（默认 +2G）：多大的文件走并行分块；连接不稳的节点（实测 v7x）设小一些，如 +64M，并且每次重试都分块
        big=""; { [ "$try" = 1 ] || [ -n "${FETCH_BIG_MIN:-}" ]; } && big=$($K exec "$pod" -- sh -c "cd /data/runs && find $RUN -type f -size ${FETCH_BIG_MIN:-+2G}")
        if $K exec "$pod" -- tar -C /data/runs -cf - $(for f in $big; do printf -- '--exclude=%s ' "$f"; done) "$RUN" | tar -C "$local_dir" -xf - \
           && fetch_big "$pod" "$local_dir" $big; then
          want=$($K exec "$pod" -- sh -c "cd /data/runs && find $RUN -type f -exec stat -c %s {} + | awk '{s+=\$1} END {print s}'")
          got=$(find "$local_dir/$RUN" -type f -exec stat -c %s {} + | awk '{s+=$1} END {print s}')
          [ "$want" = "$got" ] && break
          echo "⚠ 取回的字节数不符（Pod $want，本地 $got），第 $try 次重试"
        else
          echo "⚠ 取回中断，第 $try 次重试"
        fi
      done
      [ -f "$local_dir/$RUN/DONE" ] || echo "⚠ $RUN 没有 DONE（训练没写完）"
      UPLOAD_RUN=1   # 上传放到 Pod 退出之后（下面 touch /data/collected 以后），不再占着 TPU
      echo "运行目录 → $OUT/$RUN（本地副本 $local_dir/$RUN；Pod 放行后再上传）"
    else
      echo "⚠ Pod 里没有 /data/runs/$RUN（训练在写运行目录之前就失败了）"
    fi
    if $K exec "$pod" -- test -d /data/xla; then
      rm -rf "${local_dir:?}/xla/$RUN" && mkdir -p "$local_dir/xla/$RUN"
      # 完整的 XLA dump 只作旁证（训练步的优化后 HLO 已拷进运行目录 hlo/），取回断了不算失败
      if $K exec "$pod" -- tar -C /data/xla -cf - . | tar -C "$local_dir/xla/$RUN" -xf -; then
        gcloud storage cp -r -q "$local_dir/xla/$RUN" "$OUT/xla/" && echo "HLO → $OUT/xla/$RUN"
      else
        echo "⚠ XLA dump 取回中断（运行目录不受影响）"
      fi
    fi
    $K exec "$pod" -- cat /data/train.log > "$local_dir/logs/$RUN.train_w0.log" 2>/dev/null || true
    gcloud storage cp -q "$local_dir/logs/$RUN.train_w0.log" "$OUT/logs/" || true
    $K exec "$pod" -- touch /data/collected || true
    if [ -n "${UPLOAD_RUN:-}" ]; then
      if [ -n "${UPLOAD_ASYNC:-}" ]; then   # 由调用方（如带锁的提交脚本）放锁后自己等这个后台上传
        nohup gcloud storage cp -r -q "$local_dir/$RUN" "$OUT/" > "$local_dir/logs/$RUN.upload.log" 2>&1 &
        echo "上传在后台：$local_dir/logs/$RUN.upload.log"
      else
        gcloud storage cp -r -q "$local_dir/$RUN" "$OUT/"
      fi
    fi
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
