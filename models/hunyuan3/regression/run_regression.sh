#!/bin/bash
# 混元 3 · 日常回归 · 手动跑一档：提交训练（复用 ../frameworks/maxtext/run.sh）→ 取回 → janus 对 LKG → 写回归报告 → 上传结果。
# 不排程：现在 TPU 预算不够每天跑，只手动触发（标准第六节的节奏表是目标，见 README 第三节）。
#
# 用法（在能 kubectl 到 TPU 集群的机器上，仓库根目录或任意目录都行）：
#   export BUCKET=gs://<POC 桶> CONTEXT=<TPU 集群的 kubectl 上下文>
#   source <(bash models/hunyuan3/frameworks/maxtext/stage.sh <maxtext 克隆>)   # 打包被测提交（同 run.sh）
#   bash models/hunyuan3/regression/run_regression.sh T0 <运行名> --first-lkg           # 建首个 LKG：同一提交双跑，两次逐位
#   bash models/hunyuan3/regression/run_regression.sh T0 <运行名> [--lkg <LKG 运行名>]  # 日常：对 LKG（默认取 lkg.yaml 里这一档的最新一条）
#
# 选项：
#   --lkg <运行>         LKG 运行名（不给就读 lkg.yaml）
#   --first-lkg          建首个 LKG：跑 <运行名>-a、<运行名>-b 两次，b 对 a 判；全部逐位才是 LKG 候选，打印 lkg.yaml 条目
#   --declare <类型>     PR 声明：数学不变（默认）| 执行级 | 数学级 | 升级依赖（标准第三节；执行级 / 数学级都属「会改数值」）
#   --scope <行>         声明范围：清单里会变的行，逗号分隔，如 R02,R05,R08,R15
#   --skip-train         不提交训练，只用本地已有的运行目录重判（LOCAL_RUNS 下）
# 环境变量：
#   PLATFORM（默认 v5p）、NODEPOOL、IMAGE、LIBTPU_INIT_ARGS：原样传给 run.sh。「有别的作业在跑」默认只看同一个节点池
#                        （回归最好用专用节点池，不和对齐实验抢同一个切片）
#   JANUS_SITE_PROFILE   站点配置文件（见 run.sh 开头与 ../frameworks/maxtext/README.md「站点配置」）：给了时「有别的作业在跑」
#                        改用站点配置的 profile_busy_reason，删 JobSet 用它的 profile_delete（只删我们的）
#   EXTRA_OVERRIDES      追加的 MaxText key=value（空格分隔），例如故意改数值的对照：routed_gate_bf16_rounding=true
#                        （改了它就和档位定义不同，R02 / R05 会报，要配 --declare 与 --scope）
#   LOCAL_RUNS           本地运行根（默认 ~/janus-runs，与 run.sh 相同）
#   WAIT_BUSY=<秒>       集群上有别的 janus-* JobSet 在跑时最多等这么久再提交（默认 0：直接退出）
#   ALLOW_BUSY=1         集群上有别的 janus-* JobSet 在跑也提交（默认拒绝：同一个切片，排队会拖长、也可能抢掉别人的）
# 退出码：0 通过；1 数值失败；2 工具或跑不完；3 无结论（门槛待标定）
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
FW=$REPO/models/hunyuan3/frameworks/maxtext
TIER=${1:?用法: run_regression.sh <T0|T1> <运行名> [--lkg <运行>] [--first-lkg] [--declare <类型>] [--scope <行>] [--skip-train]}
RUN=${2:?缺运行名}; shift 2
LKG=""; FIRST=""; DECLARE=数学不变; SCOPE=""; SKIP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --lkg) LKG=$2; shift 2;;
    --first-lkg) FIRST=1; shift;;
    --declare) DECLARE=$2; shift 2;;
    --scope) SCOPE=$2; shift 2;;
    --skip-train) SKIP=1; shift;;
    *) echo "不认识的参数：$1"; exit 2;;
  esac
done
case "$DECLARE" in 数学不变|执行级|数学级|升级依赖) ;; *) echo "--declare 只能是 数学不变|执行级|数学级|升级依赖"; exit 2;; esac
PLATFORM=${PLATFORM:-v5p}
case "$PLATFORM" in v5p) NODES=2; ACCEL=tpu-v5p-slice;; *) NODES=1; ACCEL=tpu7x;; esac
# shellcheck source=../frameworks/maxtext/profile.sh
. "$FW/profile.sh"
profile_resolve   # 与 run.sh 同一套（可被站点配置覆盖）：定 CONTEXT、NAMESPACE、POOL
export CONTEXT
LOCAL_RUNS=${LOCAL_RUNS:-$HOME/janus-runs}
PY=${PYTHON:-python3}
eval "$($PY "$HERE/regress.py" args "$TIER")"
DATE=$(TZ=Asia/Hong_Kong date +%Y%m%d)
B=${BUCKET:?BUCKET=gs://<POC 桶>}/janus/hunyuan3-4l
# 运行目录与结果都放在 <模型>/regression/<平台>/<档>/<日期>/ 下（标准第十一节）：运行目录在 runs/<运行名>/，结果在 <运行名>/
DEST=$B/regression/$PLATFORM/$TIER/$DATE
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"

busy_check() {
  # 不该提交的理由由 profile.sh 的 profile_busy_reason 给：默认是同一个节点池上别的 janus-* JobSet 还没结束（含 Pending）
  #（一个节点池只有一个切片；别的节点池上的作业不算）；站点配置可以换成自己的检查。
  # WAIT_BUSY=<秒> 时每 30 秒再看一次，最多等这么久
  local waited=0 why
  while true; do
    profile_ours >/dev/null || exit 2
    why=$(profile_busy_reason)
    [ -z "$why" ] || [ -n "${ALLOW_BUSY:-}" ] && return 0
    if [ "$waited" -ge "${WAIT_BUSY:-0}" ]; then
      echo "✗ $why（等它们结束、WAIT_BUSY=<秒> 让脚本等，或 ALLOW_BUSY=1 强行排队）"; exit 2
    fi
    [ "$waited" = 0 ] && echo "… $why，最多等 ${WAIT_BUSY} 秒"
    sleep 30; waited=$((waited + 30))
  done
}

train() {   # $1 = 运行名
  local name=$1
  if [ -n "$SKIP" ]; then
    [ -f "$LOCAL_RUNS/$name/DONE" ] || { echo "✗ --skip-train 但本地没有 $LOCAL_RUNS/$name"; exit 2; }
    return 0
  fi
  busy_check
  echo "== 提交 $name（$TIER，$TIER_STEPS 步，$PLATFORM，$SITE_LABEL，节点池 $POOL）"
  local rc=0 np=()
  [ -n "${NODEPOOL:-}" ] && np+=(NODEPOOL="$NODEPOOL")
  [ -n "${JANUS_SITE_PROFILE:-}" ] && np+=(JANUS_SITE_PROFILE="$JANUS_SITE_PROFILE")
  # ALLOW_BUSY=1 时也跳过 run.sh 里站点配置的提交前检查（ALLOW_QUEUE=1）
  env OUT="$DEST/runs" XLA_DUMP=1 LOCAL_RUNS="$LOCAL_RUNS" PLATFORM="$PLATFORM" "${np[@]}" \
    ${ALLOW_BUSY:+ALLOW_QUEUE=1} \
    bash "$FW/run.sh" "$name" "$TIER_STEPS" "${TIER_TRAIN_ARGS[@]}" "${EXTRA[@]}" || rc=$?
  # 只删自己提交的 JobSet（名字规则同 run.sh）
  local js
  js=janus-$(echo "$name" | tr '[:upper:]_.' '[:lower:]--' | cut -c1-40 | sed 's/-*$//')
  profile_delete "$js"
  [ -f "$LOCAL_RUNS/$name/DONE" ] || { echo "✗ $name 没有跑完（R01 不过：先看作业日志分清基础设施还是代码）"; exit 2; }
  return $rc
}

fetch_lkg() {   # LKG 运行目录不在本地时，按 lkg.yaml 的 location 取回
  local name=$1 loc
  [ -f "$LOCAL_RUNS/$name/DONE" ] && return 0
  loc=$($PY - "$HERE/lkg.yaml" "$name" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
for e in d.get("entries") or []:
    if e.get("run") == sys.argv[2]:
        print(e["location"]); break
PY
)
  [ -n "$loc" ] || { echo "✗ 本地没有 $name，lkg.yaml 里也没有它的位置"; exit 2; }
  loc=${loc//gs:\/\/<POC 桶>/${BUCKET%/}}
  echo "== 取回 LKG $name ← $loc"
  gcloud storage cp -r -q "${loc%/}" "$LOCAL_RUNS/"
}

fetch_golden() {
  for g in "${TIER_GOLDEN[@]}"; do
    [ -f "$LOCAL_RUNS/$g/DONE" ] && continue
    mkdir -p "$LOCAL_RUNS/$g"
    for f in meta.json metrics.jsonl DONE; do gcloud storage cp -q "$B/golden/$g/$f" "$LOCAL_RUNS/$g/"; done
  done
}

fetch_noise() {   # 目录里 runs.noise 用到的标定运行（容差门的噪声参照）：只取 meta.json、metrics.jsonl、DONE，位置写在 calibration.yaml
  local name loc
  while IFS=$'\t' read -r name loc; do
    [ -z "$name" ] && continue
    [ -f "$LOCAL_RUNS/$name/DONE" ] && continue
    [ -n "$loc" ] || { echo "✗ 噪声参照 $name 不在本地，calibration.yaml 里也没有它的位置"; exit 2; }
    loc=${loc//gs:\/\/<POC 桶>/${BUCKET%/}}
    mkdir -p "$LOCAL_RUNS/$name"
    for f in meta.json metrics.jsonl DONE; do gcloud storage cp -q "${loc%/}/$f" "$LOCAL_RUNS/$name/"; done
  done < <(cd "$REPO" && $PY - "$TIER_CATALOG" "$HERE/calibration.yaml" <<'PY'
import sys, yaml
from pathlib import Path
from janus.catalog import load
cat = load(sys.argv[1])
p = Path(sys.argv[2])
locs = {}
for e in ((yaml.safe_load(p.read_text()) or {}).get("entries") or []) if p.exists() else []:
    locs.update(e.get("runs") or {})
for n in dict.fromkeys(x for pr in cat.runs.get("noise") or [] for x in pr):
    print(f"{n}\t{locs.get(n, '')}")
PY
)
}

judge() {   # $1 = 被测 $2 = LKG $3 = 结果目录名
  local test=$1 lkg=$2 res=$LOCAL_RUNS/regression-results/$3
  rm -rf "$res"; mkdir -p "$res"
  echo "== janus：$test 对 LKG $lkg"
  (cd "$REPO" && $PY -m janus run "$TIER_CATALOG" --root "$LOCAL_RUNS" --test "$test" --lkg "$lkg" --out "$res/janus" --verbose) || true
  local rc=0 extra=()
  [ -n "$FIRST" ] && extra+=(--first-lkg)
  [ -d "$LOCAL_RUNS/regression-results/$lkg" ] && extra+=(--lkg-results "$LOCAL_RUNS/regression-results/$lkg")
  (cd "$REPO" && $PY "$HERE/regress.py" report --tier "$TIER" --janus "$res/janus" --root "$LOCAL_RUNS" --test "$test" \
     --lkg "$lkg" --out "$res" --platform "$PLATFORM" --declare "$DECLARE" --scope "$SCOPE" "${extra[@]}") || rc=$?
  gcloud storage cp -r -q "$res" "$DEST/" && echo "结果 → $DEST/$3/"
  return $rc
}

fetch_golden
fetch_noise
if [ -n "$FIRST" ]; then
  A=$RUN-a; Bn=$RUN-b
  train "$A"; train "$Bn"
  rc=0; judge "$Bn" "$A" "$RUN" || rc=$?
  if [ "$rc" = 0 ]; then
    echo "== 首个 LKG 候选：$A（与 $Bn 同一提交双跑逐位）。lkg.yaml 条目（批准后追加进 $HERE/lkg.yaml 的 entries）："
    (cd "$REPO" && $PY "$HERE/regress.py" lkg-entry --tier "$TIER" --root "$LOCAL_RUNS" --run "$A" --pair "$Bn" --platform "$PLATFORM" \
       --location "gs://<POC 桶>/janus/hunyuan3-4l/regression/$PLATFORM/$TIER/$DATE/runs/$A/" \
       --results "models/hunyuan3/regression/results/$RUN/" --reason "首个 LKG：同一提交双跑逐位（$A、$Bn）")
  fi
  exit $rc
fi

if [ -z "$LKG" ]; then
  LKG=$($PY - "$HERE/lkg.yaml" "$PLATFORM" "$TIER" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1])
d = yaml.safe_load(p.read_text()) if p.exists() else {}
hits = [e for e in (d or {}).get("entries") or []
        if e["key"].get("platform") == sys.argv[2] and e["key"].get("tier") == sys.argv[3] and e["key"].get("variant", "main") == "main"
        and e.get("status") != "superseded"]   # 被取代的旧条目只留作历史，不当默认 LKG
print(hits[-1]["run"] if hits else "")
PY
)
  [ -n "$LKG" ] || { echo "✗ lkg.yaml 里没有 $PLATFORM / $TIER 的 LKG：先 --first-lkg 建一个，或用 --lkg 指定"; exit 2; }
fi
fetch_lkg "$LKG"
train "$RUN"
judge "$RUN" "$LKG" "$RUN"
