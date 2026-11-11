#!/bin/bash
# run.sh / run_regression.sh 共用：定 kubectl 目标、节点选择与「哪些作业是我们的」。只被 source。
#
# 默认：提交到自己的节点池（CONTEXT、NODEPOOL 必填）。「我们的作业」= 同一个节点池上名字以 janus- 开头的 JobSet。
#
# 站点配置：集群特有的提交设置（队列标签、节点选择、配额检查等）放在仓库外的站点配置文件里，通过
# JANUS_SITE_PROFILE=<文件路径> 指定。本文件先定义下面的默认函数，再 source 那个文件；它重定义哪个函数就用哪个。
# 接口见 README「站点配置（JANUS_SITE_PROFILE）」，最小示例见 site-profile.example.sh。
#
#   profile_resolve      设好 CONTEXT、NAMESPACE、POOL（只用于显示）、SITE_LABEL（只用于显示）、COLLECT（exec | gcs），
#                        以及 JobSet 里随站点变化的四段：JS_META（JobSet 的 labels / annotations）、POD_META（Pod 模板的 metadata）、
#                        NODE_SELECTOR（追加的 nodeSelector 行）、TOLERATIONS（追加的 tolerations 行）。空的段在 JobSet 里删掉
#   profile_ours         打印我们还没结束的 janus-* JobSet 名字，一行一个
#   profile_busy_reason  日常回归提交前：打印不该提交的理由（空 = 可以提交）
#   profile_admit        run.sh 提交前检查：不该提交就 exit 2。约定认 ALLOW_QUEUE=1 跳过（日常回归 ALLOW_BUSY=1 时这样传）
#   profile_delete       删一个 JobSet，只删我们的
#
# 调用方先设好 PLATFORM、NODES、ACCEL（run.sh 的 case 里设），再调 profile_resolve。

profile_resolve() {
  NAMESPACE=${NAMESPACE:-default}
  JS_META=""; POD_META=""; TOLERATIONS=""
  CONTEXT=${CONTEXT:?CONTEXT=<TPU 集群的 kubectl 上下文>}
  POOL=${NODEPOOL:?NODEPOOL=<节点池名>}
  SITE_LABEL=nodepool
  NODE_SELECTOR="              cloud.google.com/gke-nodepool: $POOL"
  COLLECT=exec   # 提交机 kubectl exec 取回再上传（节点池只读范围）
}

profile_ours() {
  # 打印同一个节点池上、还没结束（TERMINALSTATE 为空，含排队中）的 janus-* JobSet 名字，一行一个
  local js
  js=$(kubectl --context "$CONTEXT" -n "$NAMESPACE" get jobset -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.replicatedJobs[0].template.spec.template.spec.nodeSelector.cloud\.google\.com/gke-nodepool}{"\t"}{.status.terminalState}{"\n"}{end}' 2>/dev/null) || {
    echo "✗ 读不到 $CONTEXT 上的 JobSet 列表" >&2; return 1; }
  awk -F'\t' -v pool="$POOL" '$1 ~ /^janus-/ && $2 == pool && $3 == "" {print $1}' <<< "$js"
}

profile_busy_reason() {
  # 一个节点池只有一个切片：同一个节点池上别的 janus-* 还没结束就不提交；别的节点池上的作业不算
  local ours
  ours=$(profile_ours) || { echo "读不到 JobSet 列表"; return 0; }
  [ -z "$ours" ] || echo "节点池 $POOL 上有别的 JobSet 在跑：$(echo $ours)"
}

profile_admit() {
  return 0   # 默认不查：节点池是自己的（日常回归另用 profile_busy_reason 查同节点池）
}

profile_delete() {
  kubectl --context "$CONTEXT" -n "$NAMESPACE" delete jobset "$1" --ignore-not-found >/dev/null 2>&1 || true
}

if [ -n "${JANUS_SITE_PROFILE:-}" ]; then
  [ -f "$JANUS_SITE_PROFILE" ] || { echo "✗ JANUS_SITE_PROFILE=$JANUS_SITE_PROFILE 不存在"; exit 1; }
  # shellcheck disable=SC1090
  . "$JANUS_SITE_PROFILE"
fi
