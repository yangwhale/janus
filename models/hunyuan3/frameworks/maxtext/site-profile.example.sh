#!/bin/bash
# 站点配置的最小示例（值都是占位）：复制到仓库外，按自己的集群改，再用 JANUS_SITE_PROFILE=<那个文件> 指定。
# 被 profile.sh 在定义完默认函数之后 source；这里重定义的函数覆盖默认实现，没重定义的照旧（接口见 profile.sh 开头）。
# 不要把真实的集群名、队列名、节点池名、项目、桶名写回仓库。

profile_resolve() {
  NAMESPACE=${NAMESPACE:-<命名空间>}
  CONTEXT=${CONTEXT:?CONTEXT=<TPU 集群的 kubectl 上下文>}
  POOL="(由集群分配)"
  SITE_LABEL=example-site
  # JobSet 的 metadata 里追加的 labels / annotations（两格缩进，接在 namespace 后面）
  JS_META="  labels:
    example.com/queue: <队列名>
  annotations:
    example.com/owner: <提交人>"
  # Pod 模板的 metadata（十格缩进起）；不需要就留空
  POD_META=""
  # 追加的 nodeSelector / tolerations 行（十四格 / 十二格缩进）；加速器与拓扑两行 run.sh 已经写了
  NODE_SELECTOR="              example.com/capacity-pool: <容量池>"
  TOLERATIONS="            - {key: example.com/capacity-pool, operator: Exists, effect: NoSchedule}"
  # exec：提交机 kubectl exec 取回运行目录再上传；gcs：Pod 自己传到 $OUT（没有 pods/exec 权限、节点能写桶时用）
  COLLECT=exec
}

profile_admit() {
  # 提交前检查，例如配额余量、有没有别人在排队：不该提交就打印理由并 exit 2；ALLOW_QUEUE=1 时跳过
  [ -n "${ALLOW_QUEUE:-}" ] && return 0
  : # <站点自己的检查>
}

# 「哪些作业是我们的」不按节点池认时（例如按某个注解认），另外重定义 profile_ours、profile_busy_reason 与 profile_delete。
