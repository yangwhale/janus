#!/bin/bash
# 把 MaxText 代码与 janus 代码各打成一个包传到 GCS，打印 run.sh 要的环境变量（可以直接 source）。
#
# 用法：source <(bash stage.sh <maxtext 本地克隆> [maxtext 提交，默认 origin/hunyuan])
# 只打 git 里已提交的内容（git archive）：工作树里没提交的改动不进包；janus 要求工作树干净。
# 同一个提交每次打出来的字节相同（gzip -n 不写时间戳），sha256 可以和提交一起记下来。
set -euo pipefail
MT=${1:?maxtext 本地克隆}; MT_REV=${2:-origin/hunyuan}
JANUS=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DST=${DST:-${DATA_BUCKET:-${BUCKET:?BUCKET=gs://<POC 桶>}}/janus/pkg}   # 给了 DATA_BUCKET（Pod 读的那份，例如与 TPU 同区域的副本）就传到它下面
[ -z "$(git -C "$JANUS" status --porcelain)" ] || { echo "✗ janus 工作树有未提交的改动" >&2; exit 1; }
MSHA=$(git -C "$MT" rev-parse "$MT_REV"); JSHA=$(git -C "$JANUS" rev-parse HEAD)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git -C "$MT" archive --format=tar "$MSHA" src/maxtext | gzip -n > "$T/mt.tgz"
git -C "$JANUS" archive --format=tar "$JSHA" janus models | gzip -n > "$T/janus.tgz"
MP=$DST/maxtext-${MSHA:0:10}.tgz; JP=$DST/janus-${JSHA:0:10}.tgz
gcloud storage cp -q "$T/mt.tgz" "$MP" >&2
gcloud storage cp -q "$T/janus.tgz" "$JP" >&2
echo "export MAXTEXT_COMMIT=$MSHA MAXTEXT_PKG=$MP MAXTEXT_PKG_SHA256=$(sha256sum "$T/mt.tgz" | cut -d' ' -f1)"
echo "export JANUS_COMMIT=$JSHA JANUS_PKG=$JP JANUS_PKG_SHA256=$(sha256sum "$T/janus.tgz" | cut -d' ' -f1)"
