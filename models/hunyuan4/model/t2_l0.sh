#!/bin/bash
# 混元 4 · T2（真实权重截层 B / D1）的 L0 载入核对，CPU 上跑，做法同阶段 A–D 与 T1：
#   TPU 一侧：转换好的 MaxText 检查点反向取回成 HF 张量，逐张量对 HF 目录的 tensor_sha256.json（目录 l0-ckpt-convert-<阶段>.yaml）、
#             RoPE 列去交错逐位（verify_ckpt.py rope）；用**当前** MaxText 提交的抽象模型建映射，所以也核了检查点的参数树
#             和现在的代码对得上（AdamW 时期转换的 B 检查点要靠这一步确认还能用）。
#   GPU 一侧：每个 GPU 运行开训前的载入自检 load_check.json（每个参数 == 检查点张量、两边 FP64 平方和相等，train_hy4mg.py 写），
#             以及 meta.json 记的清单 sha256 与 TPU 用的 tensor_sha256.json 是同一份。两边都对同一份清单逐位，即两边初始权重逐位相同（传递）。
#
# 用法：bash t2_l0.sh <阶段 B|D1> <HF 目录（本地，截层的 hy4-4l）> <MaxText 检查点 …/0/items> <输出根> [GPU 运行目录（本地）…]
#   例：bash t2_l0.sh D1 /data/hy4-4l gs://<POC 桶>/janus/hunyuan4/maxtext/hy4-d1-maxtext/0/items /tmp/t2-l0 /data/golden-t2/gd2-a
# 产物：<输出根>/janus-runs/hy4-<阶段>-{hf,maxtext-ckpt}/（janus 运行目录）、<输出根>/results/<结果名>/（results.json、report.md）、
#       <输出根>/gpu-load-check-<阶段>.json（GPU 载入自检汇总）。任何一步不过，退出码非 0。
# 要 MaxText 环境（jax、maxtext）；每一步都是重进程时，外面套 systemd-run --user --scope -p MemoryMax=250G。
set -euo pipefail
STAGE=${1:?阶段 B|D1}; HF=${2:?HF 目录}; CKPT=${3:?MaxText 检查点}; OUT=${4:?输出根}; shift 4
case "$STAGE" in B|D1) ;; *) echo "T2 只有阶段 B、D1"; exit 1;; esac
OUT=$(realpath -m "$OUT")   # 下面会 cd，输出根转成绝对路径
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
PY=${PYTHON:-python3}
stg=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
CAT=models/hunyuan4/alignments/02-real-weight/catalogs/l0-ckpt-convert-$stg.yaml   # 相对仓库根（结果文件里记相对路径）
[ "$STAGE" = B ] && CAT=models/hunyuan4/alignments/01-mini/catalogs/l0-ckpt-convert-b.yaml
RUNS=$OUT/janus-runs; mkdir -p "$RUNS" "$OUT/results"
unset HY4_YML   # 转换与反向取回用默认的 hy4-4layers.yml（只看参数树，与优化器设置无关）
export JAX_PLATFORMS=cpu PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$HERE"

echo "== 1. HF 原件中属于阶段 $STAGE 的张量：逐张量重算 sha256 对 tensor_sha256.json"
$PY verify_ckpt.py hf-manifest --hf "$HF" --stage "$STAGE" --janus-run "$RUNS/hy4-$stg-hf"
echo "== 2. MaxText 检查点反向取回（当前 MaxText 提交的映射）"
$PY verify_ckpt.py maxtext --ckpt "$CKPT" --hf "$HF" --stage "$STAGE" --janus-run "$RUNS/hy4-$stg-maxtext-ckpt"
echo "== 3. RoPE 列去交错逐位"
$PY verify_ckpt.py rope --ckpt "$CKPT" --hf "$HF" --stage "$STAGE" --out "$OUT/results/hy4-$stg-l0-rope"
echo "== 4. janus 正式检查：逐张量逐位 + dtype 清单（$CAT）"
(cd "$REPO" && $PY -m janus run "$CAT" --root "$RUNS" --out "$OUT/results/hy4-$stg-l0-ckpt-convert")

echo "== 5. GPU 载入自检（$# 个运行）"
$PY - "$HF/tensor_sha256.json" "$OUT/gpu-load-check-$stg.json" "$@" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, out, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
want = hashlib.sha256(manifest.read_bytes()).hexdigest()
rows, ok_all = [], True
for r in runs:
    d = Path(r)
    lc = json.loads((d / "load_check.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    got = (meta.get("checkpoint") or {}).get("tensor_sha256_manifest")
    sq = lc.get("sumsq_rank0") or {}
    ok = bool(lc.get("pass")) and lc.get("n_mismatch_all_ranks") == 0 and sq.get("model") == sq.get("ckpt") and got == want
    ok_all &= ok
    rows.append({"run": d.name, "ok": ok, "pass": lc.get("pass"), "n_checked_rank0": lc.get("n_checked_rank0"),
                 "n_mismatch_all_ranks": lc.get("n_mismatch_all_ranks"), "sumsq_rank0": sq,
                 "manifest_sha256": got, "manifest_sha256_expected": want})
    print(("OK  " if ok else "FAIL"), d.name, rows[-1]["n_checked_rank0"], rows[-1]["n_mismatch_all_ranks"])
out.write_text(json.dumps({"kind": "gpu_load_check", "manifest_sha256": want, "ok": ok_all, "runs": rows},
                          ensure_ascii=False, indent=1))
sys.exit(0 if ok_all else 1)
PY
echo "L0 阶段 $STAGE：全部通过"
