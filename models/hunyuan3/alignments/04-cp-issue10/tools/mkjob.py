#!/usr/bin/env python3
"""issue #10 FineWeb-Edu 长跑的作业生成器：在一个 v5p-8（2x2x2，2 台主机）节点池上依次跑几次 MaxText 训练，产物留在 Pod 的 /tmp/out。

  BUCKET=gs://<POC 桶> MODE=proxy|rw4 python3 mkjob.py <JobSet 名> <节点池> '<运行名>|<MaxText key=value …>|<追加的 LIBTPU 参数>' … > job.yaml

MODE=proxy：随机初始化的缩层混元 3（hunyuan3-smoke 覆盖成 16 层），
            代码 maxtext hunyuan@e2c18718。
MODE=rw4：混元 3 真实权重截 4 层（models/hunyuan3/frameworks/maxtext/hy3-4layers.yml 那份检查点），代码 hunyuan@8ac1d175。
两种都用 FineWeb-Edu sample-10BT 第一个 parquet 分片在线切词（packing=True），镜像 tpu_pre_training 0.2.5 另装 tokamax 0.0.16。
以 ! 开头的参数原样写进脚本（例如在两次运行之间拷存档）。产物用 kubectl exec 取回（节点池只有只读存储范围）。
"""
import os
import sys
import textwrap

job, pool, *runs = sys.argv[1:]
MODE = os.environ.get("MODE", "proxy")
TOPO = os.environ.get("TOPO", "2x2x2")
NHOST = {"2x2x2": 2, "2x2x4": 4}[TOPO]
OUTDIR = os.environ.get("OUTDIR", "/tmp/out")
B = os.environ.get("BUCKET", "gs://<POC 桶>")
DATA_GCS = os.environ.get("DATA_GCS", f"{B}/janus/issue10/data/fineweb-edu-10BT-000_00000.parquet")
MT_PKG = f"{B}/janus/pkg/maxtext-e2c18718.tgz"
MT_SHA = "24c9a379760842f6b11969389cbd7976a68c486d4847d8f86eaf549b6913dd2a"
if MODE != "proxy":
    MT_PKG = f"{B}/janus/pkg/maxtext-8ac1d175.tgz"
    MT_SHA = "c3ee3c1c1a319b4c0cd6e60ff92bcda7dfb6ba09590958ffcf53b3995fea5673"
IMAGE = "us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training@sha256:b152355a1425c28e99841c014703495059705de1733df8ac837ba43b4fe52fcf"
COMMON = ("src/maxtext/configs/base.yml model_name=hunyuan3-smoke override_model_config=True base_num_decoder_layers=16 "
          "base_emb_dim=1024 base_mlp_dim=4096 base_num_query_heads=16 base_num_kv_heads=4 base_moe_mlp_dim=256 mtp_num_layers=0 "
          "per_device_batch_size=1 max_target_length=4096 attention=flash use_tokamax_splash=True packing=True add_bos=False add_eos=False "
          "enable_checkpointing=False gcs_metrics=False log_step_diagnostics=True "
          "dataset_type=grain grain_file_type=parquet grain_train_files=/tmp/data/fwe.parquet grain_worker_count=1 "
          "tokenizer_type=huggingface tokenizer_path=/tmp/tok vocab_size=120832")
DATA = ("dataset_type=grain grain_file_type=parquet grain_train_files=/tmp/data/fwe.parquet grain_worker_count=1 "
        "tokenizer_type=huggingface tokenizer_path=/tmp/tok vocab_size=120832")
if MODE == "rw4":
    COMMON = (f"/tmp/hy3-4layers.yml mtp_num_layers=0 load_parameters_path={B}/janus/hunyuan3-4l/maxtext/hy3-4l-maxtext/0/items "
              "use_tokamax_splash=True packing=True add_bos=False add_eos=False enable_checkpointing=False gcs_metrics=False log_step_diagnostics=True " + DATA)
lines = []
for r in runs:
    if r.startswith("!"):
        lines.append(r[1:])
        continue
    name, extra, lib = (r.split("|") + ["", ""])[:3]
    lines.append(f"run {name} '{lib}' {extra}")
script = f"""set -e
ulimit -c 0
gcloud storage cp {MT_PKG} /tmp/mt.tgz
echo '{MT_SHA}  /tmp/mt.tgz' | sha256sum -c -
cd /deps && rm -rf src/maxtext && tar --no-same-owner -xzf /tmp/mt.tgz
gcloud storage cp {B}/janus/pkg/tokamax-0.0.16-py3-none-any.whl /tmp/ && pip install -q --no-deps /tmp/tokamax-0.0.16-py3-none-any.whl 2>/dev/null
mkdir -p /tmp/data /tmp/tok /tmp/out
gcloud storage cp {B}/janus/issue10/cfg/hy3-4layers.yml /tmp/hy3-4layers.yml
export MAXTEXT_CONFIGS_DIR=/deps/src/maxtext/configs
gcloud storage cp {DATA_GCS} /tmp/data/fwe.parquet
gcloud storage cp {B}/janus/hunyuan3-4l/hf/hy3-4l/tokenizer.json {B}/janus/hunyuan3-4l/hf/hy3-4l/tokenizer_config.json /tmp/tok/
pip list 2>/dev/null | grep -i -E "^(jax|jaxlib|libtpu|tokamax) " > /tmp/out/versions.txt || true
export PYTHONPATH=/deps/src:/deps
export XLA_FLAGS="--xla_allow_excess_precision=false"
COMMON="{COMMON}"
run() {{
  name=$1; lib=$2; shift 2
  export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=false $lib"
  echo "=== RUN $name $(date +%T) LIBTPU=$LIBTPU_INIT_ARGS ARGS=$*"
  set +e
  python3 -c "import sys,runpy,jax; jax.distributed.initialize(); sys.argv=['train']+sys.argv[1:]; runpy.run_module('maxtext.trainers.pre_train.train', run_name='__main__')" \\
    $COMMON "$@" base_output_directory={OUTDIR}/$name run_name=$name metrics_file=/tmp/out/$name.metrics.jsonl > /tmp/out/$name.log 2>&1
  rc=$?
  set -e
  echo "$*" > /tmp/out/$name.args; echo "$LIBTPU_INIT_ARGS" >> /tmp/out/$name.args
  echo "=== DONE $name rc=$rc $(date +%T)"
  grep -E "estimated memory|Error" /tmp/out/$name.log | tail -3 || true
}}
{chr(10).join(lines)}
echo "=== ALL DONE $(date +%T)"
for _ in $(seq 3600); do [ -e /tmp/collected ] && break; sleep 1; done
exit 0
"""
yaml = f"""apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {job}
  namespace: default
spec:
  failurePolicy: {{maxRestarts: 0}}
  ttlSecondsAfterFinished: 1800
  replicatedJobs:
  - name: w
    replicas: 1
    template:
      spec:
        backoffLimit: 0
        completions: {NHOST}
        parallelism: {NHOST}
        template:
          spec:
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-nodepool: {pool}
              cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
              cloud.google.com/gke-tpu-topology: {TOPO}
            tolerations:
            - {{key: google.com/tpu, operator: Exists, effect: NoSchedule}}
            - {{key: cloud.google.com/gke-spot, operator: Exists, effect: NoSchedule}}
            containers:
            - name: jax
              image: {IMAGE}
              resources:
                limits: {{google.com/tpu: 4}}
                requests: {{google.com/tpu: 4}}
              command: [bash, -c]
              args:
              - |
{textwrap.indent(script, ' ' * 16)}"""
print(yaml)
