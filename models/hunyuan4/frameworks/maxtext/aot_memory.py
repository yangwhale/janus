"""编译期显存分析（AOT）：在 CPU 上为 v5p 2x2x2（8 个设备）编译各阶段的训练步，记 memory_analysis。不占 TPU。

    ALLOW_MULTIPLE_LIBTPU_LOAD=1 JAX_PLATFORMS=cpu python aot_memory.py --stages A,B,C,D,D3 --out <结果目录>

要装 libtpu（只用它的编译器，不需要 TPU）。用 MaxText 自己的 train_compile（compile_topology=v5p-16 = 8 片 v5p，
compile_topology_num_slices=1），配置是 hy4-4layers.yml + 阶段覆盖（D3 = 3 层 + MTP、D1 = 1 层 + MTP，阶段 D 的候选）。
命令行可再追加覆盖，例如混元 3 的生产等价重算 remat_policy=full（moe_routing_maps: device 已在 yml 里）。

只作初筛：混元 3 的经验是 temp_size 和实测峰值能差十几 GiB，装不装得下最终看 TPU 上 meta.json 的 memory.peak_bytes_in_use。
编译器报 HLO 临时量超过可用 HBM 时（RESOURCE_EXHAUSTED），记下它报的需要量与可用量。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_common  # noqa: E402

YML = mt_common.HY4_YML  # 默认 hy4-4layers.yml；HY4_YML 可换（T1：hy4-mini-t1.yml）
STAGES = mt_common.STAGES
FIELDS = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
          "generated_code_size_in_bytes")


def run_stage(stage: str, logdir: Path, extra: list[str]) -> dict:
    layers, mtp = STAGES[stage]
    log = logdir / f"aot-{stage}.log"
    extra = mt_common.forced_overrides() + list(extra)
    cmd = [sys.executable, "-m", "maxtext.trainers.pre_train.train_compile", str(YML),
           f"base_num_decoder_layers={layers}", f"mtp_num_layers={mtp}", "compile_topology=v5p-16",
           "compile_topology_num_slices=1", f"run_name=aot-{stage}", "base_output_directory=/tmp/aotout", "steps=10", *extra]
    env = dict(os.environ, JAX_PLATFORMS="cpu", ALLOW_MULTIPLE_LIBTPU_LOAD="1", TPU_SKIP_MDS_QUERY="1")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    text = log.read_text(errors="replace")
    rec = {"stage": stage, "layers": layers, "mtp": mtp, "returncode": rc, "overrides": cmd[4:]}
    m = re.search(r"Memory analysis: CompiledMemoryStats\((.*?)\)", text)
    if m:
        kv = dict(re.findall(r"(\w+)=(\d+)", m.group(1)))
        rec["memory_analysis"] = {k: int(kv[k]) for k in FIELDS if k in kv}
        ma = rec["memory_analysis"]
        rec["argument_plus_temp_bytes"] = ma["argument_size_in_bytes"] + ma["temp_size_in_bytes"]
    m = re.search(r"total memory required for HLO temporaries \(([\d.]+)G\) exceeds available HBM \(([\d.]+)G\)", text)
    if m:
        rec["oom"] = {"temp_required_G": float(m.group(1)), "available_G": float(m.group(2)),
                      "message": m.group(0)}
    m = re.search(r"Num_devices: (\d+)", text)
    rec["num_devices"] = int(m.group(1)) if m else None
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="A,B,C,D,D3")
    p.add_argument("--out", required=True)
    p.add_argument("--maxtext-commit", default=os.environ.get("MAXTEXT_COMMIT", "unknown"))
    p.add_argument("extra", nargs="*", help="追加给 MaxText 的 key=value")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    yml_sha = hashlib.sha256(YML.read_bytes()).hexdigest()           # 开编之前取，编译期间改 yml 也不影响记录
    recs = [run_stage(s, out, a.extra) for s in a.stages.split(",")]
    for r in recs:
        print(json.dumps({k: r.get(k) for k in ("stage", "returncode", "argument_plus_temp_bytes", "oom")}, ensure_ascii=False))
    res = {"schema_version": 1, "kind": "aot_memory", "target": "v5p 2x2x2（compile_topology=v5p-16），8 个设备",
           "config": {"path": f"models/hunyuan4/frameworks/maxtext/{YML.name}",
                      "sha256": yml_sha},
           "maxtext_commit": a.maxtext_commit,
           "note": "编译期估算只作初筛；最终看 TPU 实测 peak_bytes_in_use",
           "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
           "stages": recs}
    (out / "aot_memory.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
