"""把 TPU 上 optim_replay.py 的原始输出收成结果文件，另加「四要素」摘要（比什么、GPU 绝对值、TPU 绝对值、R），数字不重算。

摘要两块：
1. 每个臂逐步逐张量最差的一项；
2. 第一个臂（配置原样）里 R > 1.2 的每一项，在各臂的读数并列。
真实桶名换成 gs://<POC 桶>。

  python models/hunyuan4/analysis/optim_replay_summary.py <TPU 运行目录>/optim_replay.json <输出 json> [--note <说明>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LIMIT = 1.2
METRIC = "第 {step} 步 {name} 的主权重增量 Δ · rel_L2 对 FP64（Megatron-Core 公式）"


def four(r: dict, arm: str) -> dict:
    return {
        "metric": METRIC.format(step=r["step"], name=r["name"]),
        "gpu": r["rel_l2_gpu_vs_fp64"],
        "tpu": r[f"rel_l2_{arm}_vs_fp64"],
        "R": r[f"R_{arm}"],
        "arm": arm,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--note", default=None)
    a = p.parse_args(argv)
    d = json.loads(Path(a.src).read_text())
    arms = [x for x in d["arms"] if any(f"R_{x}" in r for r in d["rows"])]
    worst = {}
    for arm in arms:
        rs = [r for r in d["rows"] if r.get(f"R_{arm}") is not None]
        worst[arm] = four(max(rs, key=lambda r: r[f"R_{arm}"]), arm)
    over = []
    first = arms[0]
    for r in d["rows"]:
        if (r.get(f"R_{first}") or 0) > LIMIT:
            over.append(
                {"step": r["step"], "name": r["name"], "opt": r.get("opt"), "per_arm": [four(r, arm) for arm in arms]}
            )
    out = {
        "source_tpu_run": Path(a.src).parent.name,
        **({"note": a.note} if a.note else {}),
        "four_part": {
            "rule": f"R = rel_L2(Δ_TPU, Δ_FP64) ÷ rel_L2(Δ_GPU, Δ_FP64) ≤ {LIMIT} 过",
            "worst_per_arm": worst,
            f"over_{LIMIT}_in_{first}": over,
        },
        **d,
    }
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(out, ensure_ascii=False, indent=1))
    Path(a.out).write_text(text + "\n")
    print(json.dumps({k: {x: v[x] for x in ("gpu", "tpu", "R")} for k, v in worst.items()}, ensure_ascii=False))
    print(f"{len(over)} 项在 {first} 里超过 {LIMIT}")


if __name__ == "__main__":
    main()
