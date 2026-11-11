"""汇总运行里记下的注意力最大 logit（QK-Clip 的统计量），离阈值多远。

- 训练运行：metrics.jsonl 每步的 `max_attention_logit`（全部层、头的最大），报全程最小 / 最大与第几步，
  以及 `qk.clipped_total` 的和（被裁的头数）。TPU 训练运行默认不记这一列（`qk_clip_log_per_head` 关），没有就记 null。
- 第 1 步诊断：diag_step1.json 的 `qk_max_logit`（逐层逐头），报每层与全部的最大、中位数。

    python models/hunyuan4/analysis/max_logit_summary.py --train gb2-a=<运行目录> … --diag tb6diag=<运行目录> --out <输出.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _train(run: Path) -> dict:
    lo = hi = None
    lo_step = hi_step = None
    clipped = 0.0
    steps = 0
    for line in (run / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        steps += 1
        clipped += float(r.get("qk.clipped_total") or 0.0)
        v = r.get("max_attention_logit")
        if v is None:
            continue
        if hi is None or v > hi:
            hi, hi_step = v, r["step"]
        if lo is None or v < lo:
            lo, lo_step = v, r["step"]
    return {
        "steps": steps,
        "max_attention_logit_min": lo,
        "min_at_step": lo_step,
        "max_attention_logit_max": hi,
        "max_at_step": hi_step,
        "clipped_heads_total": clipped,
        "meta_sha256": _sha(run / "meta.json"),
    }


def _diag(run: Path) -> dict:
    d = json.loads((run / "diag_step1.json").read_text())
    layers, allv = {}, []
    for k, v in sorted(d["qk_max_logit"].items()):
        layers[k] = {"heads": len(v), "max": max(v), "median": statistics.median(v)}
        allv += v
    return {
        "per_layer": layers,
        "max": max(allv),
        "median": statistics.median(allv),
        "meta_sha256": _sha(run / "meta.json"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="*", default=[], help="名字=运行目录")
    p.add_argument("--diag", nargs="*", default=[], help="名字=运行目录")
    p.add_argument("--threshold", type=float, default=100.0)
    p.add_argument("--note", default="")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    out = {
        "what": "注意力最大 logit（QK-Clip 统计量）：训练运行全程、第 1 步诊断逐层逐头",
        "threshold": a.threshold,
        "train": {s.split("=", 1)[0]: _train(Path(s.split("=", 1)[1])) for s in a.train},
        "diag": {s.split("=", 1)[0]: _diag(Path(s.split("=", 1)[1])) for s in a.diag},
        "note": a.note,
    }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
