#!/usr/bin/env python3
"""hy4-mini 单开关步时 A/B：从运行目录读步时，写成一个结果文件。

两种量法，分开记、不混比：
- `tq-*`：300 步训练运行，`metrics.jsonl` 的 `step_time` 取第 11–300 步中位数（前 10 步含编译与预热）。
  `tq-base` 是全开（`t1zu-a` 那一代对齐配置，没有 M42b、M50），其余每个只关一个开关，`tq-alloff` 是对齐开关全关。
  代价 = 全开 ÷ 只关该开关 − 1（正数 = 开着更慢）。
- `tw-t42b-on` / `tw-t42b-off`（M42b）：诊断程序（`--diag-step1`）第 1 步编译后连跑 4 次，
  去掉第 1 次取中位数（`diag_step1.json` 的 `timing.run_seconds_median_excl_first`），另记实测显存峰值。

    python models/hunyuan4/analysis/tq_step_time.py --runs <运行根目录> --out <结果.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

TQ = [
    "tq-base", "tq-alloff", "tq-base2", "tq-m8", "tq-m9", "tq-m10", "tq-m12", "tq-m13", "tq-m23", "tq-m25",
    "tq-m26", "tq-m27", "tq-m28", "tq-m31", "tq-m33", "tq-m34", "tq-m36", "tq-m40", "tq-m41",
]  # fmt: skip
WINDOW = (11, 300)


def tq_row(root: Path, name: str) -> dict:
    rows = [json.loads(line) for line in open(root / name / "metrics.jsonl")]
    xs = [r["step_time"] for r in rows if WINDOW[0] <= r["step"] <= WINDOW[1]]
    meta = json.load(open(root / name / "meta.json"))
    return {
        "steps_in_window": len(xs),
        "median_s": statistics.median(xs),
        "code": meta.get("code"),
        "hardware": meta.get("hardware"),
    }


def diag_row(root: Path, name: str) -> dict:
    d = json.load(open(root / name / "diag_step1.json"))
    t = d["timing"]
    return {
        "run_seconds": t["run_seconds"],
        "median_excl_first_s": t["run_seconds_median_excl_first"],
        "peak_bytes_in_use_max": max(t["peak_bytes_in_use"].values()),
        "code": d.get("code"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="运行根目录（下有 tq-*、tw-t42b-* 运行目录）")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    root = Path(a.runs)
    tq = {n: tq_row(root, n) for n in TQ}
    base = tq["tq-base"]["median_s"]
    for n, r in tq.items():
        if n != "tq-base":
            r["cost_vs_base"] = base / r["median_s"] - 1
    on, off = diag_row(root, "tw-t42b-on"), diag_row(root, "tw-t42b-off")
    out = {
        "what": "hy4-mini 单开关步时 A/B（TPU v5p 8 个设备）",
        "tq": {
            "method": f"300 步训练运行，metrics.jsonl 的 step_time 取第 {WINDOW[0]}–{WINDOW[1]} 步中位数；"
            "cost_vs_base = tq-base ÷ 该运行 − 1（正数 = 开着更慢）",
            "base": "tq-base：t1zu-a 那一代对齐配置全开（没有 M42b、M50）；tq-mN 只关台账 MN 那个开关；tq-alloff 对齐开关全关",
            "runs": tq,
        },
        "m42b": {
            "method": "诊断程序第 1 步编译后连跑 4 次，去掉第 1 次取中位数；与 tq 的量法不同，不能并列比",
            "on": {"run": "tw-t42b-on", **on},
            "off": {"run": "tw-t42b-off", **off},
            "cost": on["median_excl_first_s"] / off["median_excl_first_s"] - 1,
            "peak_bytes_delta": on["peak_bytes_in_use_max"] - off["peak_bytes_in_use_max"],
        },
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
