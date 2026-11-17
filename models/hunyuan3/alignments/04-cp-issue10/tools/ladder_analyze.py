#!/usr/bin/env python3
"""实践 04 阶梯实验：把每一级的运行汇总成两份结果文件和一张总表。

  (i)  第 1 步对 FP64：每级各臂对同级纯数据并行基线、同级 FSDP 参照（ladder_step1.py），列出远 1.2 倍以上的张量；
  (ii) 长跑 TPU 尺子：每级的参照 = 纯数据并行基线 + 4 份只改片上内存上限的重复（64 / 56 / 40 / 24 MiB），两两配对得噪声上限 N，
       各 CP 臂对同级基线的带符号均值 / MARE / 最大单步除以 N 得 r（../../analysis/cp_tpu_ruler.py 的口径）。
  离线编译与上一级同一程序的臂（hlo_audit.py 的指纹相同）不重跑，直接用上一级的运行（SAME 表）。

  python ladder_analyze.py --runs <运行根> --fp64 <fp64_step 输出> --out-dir results/ladder [--series-dir results/series/ladder]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = ["L0", "L0p", "L1", "L2", "L3", "L4"]
CP = ["ag2", "ag4", "rg2", "rg4", "ul2", "ul4", "usp4"]
REFS = ["v64", "v56", "v40", "v24"]
SAME = {("L1", "rg2"): "L0", ("L1", "rg4"): "L0", ("L1", "usp4"): "L0", ("L2", "fsdp8"): "L1"}   # 离线编译同一程序
WINDOWS = "2:3001,100:3001,2000:3001"


PREFIX = "lbd"         # 长跑那一组（学习率恒定 1e-4）；lad = 恒定 3e-4 那一组
STEP1_ONLY = "lad"     # 第 1 步与学习率无关：FSDP 参照只跑 2 步的，前三级用这一组的


def run_name(L, arm, R=None):
    n = f"{PREFIX}-{SAME.get((L, arm), L)}-{arm}"
    if R is not None and not (R / n / "DONE").exists():
        alt = f"{STEP1_ONLY}-{SAME.get((L, arm), L)}-{arm}"
        if (R / alt / "DONE").exists() and arm == "fsdp8":
            return alt
    return n


def series(run_dir: Path, out: Path):
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "lm_loss" in r]
    out.write_text(json.dumps({"first_step": int(rows[0]["step"]), "loss": [float(r["lm_loss"]) for r in rows],
                               "grad_norm": [float(r.get("grad_norm", "nan")) for r in rows]}) + "\n")


def main(argv=None) -> int:
    global PREFIX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--fp64", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--series-dir", default=None)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--prefix", default=PREFIX)
    a = ap.parse_args(argv)
    PREFIX = a.prefix
    R, out = Path(a.runs), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = Path(a.series_dir) if a.series_dir else None
    if sd:
        sd.mkdir(parents=True, exist_ok=True)
    have = lambda n: (R / n / "DONE").exists()  # noqa: E731
    for L in a.levels.split(","):
        # (i) 第 1 步
        args, pairs = [], []
        names = {arm: run_name(L, arm, R) for arm in ["dp8", "fsdp8", *CP]}
        for arm, n in names.items():
            if have(n):
                args += ["--run", f"{arm}={R / n}"]
        if have(names["dp8"]):
            for arm in ["fsdp8", *CP]:
                if have(names[arm]):
                    pairs += ["--pair", f"{arm}:dp8" + (":fsdp8" if arm != "fsdp8" and have(names["fsdp8"]) else "")]
            subprocess.run([sys.executable, str(HERE / "ladder_step1.py"), "--fp64", a.fp64, *args, *pairs,
                            "--out", str(out / f"step1-{PREFIX}-{L}.json")], check=True)
        # (ii) 长跑
        base = names["dp8"]
        refs = [base] + [f"{PREFIX}-{L}-dp8-{r}" for r in REFS]
        if all(have(n) and (R / n / "metrics.jsonl").exists() for n in refs):
            cmd = [sys.executable, str(HERE.parents[2] / "analysis" / "cp_tpu_ruler.py"), "--windows", WINDOWS]
            cmd += [x for n in refs for x in ("--ref", f"{n.split('-')[-1] if n != base else 'base'}={R / n / 'metrics.jsonl'}")]
            cmd += [x for arm in CP if have(names[arm]) for x in ("--test", f"{arm}={R / names[arm] / 'metrics.jsonl'}")]
            subprocess.run(cmd + ["--out", str(out / f"ruler-{PREFIX}-{L}.json")], check=True)
        if sd:
            for n in {*names.values(), *refs}:
                if have(n) and n.endswith(("-fsdp8",)) is False:
                    series(R / n, sd / f"{n}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
