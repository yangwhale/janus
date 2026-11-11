"""前若干步逐步比：TPU 与 GPU 各份谁离 TPU FP32 轨迹（dtype float32、HIGHEST、MoE 稠密实现，回放同一批数据）更近。

用法：
    python models/hunyuan4/analysis/early_vs_fp32.py --ref <FP32 运行目录> --tpu <TPU 运行目录> [--tpu …] \
        --gpu <GPU 运行目录> [--gpu …] --fields lm_loss,grad_norm,mtp_loss_mt --out <结果.json>

每个运行目录只读 metrics.jsonl（每行一个 step）。GPU 的 mtp_loss_mt 与 TPU 同名（MaxText 口径）。
每一步每个量记 rel = (x − ref) / ref：TPU 每份一个值，GPU 8 份记均值、最小、最大，以及 |rel| 的中位数；
另记「这一步 TPU 的 |rel| 是否小于 GPU 每一份的 |rel|」。
--gap-windows（如 1:1,2:10,11:100,101:300）另在 TPU 与 GPU 的全部步上按窗口算 TPU 对 GPU 均值的带符号相对差
(t − mean(g)) / mean(g) 的均值、绝对值均值，与 GPU 份间极差 (max − min) / mean 的最大值（不需要 FP32 轨迹）。只作诊断，不进门槛（PROTOCOL.md 变更记录）。
同 T1 的 t1-early-grad-norm-vs-fp32.json：FP32 轨迹本身也会慢慢离开 BF16 轨迹，越往后越只说明「两边都离开了」。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(run: str) -> dict[int, dict]:
    out = {}
    for line in (Path(run) / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[int(d["step"])] = d
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True)
    p.add_argument("--tpu", action="append", required=True)
    p.add_argument("--gpu", action="append", required=True)
    p.add_argument("--fields", default="lm_loss,grad_norm")
    p.add_argument("--gap-windows", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    ref = load(a.ref)
    tpu = {Path(r).name: load(r) for r in a.tpu}
    gpu = {Path(r).name: load(r) for r in a.gpu}
    fields = a.fields.split(",")
    steps = sorted(s for s in ref if all(s in t for t in tpu.values()) and all(s in g for g in gpu.values()))
    rows, summary = [], {}
    for f in fields:
        closer = {n: 0 for n in tpu}
        for s in steps:
            r = ref[s].get(f)
            if r is None or r == 0:
                continue
            grel = [(g[s][f] - r) / r for g in gpu.values() if g[s].get(f) is not None]
            row = {"step": s, "field": f, "ref": r,
                   "gpu_mean_rel": statistics.fmean(grel), "gpu_min_rel": min(grel), "gpu_max_rel": max(grel),
                   "gpu_median_abs_rel": statistics.median(abs(x) for x in grel)}
            for n, t in tpu.items():
                x = (t[s][f] - r) / r
                row[f"tpu_rel.{n}"] = x
                c = abs(x) < min(abs(v) for v in grel)
                row[f"tpu_closer_than_every_gpu.{n}"] = c
                closer[n] += int(c)
            rows.append(row)
        n_steps = sum(1 for row in rows if row["field"] == f)
        summary[f] = {"steps": n_steps, "tpu_closer_than_every_gpu": closer}
    gaps = {}
    for w in filter(None, a.gap_windows.split(",")):
        lo, hi = (int(x) for x in w.split(":"))
        for f in fields:
            for n, t in tpu.items():
                d, spread = [], []
                for s in range(lo, hi + 1):
                    gv = [g[s][f] for g in gpu.values() if s in g and g[s].get(f) is not None]
                    if s not in t or t[s].get(f) is None or len(gv) != len(gpu):
                        continue
                    m = statistics.fmean(gv)
                    d.append((t[s][f] - m) / m)
                    spread.append((max(gv) - min(gv)) / m)
                if d:
                    gaps.setdefault(f, {}).setdefault(n, {})[w] = {
                        "steps": len(d), "signed_mean": statistics.fmean(d),
                        "abs_mean": statistics.fmean(abs(x) for x in d), "gpu_spread_max": max(spread)}
    res = {"kind": "early_vs_fp32", "ref": Path(a.ref).name, "tpu": list(tpu), "gpu": list(gpu), "fields": fields,
           "summary": summary, "tpu_minus_gpu_mean": gaps, "rows": rows}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
