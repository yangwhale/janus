"""第 1 步梯度的分组范数与主 loss：TPU 第 1 步诊断、GPU 各份第 1 步，对 TPU FP32 参照（同一程序 dtype float32、HIGHEST 的第 1 步诊断）。

    python models/hunyuan4/analysis/step1_groups_vs_fp32.py --tpu <TPU 诊断运行目录> --ref <FP32 诊断运行目录> \
        --gpu <GPU 运行目录> [--gpu …] --out <结果.json>

TPU 读 diag_step1.json 的 grad_sq（GPU 同名的分组平方和）与 grad_norm；GPU 读 metrics.jsonl 第 1 步的 grad_sq.<组> 与 grad_norm。
每组范数 = sqrt(平方和)，记 rel = (x − ref) / ref；R = |rel(TPU)| ÷ 中位数 |rel(GPU)|（R ≤ 1.2：TPU 不比 GPU 离参照远）。
只比标量，不比逐张量（逐张量要 GPU 导出的 HF 布局梯度）。参照本身对 FP64 的误差没有测。只作诊断。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def tpu_vals(run: str) -> dict:
    d = json.loads((Path(run) / "diag_step1.json").read_text())
    out = {f"norm.{k}": math.sqrt(v) for k, v in d["grad_sq"].items()}
    out["grad_norm"] = d["grad_norm"]
    return out


def gpu_vals(run: str) -> dict:
    d = json.loads((Path(run) / "metrics.jsonl").read_text().splitlines()[0])
    assert int(d["step"]) == 1
    out = {f"norm.{k[len('grad_sq.'):]}": math.sqrt(v) for k, v in d.items() if k.startswith("grad_sq.")}
    out["grad_norm"] = d["grad_norm"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpu", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--gpu", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    t, r = tpu_vals(a.tpu), tpu_vals(a.ref)
    g = {Path(x).name: gpu_vals(x) for x in a.gpu}
    rows = []
    for k in sorted(r):
        if k not in t or not all(k in v for v in g.values()) or r[k] == 0:
            continue
        grel = [(v[k] - r[k]) / r[k] for v in g.values()]
        trel = (t[k] - r[k]) / r[k]
        med = statistics.median(abs(x) for x in grel)
        rows.append({"quantity": k, "ref": r[k], "tpu_rel": trel, "gpu_rel_min": min(grel), "gpu_rel_max": max(grel),
                     "gpu_abs_rel_median": med, "R": abs(trel) / med if med else None})
    res = {"kind": "step1_groups_vs_fp32", "tpu": Path(a.tpu).name, "ref": Path(a.ref).name, "gpu": list(g), "rows": rows}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for row in rows:
        print(f"{row['quantity']:16s} tpu {row['tpu_rel']:+.2e}  gpu [{row['gpu_rel_min']:+.2e}, {row['gpu_rel_max']:+.2e}]  R {row['R']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
