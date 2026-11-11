"""L2 逐层回放（layer_replay.py --save）几次运行之间的逐位核对与四要素：只差一个开关的两次运行，每层的输出与输入梯度
是否逐位相同（相同元素的比例与相对差），以及各次运行的层输出 / 输入梯度 TPU 对 GPU 的 rel_L2。不报对 TPU FP32 参照的 R
（被测一侧产出的参照，项目负责人 2026-11-08 15:03 的规矩；R 等 CPU FP64 参照）。数字只从各运行的 layer_replay.json /
.safetensors 取。

  python models/hunyuan4/analysis/l2_switch_compare.py --runs <上一级目录> --label g=全开 --label h=… \\
      --pair h:i=M41（M40 关） --pair g:j=M41（M40 开） --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--label", action="append", default=[], help="运行名=说明")
    p.add_argument("--pair", action="append", default=[], help="开:关=说明（两个运行名）")
    p.add_argument("--prefix", default="tw-l2-")
    p.add_argument("--note", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from safetensors.numpy import load_file

    root = Path(a.runs)
    labels = dict(x.split("=", 1) for x in a.label)
    runs = {}
    for k in labels:
        d = root / f"{a.prefix}{k}"
        runs[k] = {
            "json": json.loads((d / "layer_replay.json").read_text()),
            "arr": load_file(str(d / "layer_replay.safetensors")),
        }
    per_run = {}
    for k, r in runs.items():
        rows = []
        for row in r["json"]["rows"]:
            for q in ("out", "in_grad"):
                if f"{q}.rel_l2_tpu_vs_gpu" in row:
                    rows.append({"layer": row["layer"], "quantity": q, "tpu_vs_gpu": row[f"{q}.rel_l2_tpu_vs_gpu"]})
        per_run[k] = {"label": labels[k], "overrides": r["json"].get("maxtext_overrides"), "rows": rows}
    pairs = []
    for spec in a.pair:
        ab, _, what = spec.partition("=")
        on, off = ab.split(":")
        A, B = runs[on]["arr"], runs[off]["arr"]
        rows = []
        for key in sorted(k for k in A if k.startswith("tpu.") and k in B):
            x, y = A[key].astype(np.float64), B[key].astype(np.float64)
            rows.append(
                {
                    "tensor": key[4:],
                    "bit_identical": bool(np.array_equal(A[key].view(np.uint32), B[key].view(np.uint32))),
                    "frac_equal": float(np.mean(x == y)),
                    "rel_on_vs_off": float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300)),
                }
            )
        pairs.append({"on": on, "off": off, "what": what, "rows": rows})
    out = {
        **({"note": a.note} if a.note else {}),
        "what": "L2 逐层回放：只差一个开关的两次运行逐位核对，与各次运行 TPU 对 GPU 的 rel_L2",
        "metric": "层输出 / 层输入梯度 · TPU 对 GPU 的 rel_L2；R 等 CPU FP64 参照",
        "runs": per_run,
        "pairs": pairs,
    }
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(out, ensure_ascii=False, indent=1, default=float))
    Path(a.out).write_text(text + "\n")
    for pr in pairs:
        print(pr["what"], [(r["tensor"], r["bit_identical"], round(r["frac_equal"], 4)) for r in pr["rows"]])


if __name__ == "__main__":
    main()
