"""汇总 06-rw3 第 1 步主 loss 的 BF16 舍入点拆分：每个「FP64 前向只加一类舍入」运行对 FP64 的相对差，及占 CPU BF16 前向的比例。

输入：FP64 参照的 summary.json（`--fp64`，逐行 loss 和在 micro 里）；`kd_bf16_vs_fp64.py` 的 rows.jsonl（`--bf16-rows`，
逐行 CPU BF16 前向的主 loss 和）；若干诊断运行目录（`--runs 名字=目录`，各含 step1_ref 的 summary.json：
`kpe_round_emul.py`、`mla_rope16_emul.py`、`round_sites_emul.py` 的输出）。比例按同一批行算：Σ(运行 − FP64) ÷ Σ(BF16 − FP64)。

用法：
    python main_loss_sites_summary.py --fp64 <summary.json> --bf16-rows <rows.jsonl> --runs kpe16=<目录> ... --out <json>
"""

from __future__ import annotations

import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp64", required=True)
    p.add_argument("--bf16-rows", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--gpu-lm-rel", type=float, default=None, help="GPU 第 1 步主 loss 对 FP64 的相对差（只记录）")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    F = json.load(open(a.fp64))
    f = {r["row"]: r for r in F["micro"]}
    b = {json.loads(x)["row"]: json.loads(x) for x in open(a.bf16_rows)}
    rows = sorted(b)
    base = sum(f[r]["loss_sum"] for r in rows)
    bf = sum(b[r]["lm_sum_bf16"] - f[r]["loss_sum"] for r in rows)
    out = {
        "what": "06-rw3 第 1 步主 loss：FP64 前向只加一类 BF16 舍入点，对 FP64 的相对差与占 CPU BF16 前向的比例",
        "rows": rows,
        "fp64_lm_loss": F["lm_loss"],
        "fp64_mtp_loss": F["mtp_loss"],
        "cpu_bf16_lm_rel": bf / base,
        "gpu_lm_rel": a.gpu_lm_rel,
        "runs": {},
    }
    for spec in a.runs:
        name, d = spec.split("=", 1)
        s = json.load(open(os.path.join(d, "summary.json")))
        m = {r["row"]: r for r in s["micro"]}
        assert sorted(m) == rows, (name, sorted(m))
        dl = sum(m[r]["loss_sum"] - f[r]["loss_sum"] for r in rows)
        out["runs"][name] = {
            "lm_rel": dl / base,
            "frac_of_cpu_bf16": dl / bf,
            "mtp_rel": s["mtp_loss"] / F["mtp_loss"] - 1,
            "lm_loss": s["lm_loss"],
            "mtp_loss": s["mtp_loss"],
        }
        print(name, json.dumps({k: f"{v:+.3e}" for k, v in out["runs"][name].items()}), flush=True)
    with open(a.out, "w") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
