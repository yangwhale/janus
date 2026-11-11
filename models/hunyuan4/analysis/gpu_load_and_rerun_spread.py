#!/usr/bin/env python3
"""GPU 一侧（hy4-mini，B200 × 8）：每份噪声参照运行的载入自检，与同设置重跑第 1 步的相对极差。

- 载入自检：训练入口第 1 步前核对每个参数等于检查点张量、FP64 平方和相等，不等就退出（台账 G7）；
  结果在每份运行 `meta.json` 的 `config.load_check`。
- 第 1 步相对极差：同一代各份运行第 1 步 `lm_loss`、`grad_norm` 的 (最大 − 最小) ÷ 均值。
  默认设置下 GPU 重跑不逐位，这个数说明「同设置重跑不逐位」的幅度。

    python models/hunyuan4/analysis/gpu_load_and_rerun_spread.py --runs <运行根目录> --out <结果.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GENERATIONS = {
    "g1": "abcdefgh",
    "g2": "abcdefgh",
    "g3": "abcdefgh",
    "g3x16": "abcdefghijklmnop",
}


def rel_range(xs):
    return (max(xs) - min(xs)) / statistics.mean(xs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="运行根目录（下有 g1-*、g2-*、g3-* 运行目录）")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    root = Path(a.runs)
    load, spread = {}, {}
    for gen, letters in GENERATIONS.items():
        prefix = "g3" if gen == "g3x16" else gen
        names = [f"{prefix}-{c}" for c in letters]
        lm, gn = [], []
        for n in names:
            first = json.loads(open(root / n / "metrics.jsonl").readline())
            assert first["step"] == 1, (n, first["step"])
            lm.append(first["lm_loss"])
            gn.append(first["grad_norm"])
            if n not in load:
                c = json.load(open(root / n / "meta.json"))["config"]["load_check"]
                load[n] = {
                    "pass": c["pass"],
                    "n_checked_rank0": c["n_checked_rank0"],
                    "n_mismatch_all_ranks": c["n_mismatch_all_ranks"],
                    "sumsq_equal_rank0": c["sumsq_rank0"]["model"] == c["sumsq_rank0"]["ckpt"],
                }
        spread[gen] = {
            "runs": names,
            "lm_loss_step1": lm,
            "grad_norm_step1": gn,
            "lm_loss_rel_range": rel_range(lm),
            "grad_norm_rel_range": rel_range(gn),
            "bitwise_equal": len(set(lm)) == 1 and len(set(gn)) == 1,
        }
    out = {
        "what": "hy4-mini GPU 噪声参照运行：载入自检与同设置重跑第 1 步的相对极差",
        "rel_range": "(最大 − 最小) ÷ 均值，同一代各份运行的第 1 步",
        "load_check": {"all_pass": all(v["pass"] and v["sumsq_equal_rank0"] for v in load.values()), "runs": load},
        "step1_spread": spread,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
