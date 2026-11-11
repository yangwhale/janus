"""几条训练轨迹在「logits 整体水平」相关量上的位置（案例 01）：都对一组参照运行（同设置 GPU 运行）的逐步均值比。

量：残差流探针的 lse 均值（probe_lse_mean，固定序列，--resid-probe-every）、param_norm、输出层平方和（group_sq.output，
--group-norms）、各段 loss 的带符号相对均值。参照运行之间的散布当噪声一起报。用来回答「谁离精确轨迹近」：
把全程 FP32 的运行（GPU `--fp32`、TPU 参照精度覆盖）和两侧 BF16 运行、诊断臂放在同一把尺子上。

  python level_traj.py --ref g-probe-a=<metrics.jsonl> --ref g-probe-b=… --run t-probe-1000=<metrics.jsonl> --run g-f32-1k=… \\
      [--steps 25,50,100,200,300,500,700,1000] --out <json>

metrics.jsonl 里最后一行可能是截断的（运行目录取回时还在写），读不了的行跳过。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("lm_loss", "probe_lse_mean", "param_norm", "group_sq.output")
WINDOWS = ((2, 100), (101, 300), (301, 1000))


def load(path) -> dict[int, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out[int(r["step"])] = {k: r[k] for k in KEYS if k in r}
    return out


def ref_mean(refs: dict, step: int, key: str):
    v = [d[step][key] for d in refs.values() if step in d and key in d[step]]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def summarize(name: str, d: dict, refs: dict, steps) -> dict:
    res = {"run": name, "at": {}, "loss_windows": {}}
    for s in steps:
        row = {}
        for k in ("probe_lse_mean", "param_norm", "group_sq.output"):
            m, n = ref_mean(refs, s, k)
            if m is not None and s in d and k in d[s]:
                row[k] = d[s][k] - m
        if row:
            res["at"][str(s)] = row
    for a, b in WINDOWS:
        rel = []
        for s in range(a, b + 1):
            m, _ = ref_mean(refs, s, "lm_loss")
            if m is not None and s in d and "lm_loss" in d[s]:
                rel.append((d[s]["lm_loss"] - m) / abs(m))
        if rel:
            res["loss_windows"][f"{a}-{b}"] = {"mean_rel": float(np.mean(rel)), "n": len(rel)}
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", action="append", required=True, help="名字=metrics.jsonl（参照，同设置的 GPU 运行）")
    p.add_argument("--run", action="append", default=[], help="名字=metrics.jsonl（被比的运行）")
    p.add_argument("--steps", default="25,50,100,200,300,500,700,975,1000")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    refs = {x.split("=", 1)[0]: load(x.split("=", 1)[1]) for x in a.ref}
    runs = {x.split("=", 1)[0]: load(x.split("=", 1)[1]) for x in a.run}
    steps = [int(x) for x in a.steps.split(",") if x]
    out = {"refs": sorted(refs), "steps": steps, "runs": {}, "ref_spread": {}}
    for name, d in {**runs, **{f"[参照] {k}": v for k, v in refs.items()}}.items():
        out["runs" if not name.startswith("[参照]") else "ref_spread"][name] = summarize(name, d, refs, steps)
    for key, fmt in (("probe_lse_mean", "{:+9.4f}"), ("param_norm", "{:+10.2e}"), ("group_sq.output", "{:+9.2f}")):
        print(f"{key}（减去参照逐步均值）")
        print(f"{'运行':<24}" + "".join(f"{s:>10d}" for s in steps))
        for section in ("runs", "ref_spread"):
            for name, r in out[section].items():
                cells = [fmt.format(r["at"][str(s)][key]).rjust(10) if str(s) in r["at"] and key in r["at"][str(s)]
                         else f"{'-':>10}" for s in steps]
                print(f"{name[:24]:<24}" + "".join(cells))
    print("loss 分段带符号相对均值（减去参照逐步均值）")
    for section in ("runs", "ref_spread"):
        for name, r in out[section].items():
            print(f"{name[:24]:<24}" + "  ".join(f"{w} {v['mean_rel']:+.2e}" for w, v in r["loss_windows"].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
