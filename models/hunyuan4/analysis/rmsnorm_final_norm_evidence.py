"""RMSNorm 多舍一次在 libtpu 0.0.52 上变强的证据汇总：程序里最后一层 norm 的归一化值有没有被单独写回内存、
同一份权重上的前向差、训练曲线上的差。

1. 程序（--hlo 名字=运行目录 …，读运行目录里 hlo/ 下的优化后 HLO）：找「归一化值被单独舍成 BF16 写回内存」的融合——
   融合的 ROOT 是转成 BF16 的 convert，来自默认写法里「归一化值舍成 dtype」那一行（栈帧的行号按运行的 MaxText 提交
   现查）、前向（op_name 带 jvp、不带 transpose）。按「在不在层的 scan 里」分开数（op_name 里
   while/body 出现 1 次 = 层外，即最后一层 norm 与 MTP；2 次 = 层内）。默认写法下这类融合出现，就说明 MaxText 写的
   第一次舍入真的发生了；不出现时，那次 BF16 转换留在融合内部，被 excess precision 省掉。
2. 同一份权重上的前向（--frozen 名字=被测运行,对照运行 …）：学习率 0 的冻结运行（train_hy4.py --param-override
   同一份 HF 快照 … learning_rate=0 routed_bias_update_rate=0），逐批 lm_loss 的相对差（被测 − 对照）÷ 对照。
3. 训练曲线（--curve 名字=被测运行,对照运行 …）：lm_loss 相对差按步段取平均。

    python rmsnorm_final_norm_evidence.py --root ~/janus-runs --maxtext-repo <maxtext 克隆> \\
        --hlo r46=ts-e-rms2r r52=ts52-e-rms2r ... --frozen r52=fz52-r,fz52-g ... --curve v7=ts52-e-rms2r,ts-g52-a ... \\
        --out <json>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

import numpy as np

WINDOWS = ((1, 10), (11, 50), (51, 100), (101, 200), (201, 300))


def _table(lines, name):
    out, i = {}, lines.index(name) + 1
    while i < len(lines) and lines[i] and lines[i][0].isdigit():
        k, _, v = lines[i].partition(" ")
        out[int(k)] = v
        i += 1
    return out


def innermost_frame(lines):
    """stack_frame_id → 最内层栈帧的（源文件名, 行号）。"""
    files = {k: v.strip('"').split("/")[-1] for k, v in _table(lines, "FileNames").items()}
    locs = {
        k: (int(re.search(r"file_name_id=(\d+)", v).group(1)), int(re.search(r"line=(\d+)", v).group(1)))
        for k, v in _table(lines, "FileLocations").items()
    }
    frames = {k: int(re.search(r"file_location_id=(\d+)", v).group(1)) for k, v in _table(lines, "StackFrames").items()}
    return {k: (files[locs[v][0]], locs[v][1]) for k, v in frames.items()}


Y_LINE = "y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)"


def y_line(maxtext_repo: Path, commit: str) -> int:
    """默认写法里「归一化值舍成 dtype」那一行在这个 MaxText 提交的 normalizations.py 里的行号。"""
    src = subprocess.run(
        ["git", "-C", str(maxtext_repo), "show", f"{commit}:src/maxtext/layers/normalizations.py"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    hits = [i + 1 for i, ln in enumerate(src) if ln.strip() == Y_LINE]
    assert len(hits) == 1, hits
    return hits[0]


def materialized_normalized(hlo_text: str, line: int) -> dict:
    lines = hlo_text.splitlines()
    fid = innermost_frame(lines)
    comps, cur = collections.defaultdict(list), None
    for ln in lines:
        m = re.match(r"^(?:ENTRY )?%([\w.\-]+) .*\{$", ln)
        if m:
            cur = m.group(1)
            continue
        if ln.startswith("}"):
            cur = None
            continue
        if cur:
            comps[cur].append(ln)
    out = collections.Counter()
    for body in comps.values():
        roots = [ln for ln in body if ln.lstrip().startswith("ROOT ")]
        if len(roots) != 1:
            continue
        r = roots[0]
        m = re.search(r"= bf16\[[\d,]+\]\S* convert\(.*op_name=\"([^\"]*)\".*stack_frame_id=(\d+)", r)
        if not m or fid.get(int(m.group(2))) != ("normalizations.py", line):
            continue
        op = m.group(1)
        if "transpose" in op or "jvp" not in op:
            continue
        out["outer" if op.count("while/body") <= 1 else "in_layers"] += 1
    return dict(out)


def lm(run_dir: Path) -> dict:
    return {
        x["step"]: x["lm_loss"]
        for x in map(json.loads, open(run_dir / "metrics.jsonl"))
        if "lm_loss" in x
    }


def rel(a: dict, b: dict, steps) -> np.ndarray:
    return np.array([(a[s] - b[s]) / b[s] for s in steps])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--maxtext-repo", required=True, type=Path, help="maxtext 克隆（按运行的提交找那一行的行号）")
    ap.add_argument("--hlo", nargs="*", default=[])
    ap.add_argument("--frozen", nargs="*", default=[])
    ap.add_argument("--curve", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"hlo": {}, "frozen": {}, "curve": {}}
    for spec in a.hlo:
        name, run = spec.split("=")
        fs = sorted((a.root / run / "hlo").glob("*after_optimizations.txt"))
        commit = json.load(open(a.root / run / "meta.json"))["config"]["args"]["maxtext_commit"]
        res["hlo"][name] = {
            "run": run,
            "maxtext": commit,
            "materialized_normalized_value": materialized_normalized(fs[0].read_text(), y_line(a.maxtext_repo, commit)),
        }
        print(name, res["hlo"][name], flush=True)
    for spec in a.frozen:
        name, runs = spec.split("=")
        t, b = runs.split(",")
        A, B = lm(a.root / t), lm(a.root / b)
        steps = sorted(set(A) & set(B))
        v = rel(A, B, steps)
        res["frozen"][name] = {
            "test": t,
            "base": b,
            "steps": [steps[0], steps[-1]],
            "rel_mean": float(v.mean()),
            "rel_se": float(v.std(ddof=1) / len(v) ** 0.5),
            "rel_min": float(v.min()),
            "rel_max": float(v.max()),
            "rel_by_step": {str(st): float(x) for st, x in zip(steps, v)},
        }
        print(name, res["frozen"][name], flush=True)
    for spec in a.curve:
        name, runs = spec.split("=")
        t, b = runs.split(",")
        A, B = lm(a.root / t), lm(a.root / b)
        res["curve"][name] = {
            "test": t,
            "base": b,
            "rel_mean_by_window": {f"{s0}-{s1}": float(rel(A, B, range(s0, s1 + 1)).mean()) for s0, s1 in WINDOWS},
        }
        print(name, res["curve"][name], flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
