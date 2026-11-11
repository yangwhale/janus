"""只改执行方式的 TPU 重复运行，按 300 步窗口算 r 的摆动：给开关表「升幅不到多少算波动内」重定这条线。

开关表（01-mini ALIGNMENT-REPORT「生产配置建议」）的读数是：只关一个开关跑 300 步（`tq-*`），每个量的每个判定量
（带符号均值、MARE、最大单步）各自 ÷ GPU 波动上限得 r，取比全开（`tq-base`）升得最多的那一项。这条线原来只来自一个对照
（QK-Clip 不含偏置项，0.48 → 0.93）。这里换成同一配置、只改执行方式的几份运行：

- 每份都套 `tq-base-l4.yaml` 的检查（基线 `g2-a`，噪声 `g2-a` … `g2-h` 两两 28 对，窗口第 1–300 步；
  `--ruler g3` 改用 `t1zza-l4-n16.yaml` 的 16 份 120 对，窗口截到第 1–300 / 101–300 步）；
- 同组里任取有序的两份 (i, j)，算「把 i 换成 j 时升得最多的那一项」D_ij = max_项 (r_j − r_i)，这就是开关表那个读数
  在「什么都没改、只换执行方式」时的分布；另记每一项在组内的极差。

    python exec_spread_300.py --root <含 g2-* / g3-* 与各 TPU 运行的根目录> \\
        --group on=t1zw-a,ts-c-nolhs-1k,tz-on-vm32b,tz-on-vm64 --group off=t1zz-a,tz-off-nolhs,tz-off-vm32,tz-off-vm64 \\
        [--ruler tq|g3] [--switch-results <01-mini/results 目录>] --out <json>
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
from mini_base2_loo import run_catalog

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "..", "alignments", "01-mini", "catalogs", "tq-base-l4.yaml")
CAT_G3 = os.path.join(HERE, "..", "alignments", "01-mini", "gpu-noise", "catalogs", "t1zza-l4-n16.yaml")
STATS = ("signed_mean", "mare", "max_step")


def template(ruler: str) -> dict:
    with open(CAT if ruler == "tq" else CAT_G3) as f:
        base = yaml.safe_load(f)
    if ruler == "g3":  # 窗口截到前 300 步，其余照抄
        base = copy.deepcopy(base)
        for c in base["checks"]:
            w = c["select"].get("window")
            if w:
                lo = int(w.split(":")[0])
                c["select"]["window"] = f"{lo}:301"
        base["checks"] = [
            c for c in base["checks"] if "window" not in c["select"] or int(c["select"]["window"].split(":")[0]) < 301
        ]
    return base


def items(checks: dict) -> dict:
    """{检查 id: judged 条目} → {'检查·判定量': r}，三个判定量各一项（开关表同样按项取升幅）。"""
    out = {}
    for cid, v in checks.items():
        for s in STATS:
            if s in v["r_each"]:
                out[f"{cid}·{s}"] = v["r_each"][s]
    return out


def max_rise(a: dict, b: dict) -> tuple[float, str]:
    ks = [k for k in a if k in b]
    k = max(ks, key=lambda k: b[k] - a[k])
    return b[k] - a[k], k


def switch_rises(res_dir: str) -> dict:
    """01-mini/results 里已提交的 tq-* 结果：每个只关一个开关的臂对 tq-base 升得最多的那一项。"""
    from rw3_base2_replay import judged

    def load(n):
        with open(os.path.join(res_dir, f"{n}-l4.results.json")) as f:
            return items(judged(json.load(f)))

    base = load("tq-base")
    out = {}
    for fn in sorted(os.listdir(res_dir)):
        m = re.match(r"(tq-(?!base-|alloff)[\w-]+)-l4\.results\.json$", fn)
        if not m:
            continue
        t = load(m.group(1))
        d, k = max_rise(base, t)
        out[m.group(1)] = {"rise": d, "item": k, "base": base[k], "off": t[k]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--group", action="append", required=True, help="名字=运行,运行,…（同一配置、只改执行方式）")
    ap.add_argument("--ruler", choices=("tq", "g3"), default="tq")
    ap.add_argument("--switch-results", help="01-mini/results：顺带列出 tq-* 各臂的升幅，对照新线")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in a.group}
    base = template(a.ruler)
    work = tempfile.mkdtemp(prefix="exec-spread-")
    names = [n for v in groups.values() for n in v]
    jobs = {n: dict(base, stage=f"spread-{n}", runs=dict(base["runs"], test=n)) for n in names}
    with ThreadPoolExecutor(a.jobs) as ex:
        done = dict(zip(jobs, ex.map(lambda kv: run_catalog(kv[1], a.root, work, kv[0]), jobs.items())))
    r = {n: items(done[n]["checks"]) for n in names}

    pairs, per_item = [], {}
    for g, ns in groups.items():
        for i, j in itertools.permutations(ns, 2):
            d, k = max_rise(r[i], r[j])
            pairs.append({"group": g, "from": i, "to": j, "max_rise": d, "item": k})
        for k in r[ns[0]]:
            v = [r[n][k] for n in ns if k in r[n]]
            per_item.setdefault(k, {})[g] = {"min": min(v), "max": max(v), "range": max(v) - min(v)}
    d = np.array([p["max_rise"] for p in pairs])
    out = {
        "what": __doc__.split("\n")[0],
        "ruler": "tq-base-l4.yaml（g2-a 基线，g2 8 份 28 对，第 1–300 步）"
        if a.ruler == "tq"
        else "t1zza-l4-n16.yaml（g3-a 基线，g3 16 份 120 对），窗口截到第 1–300 / 101–300 步",
        "groups": groups,
        "r": r,
        "max_rise_over_ordered_pairs": {
            "n": int(len(d)),
            "min": float(d.min()),
            "median": float(np.median(d)),
            "p90": float(np.quantile(d, 0.9)),
            "max": float(d.max()),
            "values": sorted(float(x) for x in d),
        },
        "pairs": pairs,
        "per_item_range": per_item,
    }
    if a.switch_results:
        out["switch_rises"] = switch_rises(a.switch_results)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    m = out["max_rise_over_ordered_pairs"]
    print(
        f"有序对 {m['n']}：升得最多一项 min {m['min']:.2f} 中位 {m['median']:.2f} p90 {m['p90']:.2f} max {m['max']:.2f}"
    )
    for k, v in per_item.items():
        print(k, {g: round(x["range"], 2) for g, x in v.items()})
    for k, v in (out.get("switch_rises") or {}).items():
        print(k, f"{v['rise']:.2f}", v["item"], f"{v['base']:.2f}→{v['off']:.2f}")


if __name__ == "__main__":
    main()
