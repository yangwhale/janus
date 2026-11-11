"""GPU 重跑的留一法 r 分布：给 TPU 自比的「略超」做事后解读用（不改判法）。

做法（只用已有的 GPU 运行，不跑新作业）：同设置 8 份 GPU 运行，轮流拿一份 x 当被测、另一份 g 当 golden（8 × 7 = 56 种），
噪声参照用 x 之外 7 份的两两 21 对（含 g，与 TPU 自比里「噪声含 golden、不含被测」的处境一样）；检查与 TPU 自比的目录相同
（同一个目录文件，只换被测、golden、噪声）。每种组合记下每条检查三个判定量的 r 与最差 r，统计落到各档的比例。

    python models/hunyuan4/analysis/gpu_loo_r.py --root <运行目录根> --catalog <TPU 自比的目录> --prefix g2 --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as stt
import subprocess
import sys
import tempfile

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KEYS = ("signed_mean", "mare", "max_step")


def run_one(cat_path, root, work):
    subprocess.run([sys.executable, "-m", "janus", "run", cat_path, "--root", root, "--out", work],
                   cwd=REPO, capture_output=True, text=True)
    res = json.load(open(os.path.join(work, "results.json")))
    out = {}
    for c in res["checks"]:
        up, m = c["threshold"]["noise"]["upper"], c["metrics"]
        out[c["id"]] = {k: abs(m.get(k, m.get(k + "_max"))) / up[k] if up[k] else 0.0 for k in KEYS}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--prefix", default="g2")
    p.add_argument("--letters", default="abcdefgh", help="GPU 运行名的后缀字母（默认 8 份 a…h；扩到 16 份时给 abcdefghijklmnop）")
    p.add_argument("--jobs", type=int, default=1, help="并行跑几个目录（每个是一次 janus run）")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    base = yaml.safe_load(open(a.catalog))
    runs = [f"{a.prefix}-{c}" for c in a.letters]
    combos = []
    with tempfile.TemporaryDirectory() as tmp:

        def one(pair):
            x, g = pair
            rest = [r for r in runs if r != x]
            cat = dict(base)
            cat["runs"] = {"test": x, "golden": [g], "noise": [list(pr) for pr in itertools.combinations(rest, 2)],
                           "root": base["runs"].get("root")}
            cat["stage"] = f"gpu-loo-{x}-vs-{g}"
            cp = os.path.join(tmp, f"{x}-{g}.yaml")
            yaml.safe_dump(cat, open(cp, "w"), allow_unicode=True, sort_keys=False)
            r = run_one(cp, a.root, os.path.join(tmp, f"{x}-{g}"))
            worst = max((v, cid, k) for cid, d in r.items() for k, v in d.items())
            return {"test": x, "golden": g, "r": r, "r_max": worst[0], "worst": f"{worst[1]}·{worst[2]}"}

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            combos = list(ex.map(one, itertools.permutations(runs, 2)))
    rm = sorted(c["r_max"] for c in combos)
    n = len(rm)
    q = lambda f: rm[min(n - 1, int(f * n))]  # noqa: E731
    out = {"what": f"GPU 同设置 {len(runs)} 份的留一法 r：{n} 种（被测, golden）组合，噪声用被测之外 {len(runs) - 1} 份两两组合；检查同 TPU 自比的目录",
           "catalog": os.path.relpath(a.catalog, REPO), "prefix": a.prefix, "combos": n,
           "frac_r_max_gt_1": sum(x > 1 for x in rm) / n, "frac_r_max_gt_2": sum(x > 2 for x in rm) / n,
           "r_max_median": stt.median(rm), "r_max_q90": q(0.9), "r_max_max": rm[-1],
           "per_check_frac_gt_1": {cid: sum(max(c["r"][cid].values()) > 1 for c in combos) / n for cid in combos[0]["r"]},
           "worst_counts": {w: sum(c["worst"] == w for c in combos) for w in sorted({c["worst"] for c in combos})},
           "combos_detail": combos}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("combos", "frac_r_max_gt_1", "frac_r_max_gt_2", "r_max_median", "r_max_q90", "r_max_max")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
