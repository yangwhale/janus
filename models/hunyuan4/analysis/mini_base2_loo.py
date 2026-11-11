"""mini 1000 步 base2 开 / 关多份 TPU 的 r 分布，与 GPU 16 份留一法的 r 分布放在一起比。

做法：
- TPU：每一份都用 `01-mini/gpu-noise/catalogs/t1zza-l4-n16.yaml` 的检查（基线 `g3-a`，噪声 `g3-a` … `g3-p` 两两 120 对）重判；
- GPU 留一法：`g3-b` … `g3-p` 每份轮流当被测，基线仍是 `g3-a`，噪声用去掉这一份后剩下 15 份的两两 105 对；
- 每条检查的 r = 三个判定量（带符号均值、MARE、最大单步）各自 ÷ 噪声上限后取最大，同 `rw3_base2_replay.judged`；
  另记主 loss 的带符号均值（对 `g3-a`）。

    python mini_base2_loo.py --root <含 g3-* 与各 TPU 运行的根目录> \\
        --on t1zw-a ts-c-nolhs-1k ... --off t1zz-a tz-off-nolhs ... --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
from rw3_base2_replay import judged

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "alignments", "01-mini", "gpu-noise", "catalogs", "t1zza-l4-n16.yaml")
GPU = [f"g3-{c}" for c in "abcdefghijklmnop"]


def run_catalog(cat: dict, root: str, work: str, name: str) -> dict:
    path = os.path.join(work, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cat, f, allow_unicode=True, sort_keys=False)
    out = os.path.join(work, name)
    subprocess.run(
        [sys.executable, "-m", "janus", "run", path, "--root", root, "--out", out],
        stdout=subprocess.DEVNULL,
    )  # 退出码 1 / 2 = 有检查不过或报错，照样读 results.json
    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    j = judged(res)
    sm = next(c for c in res["checks"] if c["id"] == "L4.loss")["metrics"]
    return {"checks": j, "loss_signed_mean": sm.get("signed_mean_max", sm.get("signed_mean"))}


def lm(root: str, run: str) -> dict:
    out = {}
    with open(os.path.join(root, run, "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("lm_loss") is not None:
                out[r["step"]] = r["lm_loss"]
    return out


def vs_gpu_mean(t: dict, gs: list, lo: int = 2, hi: int = 1000) -> float:
    """主 loss 对若干份 GPU 逐步均值的相对差，在第 lo–hi 步上取平均（第 1 步单独记，同 janus run 的 first_step 1）。"""
    st = [s for s in range(lo, hi + 1) if s in t and all(s in g for g in gs)]
    return float(np.mean([(t[s] - np.mean([g[s] for g in gs])) / np.mean([g[s] for g in gs]) for s in st]))


def summary(rows: dict, cid: str) -> dict:
    v = [r["checks"][cid]["r"] for r in rows.values() if r["checks"][cid]["r"] is not None]
    return {"n": len(v), "min": min(v), "median": float(np.median(v)), "max": max(v), "values": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--on", nargs="+", required=True)
    ap.add_argument("--off", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    with open(TEMPLATE) as f:
        base = yaml.safe_load(f)
    work = tempfile.mkdtemp(prefix="mini-loo-")
    jobs = {}
    for n in a.on + a.off:
        jobs[n] = dict(base, stage=f"loo-{n}", runs=dict(base["runs"], test=n))
    for x in GPU[1:]:
        noise = [list(p) for p in itertools.combinations([g for g in GPU if g != x], 2)]
        # GPU 没有 TPU 的 param_norm_exact 列，被测也读 param_norm_fp64
        checks = [{**c, "select": {k: v for k, v in c["select"].items() if k != "test_field"}} for c in base["checks"]]
        jobs[x] = dict(base, stage=f"loo-{x}", runs=dict(base["runs"], test=x, noise=noise), checks=checks)
    with ThreadPoolExecutor(a.jobs) as ex:
        done = dict(zip(jobs, ex.map(lambda kv: run_catalog(kv[1], a.root, work, kv[0]), jobs.items())))
    tpu = {n: dict(done[n], arm=arm) for arm, names in (("on", a.on), ("off", a.off)) for n in names}
    gpu = {x: done[x] for x in GPU[1:]}
    for n, t in {**tpu, **gpu}.items():
        print(n, t.get("arm", "gpu"), {k: round(v["r"], 2) for k, v in t["checks"].items() if v["r"] is not None})

    cids = [c["id"] for c in base["checks"]]
    G = {g: lm(a.root, g) for g in GPU}
    mean16 = {n: vs_gpu_mean(lm(a.root, n), list(G.values())) for n in tpu}
    mean_loo = {x: vs_gpu_mean(G[x], [G[g] for g in GPU if g != x]) for x in GPU}
    out = {
        "what": "mini 1000 步：TPU base2 开 / 关各几份对 g3-a 的 r（16 份 120 对尺子），GPU g3-b … g3-p 留一法的 r（15 份 105 对）",
        "on": a.on,
        "off": a.off,
        "by_check": {
            cid: {
                "on": summary({n: tpu[n] for n in a.on}, cid),
                "off": summary({n: tpu[n] for n in a.off}, cid),
                "gpu_loo": summary(gpu, cid),
                "gpu_loo_rank_of_tpu": {
                    n: sum(g["checks"][cid]["r"] < tpu[n]["checks"][cid]["r"] for g in gpu.values()) for n in tpu
                },
            }
            for cid in cids
        },
        "loss_signed_mean": {
            "on": {n: tpu[n]["loss_signed_mean"] for n in a.on},
            "off": {n: tpu[n]["loss_signed_mean"] for n in a.off},
            "gpu_loo": {x: g["loss_signed_mean"] for x, g in gpu.items()},
        },
        "loss_vs_gpu_mean": {
            "what": "主 loss 第 2–1000 步对 GPU 逐步均值的相对差的平均：TPU 对 16 份，GPU 每份对其余 15 份",
            "on": {n: mean16[n] for n in a.on},
            "off": {n: mean16[n] for n in a.off},
            "gpu_loo": mean_loo,
            "gpu_loo_std": float(np.std(list(mean_loo.values()), ddof=1)),
        },
        "tpu": {n: {"arm": t["arm"], "checks": t["checks"]} for n, t in tpu.items()},
        "gpu": gpu,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(
        "vs gpu mean",
        {k: f"{v:+.2e}" for k, v in mean16.items()},
        "gpu loo std",
        f"{out['loss_vs_gpu_mean']['gpu_loo_std']:.2e}",
        "gpu range",
        f"{min(mean_loo.values()):+.2e}..{max(mean_loo.values()):+.2e}",
    )
    for cid, v in out["by_check"].items():
        print(
            cid,
            "on",
            [round(x, 2) for x in v["on"]["values"]],
            "off",
            [round(x, 2) for x in v["off"]["values"]],
            "gpu",
            f"{v['gpu_loo']['min']:.2f}/{v['gpu_loo']['median']:.2f}/{v['gpu_loo']['max']:.2f}",
        )


if __name__ == "__main__":
    main()
