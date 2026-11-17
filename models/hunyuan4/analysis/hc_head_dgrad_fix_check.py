"""hc_head_dgrad_force_bf16_rounding（MaxText 开关，台账 M65）的上卡核对：开关开 / 关的运行，第 1 步训练梯度对 CPU FP64，
按 TPU 自比第 1 步的判法逐张量判，另报 hc_head 三个张量、逐步量是否逐位相同。

每一对（被测, 对照）：
  - 第 1 步训练梯度从全量抓数的动量反推（tpu_self_compare.train_grad）；d = rel(被测, 对照)；
  - 尺子与 GPU 离 FP64 的中位数是只跟 GPU 有关的量，取自一份已入库的自比结果文件（--gpu-from，默认 golden 那份 3 份 GPU 的数），
    逐张量与 tpu_self_compare.py 同一判法（含 2026-11-12 起的 iHC 豁免上限）：scale / attributed / ihc_accepted / unattributed；
  - 另报 hc_head 三个张量离 FP64、hc_head_base 四个元素，逐步量（步时除外）是否逐位相同、第一处不同的步，步时中位数（第 11 步起）；
  - 给了 --catalog 时另比前 300 步曲线：拿这份 janus 目录（尺子是 GPU 8 份 28 对），把 golden 换成这一对的对照，三个判定量 ÷ 尺子的最差 r。

    python models/hunyuan4/analysis/hc_head_dgrad_fix_check.py --root <运行目录根> --fp64 <FP64 参照目录> \
        --pair <被测>:<对照> [--pair …] [--catalog models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml] --out <json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import tpu_self_compare as tsc  # noqa: E402

GPU_EXPERTS = 4  # GPU 第 1 步抓数只有 rank 0 本地 0–3 号专家（g2s-*.s1.grad.hf.json 的 experts）
RESULTS = Path(__file__).resolve().parents[1] / "alignments/04-tpu-self/results"


def verdict(row):
    if row["over_scale"] <= 1:
        return "scale"
    if row["r64"] <= tsc.R64_LIMIT:
        return "attributed"
    if tsc.IHC.search(row["tensor"]) and row["r64"] <= tsc.IHC_R64_CAP:
        return "ihc_accepted"
    return "unattributed"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--fp64", required=True)
    p.add_argument("--pair", action="append", required=True, help="被测:对照（运行名）")
    p.add_argument(
        "--gpu-from",
        default=str(RESULTS / "ts-g-b-vs-golden.json"),
        help="取逐张量 GPU 尺子与 GPU 离 FP64 中位数的结果文件",
    )
    p.add_argument("--catalog", default=None, help="曲线检查的 janus 目录（golden 换成每一对的对照）；不给就不比曲线")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from safetensors import safe_open

    gpu = {r["tensor"]: r for r in json.loads(Path(a.gpu_from).read_text())["step1_rows"]}
    F = safe_open(os.path.join(a.fp64, "grads_hf.safetensors"), "pt")
    out = {
        "what": __doc__.splitlines()[0],
        "gpu_from": os.path.relpath(a.gpu_from, RESULTS.parents[3]),
        "rule": tsc.IHC_RULE,
        "pairs": {},
    }
    cache = {}

    def grads(run):
        if run not in cache:
            d = os.path.join(a.root, run)
            m = tsc.metrics(d)
            cache[run] = (tsc.Dump(d), m[1]["raw_grad_norm"], m)
        return cache[run]

    for pr in a.pair:
        test, ref = pr.split(":")
        T, gnt, mt = grads(test)
        G, gng, mg = grads(ref)
        bm = tsc.bitwise_metrics(mt, mg)
        rows, head = [], {}
        for n in sorted(gpu):
            gt, gg = tsc.train_grad(T, n, gnt), tsc.train_grad(G, n, gng)
            g = gpu[n]
            # 路由专家：同 tpu_self_compare.step1_grads，尺子只在 GPU 抓数有的专家（0–3 号）上量，d 与离 FP64 也只比这几个
            ex = slice(0, GPU_EXPERTS) if ".mlp.experts." in n and gg.ndim == 3 else None
            r = F.get_tensor(n).double().numpy()
            if ex is not None and r.shape[0] != gg.shape[0]:
                r = r[:GPU_EXPERTS] if r.shape[0] > GPU_EXPERTS else r
            elif ex is not None:
                r = r[ex]
            ts_, gs_ = (gt[ex], gg[ex]) if ex is not None else (gt, gg)
            rn = float(np.linalg.norm(r))
            row = {
                "tensor": n,
                "d": tsc.rel(ts_, gs_, float(np.linalg.norm(gs_))),
                "scale": g["scale"],
                "test_vs_fp64": tsc.rel(ts_, r, rn),
                "ref_vs_fp64": tsc.rel(gs_, r, rn),
                "gpu_vs_fp64": g["gpu_vs_fp64"],
            }
            row["over_scale"] = row["d"] / max(row["scale"], tsc.SCALE_FLOOR)
            den = max(row["ref_vs_fp64"], row["gpu_vs_fp64"])
            row["r64"] = row["test_vs_fp64"] / den if den else (1.0 if row["test_vs_fp64"] == 0 else math.inf)
            row["verdict"] = verdict(row)
            rows.append(row)
            if n.startswith("model.hc_head."):
                head[n.split(".")[-1]] = {
                    "test_vs_fp64": row["test_vs_fp64"],
                    "ref_vs_fp64": row["ref_vs_fp64"],
                    "gpu_vs_fp64": row["gpu_vs_fp64"],
                    "r64": row["r64"],
                    "verdict": row["verdict"],
                }
                if n.endswith("hc_head_base"):
                    head[n.split(".")[-1]]["elements_test"] = [float(x) for x in np.ravel(gt)[:4]]
                    head[n.split(".")[-1]]["elements_ref"] = [float(x) for x in np.ravel(gg)[:4]]
                    head[n.split(".")[-1]]["elements_fp64"] = [float(x) for x in np.ravel(r)[:4]]
        cnt = {}
        for r_ in rows:
            cnt[r_["verdict"]] = cnt.get(r_["verdict"], 0) + 1
        un = sorted((r_ for r_ in rows if r_["verdict"] == "unattributed"), key=lambda r_: -r_["r64"])
        out["pairs"][pr] = {
            "metrics_bitwise": bm["equal"],
            "first_differ_step": bm["first_differ_step"],
            "n_tensors": len(rows),
            "n_bitwise": sum(r_["d"] == 0 for r_ in rows),
            "counts": cnt,
            "unattributed": [
                {k: r_[k] for k in ("tensor", "r64", "over_scale", "test_vs_fp64", "ref_vs_fp64")} for r_ in un[:10]
            ],
            "test_vs_fp64_over_ref_median": float(
                np.median([r_["test_vs_fp64"] / r_["ref_vs_fp64"] for r_ in rows if r_["d"] > 0 and r_["ref_vs_fp64"]])
            )
            if any(r_["d"] > 0 for r_ in rows)
            else 1.0,
            "hc_head": head,
        }
        out["pairs"][pr]["step_time_median_from_11"] = {"test": tsc.step_time(mt), "ref": tsc.step_time(mg)}
        if a.catalog:
            import re
            import tempfile

            cat = re.sub(r"golden: \[[^\]]*\]", f"golden: [{ref}]", Path(a.catalog).read_text(), count=1)
            with tempfile.TemporaryDirectory() as td:
                cp = os.path.join(td, "catalog.yaml")
                Path(cp).write_text(cat)
                c = tsc.curves(cp, a.root, test, os.path.join(td, "w.janus"))
            out["pairs"][pr]["curves"] = {
                "r_max": c["r_max"],
                "checks": [{k: x[k] for k in ("id", "worst_key", "r")} for x in c["checks"]],
            }
        h = head.get("hc_head_base", {})
        print(
            f"{pr}: 逐步量逐位 {bm['equal']}，{out['pairs'][pr]['counts']}，hc_head_base 离 FP64 {h.get('test_vs_fp64', float('nan')):.3g}（对照 {h.get('ref_vs_fp64', float('nan')):.3g}）"
            + (f"，曲线最差 r {out['pairs'][pr]['curves']['r_max']:.3g}" if a.catalog else "")
        )
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
