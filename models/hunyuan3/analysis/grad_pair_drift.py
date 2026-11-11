"""同一状态、同一 batch 的两份梯度，差值沿漂移方向每步多走多少（案例 01「逐步拆分」「同一状态多个 batch」「反向互换」各行的数）。

    per_step_frac_of_dd = −lr · c · ⟨(g_a − g_f) / (√v̂ + ε), d⟩ / |d|²

g_a 是被比的梯度，g_f 是参照（例：GPU 全程 FP32、BF16 权重）；v̂ 取一份运行第 K 步的 exp_avg_sq 做 Adam 逐元素缩放（偏差校正按 K）；
c 是第 K 步的裁剪系数（side 决定 Megatron / optax 的公式）；d 是方向（两份主权重之差，step_decomp.load_drift 的写法）。
负 = 这一步因为 g_a 与参照不同，沿 d 往反方向多走了漂移本身的这么多。按 step_decomp.groups 分组（全体、第 0 层、第 0 层输出投影……）。

  python grad_pair_drift.py --a <运行>@<步>[:前缀] --f <运行>@<步>[:前缀] --v <运行>@<K>[:gpu|tpu] \\
      --dir 名=<A 运行>@<s>,<B 运行>@<t> [--dir …] [--out <json>]

前缀默认 grad.（GPU 抓数）；TPU 的 --grad-probe-at 探针用 gp_bf16. / gp_f32.。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import step_decomp as sd

from janus.store import Run


def _spec(x: str, default: str = "grad."):
    path, _, pre = x.partition(":")
    r, s = path.rsplit("@", 1)
    return Run.open(r), int(s), (pre or default)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True)
    p.add_argument("--f", required=True)
    p.add_argument("--v", required=True)
    p.add_argument("--dir", action="append", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    A, sa, pa = _spec(a.a)
    F, sf, pf = _spec(a.f)
    vpath, _, side = a.v.partition(":")
    V, k = Run.open(vpath.rsplit("@", 1)[0]), int(vpath.rsplit("@", 1)[1])
    side = side or "gpu"
    dirs = {x.split("=", 1)[0]: sd.load_drift(x.split("=", 1)[1]) for x in a.dir}
    names = sorted(set.intersection(*[set(d) for d in dirs.values()]))
    na, nf = set(A.names(sa)), set(F.names(sf))
    res = {dn: {} for dn in dirs}
    for n in names:
        if pa + n not in na or pf + n not in nf:
            continue
        ga = sd.f64(A.tensor(sa, pa + n))
        gf = sd.f64(F.tensor(sf, pf + n))
        v = sd.f64(V.tensor(k, "exp_avg_sq." + n))
        pre = 1.0 / (np.sqrt(v / (1 - sd.B2**k)) + sd.EPS)
        dg = ga - gf
        for dn, D in dirs.items():
            d = D[n]
            for g in sd.groups(n):
                r = res[dn].setdefault(g, {"dgpre_d": 0.0, "fpre_d": 0.0, "dd": 0.0, "dg_sq": 0.0, "f_sq": 0.0})
                r["dgpre_d"] += float(np.sum(dg * pre * d))
                r["fpre_d"] += float(np.sum(gf * pre * d))
                r["dd"] += float(np.sum(d * d))
                r["dg_sq"] += float(np.sum(dg * dg))
                r["f_sq"] += float(np.sum(gf * gf))
    c = sd.clip_coef(float(V.metrics()[k]["grad_norm"]), side)
    for rr in res.values():
        for r in rr.values():
            r["per_step_frac_of_dd"] = -sd.LR * c * r["dgpre_d"] / r["dd"]
            r["per_step_update_frac_of_dd"] = -sd.LR * c * r["fpre_d"] / r["dd"]
            r["rel_l2"] = (r["dg_sq"] / r["f_sq"]) ** 0.5
    for dn, rr in res.items():
        print(
            f"[{dn}] "
            + "  ".join(
                f"{g}: {rr[g]['per_step_frac_of_dd']:+.3e} (rel {rr[g]['rel_l2']:.2e})"
                for g in ("all", "layer0", "l0_proj", "attn_proj")
                if g in rr
            )
        )
    if a.out:
        Path(a.out).write_text(json.dumps({"a": a.a, "f": a.f, "v": a.v, "clip_coef": c, "res": res}, indent=1))


if __name__ == "__main__":
    main()
