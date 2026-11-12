"""issue #10：用「只改执行方式」的 TPU 运行造一把尺子，判各上下文并行（CP）方案的逐步 loss 差。

口径照 docs/guide/ruler.md 的逐步曲线：dₜ = (aₜ − bₜ) / |bₜ|，第一步单独报不进统计；每个窗口算带符号均值、MARE、最大单步；
噪声上限 N = 参照运行两两配对、同一窗口同一统计量的最大值；r = 被测 / N，三个统计量取最差；r ≤ 1 在噪声内，1–2 待定，> 2 超。
和《尺子》的差别：参照不是 GPU 同设置重跑，而是 TPU 同一程序只改编译开关（片上内存上限、splash 分块）的几份。
TPU 同程序重跑逐位相同，这几份的差来自换了求和顺序之后在 MoE 路由上被放大的轨迹分叉，比 GPU 重跑的差大得多，
所以这把尺子回答的是「CP 和换一个编译开关分不分得开」，不是「CP 和 GPU 分不分得开」。

逐步数据是 MaxText 的 metrics 文件（每行 JSON，step 从 0 起，loss 在 learning/loss），或本实践 results/series/<运行>.json。

  python cp_tpu_ruler.py --ref base=<B.jsonl> --ref vmem64=<...> ... --test ring=<...>[@<另一个基线 jsonl>] --windows 1:3000,100:3000,2000:3000 --out x.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def load(p):
    """MaxText metrics（每行 JSON）或本实践 results/series/<运行>.json（{"first_step", "loss": [...]}）。"""
    if str(p).endswith(".json"):
        x = json.loads(Path(p).read_text())
        return {x["first_step"] + i: float(v) for i, v in enumerate(x["loss"])}
    d = {}
    for line in open(p):
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in x and "learning/loss" in x:
            d[int(x["step"])] = float(x["learning/loss"])
    return d


def stats(a, b, w):
    lo, hi = w
    s = [k for k in sorted(set(a) & set(b)) if lo <= k < hi and k != min(b)]
    d = np.array([(a[k] - b[k]) / abs(b[k]) for k in s])
    i = int(np.argmax(np.abs(d)))
    return {"signed_mean": float(d.mean()), "mare": float(np.abs(d).mean()), "max_step": float(np.abs(d[i])),
            "max_at": s[i], "n": len(s)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", required=True, help="名=metrics；第一个是基线")
    ap.add_argument("--test", action="append", default=[], help="名=metrics[@基线 metrics]（默认对第一个 --ref）")
    ap.add_argument("--windows", default="1:3000,100:3000,2000:3000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    wins = [tuple(int(x) for x in w.split(":")) for w in a.windows.split(",")]
    refs = [(k, load(v)) for k, v in (x.split("=", 1) for x in a.ref)]
    base_name, base = refs[0]
    res = {"refs": [k for k, _ in refs], "base": base_name, "windows": {}, "tests": {}}
    keys = ["signed_mean", "mare", "max_step"]
    for w in wins:
        wk = f"{w[0]}:{w[1]}"
        pairs = {}
        for (n1, d1), (n2, d2) in itertools.combinations(refs, 2):
            st = stats(d2, d1, w)
            pairs[f"{n2}-{n1}"] = st
        N = {k: max(abs(p[k]) for p in pairs.values()) for k in keys}
        res["windows"][wk] = {"N": N, "pairs": pairs}
        # leave-one-out：每一份参照轮流当被测（对基线；基线自己当被测时对第二份），N 用其余份两两
        loo = []
        for i, (ni, di) in enumerate(refs):
            rest = [r for j, r in enumerate(refs) if j != i]
            bn, bd = (refs[1] if i == 0 else refs[0])
            if len(rest) < 3:
                break
            Nr = {k: max(abs(stats(d2, d1, w)[k]) for (_, d1), (_, d2) in itertools.combinations(rest, 2)) for k in keys}
            st = stats(di, bd, w)
            loo.append({"test": ni, "vs": bn, "r": max(abs(st[k]) / Nr[k] for k in keys)})
        res["windows"][wk]["loo"] = loo
    for t in a.test:
        name, path = t.split("=", 1)
        bpath = None
        if "@" in path:
            path, bpath = path.split("@", 1)
        td, bd = load(path), (load(bpath) if bpath else base)
        out = {"vs": bpath or base_name}
        for w in wins:
            wk = f"{w[0]}:{w[1]}"
            st = stats(td, bd, w)
            N = res["windows"][wk]["N"]
            st["r"] = {k: abs(st[k]) / N[k] for k in keys}
            st["r_worst"] = max(st["r"].values())
            st["verdict"] = "在噪声内" if st["r_worst"] <= 1 else ("待定" if st["r_worst"] <= 2 else "超")
            out[wk] = st
        res["tests"][name] = out
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for wk, wv in res["windows"].items():
        print(f"[{wk}] N signed {wv['N']['signed_mean']:.2e} MARE {wv['N']['mare']:.2e} max {wv['N']['max_step']:.2e}; "
              f"LOO r max {max([x['r'] for x in wv['loo']] or [float('nan')]):.2f}")
    for name, out in res["tests"].items():
        cells = "  ".join(f"{wk}: sgn {out[wk]['signed_mean']:+.2e} r {out[wk]['r_worst']:.2f} {out[wk]['verdict']}"
                          for wk in res["windows"])
        print(f"{name:14s} {cells}")


if __name__ == "__main__":
    main()
