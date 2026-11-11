"""生产配方臂：同配方 GPU 尺子与现行 g3 尺子差几倍，以及 TPU 那份在两把尺子下的 r。

    python models/hunyuan4/analysis/prodrecipe_ruler_compare.py \\
        --own models/hunyuan4/alignments/01-mini/results/t1pr-l4-gpu8.results.json \\
        --borrowed models/hunyuan4/alignments/01-mini/results/t1pr-l4.results.json --out <json>

两份结果文件的被测与基线相同（TPU p1t-a 对 GPU p1g-a），只有噪声对不同，所以 TPU 的判定量相同、上限不同。
r = |判定量| ÷ 上限（三个量各算，取最大），与正式判定同一口径；尺子比 = 同配方上限 ÷ 现行上限。
"""

from __future__ import annotations

import argparse
import json

KEYS = ("signed_mean", "mare", "max_step")


def table(path):
    d = json.load(open(path))
    out = {}
    for c in d["checks"]:
        up = c["threshold"]["noise"]["upper"]
        m = c["metrics"]
        v = {k: abs(m.get(k, m.get(k + "_max"))) for k in KEYS}
        out[c["id"]] = {
            "status": c["status"],
            "upper": {k: up[k] for k in KEYS},
            "value": v,
            "refs": c["threshold"]["noise"].get("refs"),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--own", required=True, help="同配方尺子的结果文件")
    p.add_argument("--borrowed", required=True, help="借现行 g3 尺子的结果文件")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    own, bor = table(a.own), table(a.borrowed)
    rows = []
    for cid, o in own.items():
        b = bor[cid]
        assert all(abs(o["value"][k] - b["value"][k]) <= 1e-12 * max(1.0, abs(b["value"][k])) for k in KEYS), cid
        r_own = {k: o["value"][k] / o["upper"][k] for k in KEYS}
        r_bor = {k: b["value"][k] / b["upper"][k] for k in KEYS}
        ko, kb = max(r_own, key=r_own.get), max(r_bor, key=r_bor.get)
        rows.append(
            {
                "id": cid,
                "status_own": o["status"],
                "status_borrowed": b["status"],
                "r_own": r_own[ko],
                "r_own_metric": ko,
                "r_borrowed": r_bor[kb],
                "r_borrowed_metric": kb,
                "ruler_ratio_own_over_borrowed": {k: o["upper"][k] / b["upper"][k] for k in KEYS},
                "upper_own": o["upper"],
                "upper_borrowed": b["upper"],
                "tpu_value": o["value"],
                "refs": [o["refs"], b["refs"]],
            }
        )
    res = {
        "schema_version": 1,
        "stage": "t1pr-ruler-compare",
        "model": "hunyuan4-mini",
        "inputs": {"own": a.own.split("/")[-1], "borrowed": a.borrowed.split("/")[-1]},
        "checks": rows,
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for r in rows:
        rr = r["ruler_ratio_own_over_borrowed"]
        print(
            f"{r['id']:22s} own {r['status_own']:12s} r={r['r_own']:.2f}({r['r_own_metric']})  "
            f"borrowed r={r['r_borrowed']:.2f}  ruler×: " + " ".join(f"{k}={rr[k]:.2f}" for k in KEYS)
        )


if __name__ == "__main__":
    main()
