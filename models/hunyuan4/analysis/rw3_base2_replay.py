"""真实宽度三层 base2 开 / 关短回放的汇总：两臂各自的判档、带符号均值的符号，以及两臂直接相减。

输入：
- `--results 名字=<janus results.json>`：`rw3-l4-b2r250.yaml`（或同口径的截窗目录）对每一臂判出的结果；
- `--root`：运行目录的根（含 GPU `rw1k-a` 与两臂 TPU 运行），读 `metrics.jsonl`；
- `--arms on=<运行名> off=<运行名>`、`--golden rw1k-a`、`--windows 1:251 101:251 1:165`。

输出（JSON）：
- 每臂每条检查：档（pass / inconclusive / fail）、r = 三个判定量各自 ÷ 噪声上限 N 后取最大、取到最大的那个量；
- 每臂对 GPU 基线的带符号均值（同检查目录的口径：`janus.checks.series.window_stats`，分母 golden，first_step 1，
  即第 1 步单独记成 d0、不进窗口统计），按窗口列，带标准误；
- 两臂直接相减：`(on − off) / |off|` 的带符号均值、MARE、最大单步，同窗口；再除以同一检查同窗口的 GPU 噪声上限 N，
  得到「两臂之间的差有几个 N」（N 只取全程 / 后段窗口里检查目录判过的那一个；1:165 用 rw3-l4-first164 的 N 时另给）。

    python rw3_base2_replay.py --root /tmp/hy4rw --arms on=b2r-on off=b2r-off \\
        --results on=<on 的 results.json> off=<off 的 results.json> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os

from janus.checks.series import window_stats

FIRST_STEP = 1  # 运行目录 meta 的 steps.first：第 1 步单独报（d0），窗口统计从第 2 步起，同 janus run
FIELDS = {  # 检查 id → (golden 字段, test 字段)
    "L4.loss": ("lm_loss", "lm_loss"),
    "L4.mtp_loss": ("mtp_loss_mt", "mtp_loss_mt"),
    "L4.indexer_loss": ("indexer_loss", "indexer_loss"),
    "L4.grad_norm": ("grad_norm", "grad_norm"),
    "L4.param_norm": ("param_norm_fp64", "param_norm_exact"),
}


def load_metrics(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[int(r["step"])] = r
    return out


def series(m, field):
    return {s: r[field] for s, r in m.items() if field in r and isinstance(r[field], (int, float))}


def parse_window(w):
    a, b = w.split(":")
    return int(a), int(b)


def judged(results):
    """results.json → {检查 id: {status, r, by, window, upper}}。"""
    out = {}
    for c in results["checks"]:
        up = (c.get("threshold") or {}).get("noise", {}).get("upper") or {}
        met = c.get("metrics") or {}
        rs = {}
        for k in ("signed_mean", "mare", "max_step"):
            v = met.get(f"{k}_max", met.get(k))
            if v is not None and up.get(k):
                rs[k] = abs(v) / up[k]
        by = max(rs, key=rs.get) if rs else None
        out[c["id"]] = {
            "status": c["status"],
            "r": rs.get(by) if by else None,
            "by": by,
            "r_each": rs,
            "window": met.get("window"),
            "upper": up,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", nargs=2, required=True, help="on=<运行名> off=<运行名>")
    ap.add_argument("--golden", default="rw1k-a")
    ap.add_argument("--results", nargs="*", default=[], help="名字=<results.json>（名字同 --arms）")
    ap.add_argument("--extra-results", nargs="*", default=[], help="标签=<results.json>：只抄档与 r（如前 164 步截窗）")
    ap.add_argument("--windows", nargs="*", default=["1:251", "101:251", "1:165"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = dict(x.split("=", 1) for x in a.arms)
    g = load_metrics(os.path.join(a.root, a.golden, "metrics.jsonl"))
    m = {k: load_metrics(os.path.join(a.root, v, "metrics.jsonl")) for k, v in arms.items()}
    res = {k: judged(json.load(open(p))) for k, p in (x.split("=", 1) for x in a.results)}
    extra = {k: judged(json.load(open(p))) for k, p in (x.split("=", 1) for x in a.extra_results)}
    windows = [parse_window(w) for w in a.windows]

    out = {
        "what": "真实宽度三层 base2 开 / 关短回放：两臂各对 GPU 基线的判档与带符号均值，以及两臂直接相减",
        "golden": a.golden,
        "arms": arms,
        "steps": {k: max(v) for k, v in m.items()},
        "judged": res,
        "judged_extra": extra,
        "signed_vs_golden": {},
        "arm_difference": {},
    }

    def n_upper(cid, w):
        for r in res.values():
            for k, v in r.items():
                base = k.removesuffix(".tail")
                if base == cid and v["window"] == f"{w[0]}:{w[1]}":
                    return v["upper"]
        for r in extra.values():
            for k, v in r.items():
                if k.removesuffix(".tail") == cid and v["window"] == f"{w[0]}:{w[1]}":
                    return v["upper"]
        return None

    for cid, (gf, tf) in FIELDS.items():
        for w in windows:
            key = f"{cid}@{w[0]}:{w[1]}"
            row = {}
            for arm, mm in m.items():
                if max(mm) < w[1] - 1:
                    continue
                _, st = window_stats(series(mm, tf), series(g, gf), w, "golden", FIRST_STEP)
                row[arm] = {k: st[k] for k in ("d0", "signed_mean", "signed_mean_se", "mare", "max_step")}
            if row:
                out["signed_vs_golden"][key] = row
            if all(max(mm) >= w[1] - 1 for mm in m.values()):
                _, st = window_stats(series(m["on"], tf), series(m["off"], tf), w, "golden", FIRST_STEP)
                d = {k: st[k] for k in ("d0", "signed_mean", "signed_mean_se", "mare", "max_step", "max_step_at")}
                up = n_upper(cid, w)
                if up:
                    d["in_units_of_N"] = {
                        k: abs(d[k]) / up[k] for k in ("signed_mean", "mare", "max_step") if up.get(k)
                    }
                    d["N"] = up
                out["arm_difference"][key] = d

    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k, r in res.items():
        print(k, {c: (v["status"], round(v["r"], 2) if v["r"] is not None else None, v["by"]) for c, v in r.items()})
    for k, r in out["signed_vs_golden"].items():
        print(k, {arm: f"{v['signed_mean']:+.3e}±{v['signed_mean_se']:.1e}" for arm, v in r.items()})
    for k, d in out["arm_difference"].items():
        print(
            "on-off", k, f"{d['signed_mean']:+.3e}", {kk: round(vv, 2) for kk, vv in d.get("in_units_of_N", {}).items()}
        )


if __name__ == "__main__":
    main()
