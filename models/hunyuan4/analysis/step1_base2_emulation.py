"""第 1 步主 loss 的偏差定责：在 CPU FP64 前向里只加 TPU splash base2 模式的那一处差，看 loss 移多少，和 TPU 实测比。

输入：
- `--micro 名字=<jsonl>`：`frameworks/megatron/step1_ref.py --forward-only`（可带 `--emulate-base2`）每行打印的 MICRO 记录；
  名字写成 `<截层>:<变体>`，变体 `fp64` 是原样参照，其余（kappa / full / q16 / kappa-L0 …）对同一截层的 `fp64` 算相对移动；
- `--observed <json>`：两边平台第 1 步的实测（`{截层: {"fp64_ref_note": …, "runs": {名字: {"side", "lm_loss", "mtp_loss_mt"?, "note"?}}}}`）。

合成口径同 `step1_loss_vs_fp64.py`：主 loss = Σ loss 和 ÷ Σ 有效 token 数；MTP（MaxText 口径）= 0.1 × Σ(MTP 均值 × token 数) ÷ Σ token 数。
输出每个变体的相对移动（全局与逐行的最小 / 最大）、平台实测对 FP64 的相对差，以及「TPU 减去模拟出的 base2 移动」（推断：
假设各处差可加）。

    python step1_base2_emulation.py --micro D3:fp64=a.jsonl D3:kappa=b.jsonl … --observed obs.json --json out.json
    python step1_base2_emulation.py --micro-all step1-base2-emulation.micro.jsonl --observed obs.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics


def read_micro(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("MICRO "):
            line = line[6:]
        if line.startswith("{"):
            r = json.loads(line)
            rows[r["row"]] = r
    return rows


def agg(rows, mtp_factor=0.1):
    n = sum(r["n_tokens"] for r in rows.values())
    out = {"rows": len(rows), "lm_loss": sum(r["loss_sum"] for r in rows.values()) / n}
    if all("mtp_mean" in r for r in rows.values()):
        out["mtp_loss_mt"] = mtp_factor * sum(r["mtp_mean"] * r["n_tokens"] for r in rows.values()) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", nargs="*", default=[], help="名字=MICRO jsonl（名字 <截层>:<变体>）")
    ap.add_argument(
        "--micro-all", default="", help="合在一起的 jsonl：每行一条 MICRO 记录另加 run 字段（<截层>:<变体>）"
    )
    ap.add_argument("--observed", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    runs = {}
    for spec in a.micro:
        name, path = spec.split("=", 1)
        stage, var = name.split(":", 1)
        runs.setdefault(stage, {})[var] = read_micro(path)
    if a.micro_all:
        for line in open(a.micro_all):
            if line.strip():
                r = json.loads(line)
                stage, var = r.pop("run").split(":", 1)
                runs.setdefault(stage, {}).setdefault(var, {})[r["row"]] = r
    obs = json.load(open(a.observed))
    out = {
        "what": "CPU FP64 前向只加 TPU splash base2 那一处差（step1_ref.py --emulate-base2）的 loss 移动，对 TPU / GPU 第 1 步实测",
        "kappa": 1.4453125 / 1.4426950408889634,
        "stages": {},
    }
    for stage, vs in runs.items():
        base_rows = vs["fp64"]
        base = agg(base_rows)
        st = {"fp64": base, "emulated": {}, "observed": {}}
        for var, rows in vs.items():
            if var == "fp64":
                continue
            common = sorted(set(rows) & set(base_rows))
            assert len(common) == len(rows), (stage, var, "行和参照对不上")
            g = agg(rows)
            sub = agg({r: base_rows[r] for r in common})
            e = {"rows": len(common)}
            for k in ("lm_loss", "mtp_loss_mt"):
                if k in g:
                    e[k] = g[k]
                    e[k + ".shift_rel"] = (g[k] - sub[k]) / sub[k]
            per = [(rows[r]["loss_sum"] - base_rows[r]["loss_sum"]) / base_rows[r]["loss_sum"] for r in common]
            e["lm_loss.shift_rel_per_row"] = {"min": min(per), "median": statistics.median(per), "max": max(per)}
            if "mtp_loss_mt" in g:
                pm = [(rows[r]["mtp_mean"] - base_rows[r]["mtp_mean"]) / base_rows[r]["mtp_mean"] for r in common]
                e["mtp.shift_rel_per_row"] = {"min": min(pm), "median": statistics.median(pm), "max": max(pm)}
            st["emulated"][var] = e
        o = obs.get(stage, {})
        for rn, rv in o.get("runs", {}).items():
            d = {"side": rv["side"]}
            for k in ("lm_loss", "mtp_loss_mt"):
                if rv.get(k) is not None and k in base:
                    d[k] = rv[k]
                    d[k + ".rel_vs_fp64"] = (rv[k] - base[k]) / base[k]
            if rv.get("note"):
                d["note"] = rv["note"]
            st["observed"][rn] = d
        gpu = [v["lm_loss.rel_vs_fp64"] for v in st["observed"].values() if v["side"] == "gpu"]
        if gpu:
            gmed = statistics.median(gpu)
            st["gpu_lm_rel_median"] = gmed
            kap = st["emulated"].get("kappa", {}).get("lm_loss.shift_rel")
            full = st["emulated"].get("full", {}).get("lm_loss.shift_rel")
            q16 = st["emulated"].get("q16", {}).get("lm_loss.shift_rel")
            st["R_lm"] = {}
            for rn, v in st["observed"].items():
                if v["side"] != "tpu" or "lm_loss.rel_vs_fp64" not in v:
                    continue
                t = v["lm_loss.rel_vs_fp64"]
                r = {"R": abs(t) / abs(gmed)}
                if kap is not None and v.get("base2", True) is not False and not o["runs"][rn].get("base2_off"):
                    r["minus_kappa_rel"] = t - kap
                    r["R_minus_kappa"] = abs(t - kap) / abs(gmed)
                    if full is not None and q16 is not None:
                        r["minus_full_minus_q16_rel"] = t - (full - q16)
                        r["R_minus_base2"] = abs(t - (full - q16)) / abs(gmed)
                st["R_lm"][rn] = r
        out["stages"][stage] = st
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
