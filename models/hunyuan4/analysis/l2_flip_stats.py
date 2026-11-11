"""L2「索引器 top-k 翻转与路由翻转（统计）」：从路由一致率的四份结果（第 1、8、30、100 步，routing_agreement.py 的输出）
摘出现行一代 TPU（t1zw 那一代的抓数；M50 只改 KD 记录值、不改前向，选择与 t1zg 相同）每层的翻转统计，写成一份结果。

每层记：TPU 对 CPU FP64 自选、GPU 各份对 FP64、TPU 对 GPU 各份、GPU 两两的翻转率（DSA：不重合率与「选中集合不全同」的
query 比例）；路由另记 FP64 分差分桶（[0, 1e-4)、[1e-4, 1e-3)、[1e-3, 1e-2)、≥ 1e-2）里 token 数与 TPU 对 GPU 的翻转率。
判定沿用路由一致率的对真值口径（PROTOCOL 第 5 节）：每层 R = TPU 对 FP64 ÷ GPU 各份对 FP64 的最大值，≤ 1.2 过，全部层取最差。
这一格只是把统计摊开看，判定与 L3 路由一致率同一条规则、同一批数。

  python l2_flip_stats.py --results-dir models/hunyuan4/alignments/01-mini/results --out <结果 json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STEPS = (1, 8, 30, 100)
DIAG = {"tw-ra1z"}  # 关 base2 的诊断代，不算现行


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out_r, out_d = [], []
    worst = {"routing": (0.0, None), "dsa": (0.0, None)}
    for s in STEPS:
        name = f"t1-route-agree-s{s}.json"
        d = json.loads((Path(a.results_dir) / name).read_text())
        tp = [t for t in d["tpu"] if t not in DIAG]
        assert len(tp) == 1, tp
        tp = tp[0]
        gs = d["gpu"]
        for L, v in d["routing"].items():
            fr = v["flip_rate"]
            mb = v["margin_buckets"]
            row = {"step": s, "layer": L, "tpu_run": tp, "gpu_runs": gs, "n_tokens": v["n_tokens"],
                   "tpu_vs_fp64": v["vs_fp64_flip_rate"][tp], "gpu_vs_fp64": [v["vs_fp64_flip_rate"][g] for g in gs],
                   "tpu_vs_gpu": [fr[f"{tp}~{g}"] for g in gs],
                   "gpu_pairs": [x for k, x in fr.items() if not k.startswith(tp) and not any(k.startswith(t) for t in DIAG)],
                   "margin_edges": mb["edges"], "margin_count": mb["count"],
                   "tpu_vs_gpu_by_margin": [mb["flip_rate"][f"{tp}~{gs[0]}"][i] for i in range(len(mb["count"]))],
                   "R": v["tpu"][tp]["truth"]["R"]}
            out_r.append(row)
            if row["R"] > worst["routing"][0]:
                worst["routing"] = (row["R"], f"第 {s} 步第 {L} 层")
        for L, v in d["dsa"].items():
            row = {"step": s, "layer": L, "tpu_run": tp, "n_queries": v["n_queries"],
                   "tpu_vs_fp64": v["vs_fp64_nonoverlap_rate"][tp], "gpu_vs_fp64": [v["vs_fp64_nonoverlap_rate"][g] for g in gs],
                   "tpu_vs_gpu": [v["nonoverlap_rate"][f"{tp}~{g}"] for g in gs],
                   "tpu_vs_gpu_query_differ_frac": [v["query_differ_frac"][f"{tp}~{g}"] for g in gs],
                   "gpu_pairs": [x for k, x in v["nonoverlap_rate"].items() if not k.startswith(tp) and not any(k.startswith(t) for t in DIAG)],
                   "R": v["tpu"][tp]["truth"]["R"]}
            out_d.append(row)
            if row["R"] > worst["dsa"][0]:
                worst["dsa"] = (row["R"], f"第 {s} 步第 {L} 层")

    def chk(cid, rows, kind, unit):
        r, where = worst[kind]
        w = max(rows, key=lambda x: x["R"])
        st = "pass" if r <= 1.2 else ("inconclusive" if r <= 2 else "fail")
        return {"id": cid, "type": "discrete_agreement", "level": "L2", "status": st,
                "threshold": {"kind": "numeric", "values": {"r_ratio": 1.2},
                              "source": "PROTOCOL.md 第 5 节「路由一致率」：对真值口径，每层 R ≤ 1.2 过"},
                "metrics": {"r_ratio": r, "rel_l2_gpu_vs_ref": max(w["gpu_vs_fp64"]), "rel_l2_tpu_vs_ref": w["tpu_vs_fp64"],
                            "layers": len(rows), "steps": list(STEPS),
                            "tpu_vs_fp64_range": [min(x["tpu_vs_fp64"] for x in rows), max(x["tpu_vs_fp64"] for x in rows)],
                            "gpu_vs_fp64_range": [min(min(x["gpu_vs_fp64"]) for x in rows), max(max(x["gpu_vs_fp64"]) for x in rows)]},
                "message": f"{kind == 'routing' and '路由' or 'DSA top-k'} {len(rows)} 个（步, 层）：TPU 对 FP64 的{unit} "
                           f"{min(x['tpu_vs_fp64'] for x in rows):.3g}–{max(x['tpu_vs_fp64'] for x in rows):.3g}，"
                           f"GPU {min(min(x['gpu_vs_fp64']) for x in rows):.3g}–{max(max(x['gpu_vs_fp64']) for x in rows):.3g}；"
                           f"最差 {where}：GPU {max(w['gpu_vs_fp64']):.3g}、TPU {w['tpu_vs_fp64']:.3g} → R {r:.3f}",
                "details_inline": {"rows": rows}}

    res = {"schema_version": 1, "stage": "t1-l2-flip-stats", "model": "hunyuan4-mini",
           "inputs": {"route_agree": [f"t1-route-agree-s{s}.json" for s in STEPS]},
           "checks": [chk("L2.flips.routing", out_r, "routing", "翻转率"), chk("L2.flips.dsa_topk", out_d, "dsa", "不重合率")]}
    hi = [(x["step"], x["layer"], x["tpu_vs_gpu_by_margin"][-1]) for x in out_r]
    res["checks"][0]["metrics"]["tpu_vs_gpu_flip_rate_margin_ge_1e-2_max"] = max(h[2] for h in hi)
    res["summary"] = {k: sum(c["status"] == k for c in res["checks"]) for k in ("pass", "fail", "inconclusive", "error")}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for c in res["checks"]:
        print(c["status"], c["id"], c["message"])


if __name__ == "__main__":
    main()
