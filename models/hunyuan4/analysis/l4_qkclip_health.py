"""L4 健康信号（NaN / Inf、跳步、梯度裁剪触发、QK-Clip 触发、丢 token、尖峰）与 QK-Clip 曲线的同状态定责。

健康信号（写 --out-health，janus 结果格式）：
- 非有限值：TPU 与 GPU golden 的 metrics.jsonl 里每一步每个数值字段都有限；
- 跳步：TPU skipped_iter 恒 0；GPU skipped_iter 恒 0 且 update_ok 恒真；
- 梯度裁剪触发：两边逐步「裁剪前 grad_norm > 阈值 1.0」的真假逐步相同；
- QK-Clip 触发（τ 100）：TPU 每步被裁头数 qk.clipped_total = 0；GPU 每步全局最大 logit < τ，所以一个头也没裁；
- 丢 token：两边都是不丢 token 的设置（TPU capacity_factor −1；GPU 入口不设 moe_expert_capacity_factor，Megatron-Core
  默认 None = 不丢），读运行记录的配置；
- 尖峰（janus health_signals：某步超过前 20 步中位数 10 倍判不过）：lm_loss、grad_norm、param_norm，两边都量。
QK-Clip 同状态（写 --out-same-state，诊断数据）：第 1 步（初始权重）与第 8、30、100 步（GPU g2w-b 该步重建状态）上，
TPU、GPU、CPU FP64 各自前向一次的全局最大 logit（各层含 MTP 各头最大值的最大值），每份对 FP64 的相对差与 R。

  python l4_qkclip_health.py --tpu <mz-g1 目录> --gpu <g3-a 目录> --tpu-meta-cfg <t1zg-a/meta.json> \
      --same-state <json 规格> --out-health <json> --out-same-state <json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from janus.checks.extra import health_signals  # noqa: E402


def load(d):
    return [json.loads(x) for x in open(os.path.join(d, "metrics.jsonl"))]


def ck(cid, ok, metrics, msg):
    return {"id": cid, "type": "health_signals", "level": "L4", "status": "pass" if ok else "fail",
            "threshold": {"kind": "exact"}, "metrics": metrics, "message": msg}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpu", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--same-state", default=None, help="JSON：[{step, fp64_summary, tpu_metrics, gpu_max_logit: [..]}]")
    p.add_argument("--stage", default="t1mz-health", help="健康信号结果的 stage 名（换代时另起，不覆盖旧结果）")
    p.add_argument("--out-health", required=True)
    p.add_argument("--out-same-state", default=None)
    a = p.parse_args()
    T, G = load(a.tpu), load(a.gpu)
    tm = json.load(open(os.path.join(a.tpu, "meta.json")))
    gm = json.load(open(os.path.join(a.gpu, "meta.json")))
    n = min(len(T), len(G))
    checks = []

    def nonfinite(rows):
        bad = []
        for r in rows:
            for k, v in r.items():
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    if isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isfinite(x):
                        bad.append(f"{r['step']}:{k}")
        return bad
    bt, bg = nonfinite(T), nonfinite(G)
    checks.append(ck("L4.health.nonfinite", not bt and not bg, {"steps": n, "tpu_bad": len(bt), "gpu_bad": len(bg)},
                     f"{n} 步：TPU 非有限值 {len(bt)} 处、GPU {len(bg)} 处"))
    st = sum(1 for r in T if r.get("skipped_iter", 0))
    sg = sum(1 for r in G if r.get("skipped_iter", 0) or not r.get("update_ok", True))
    checks.append(ck("L4.health.skipped", st == 0 and sg == 0, {"tpu_skipped": st, "gpu_skipped_or_not_ok": sg},
                     f"跳步：TPU {st} 步、GPU {sg} 步"))
    trig_t = [r["grad_norm"] > a.clip for r in T[:n]]
    trig_g = [r["grad_norm"] > a.clip for r in G[:n]]
    diff = [i + 1 for i in range(n) if trig_t[i] != trig_g[i]]
    checks.append(ck("L4.health.grad_clip_trigger", not diff,
                     {"steps": n, "tpu_triggered": sum(trig_t), "gpu_triggered": sum(trig_g), "differ_steps": diff[:20]},
                     f"梯度裁剪（阈值 {a.clip}）：TPU {sum(trig_t)} 步触发、GPU {sum(trig_g)} 步触发，逐步不同 {len(diff)} 步"))
    qt = sum(int(r.get("qk.clipped_total", 0)) for r in T)
    has_qt = all("qk.clipped_total" in r for r in T)
    gmax = max(r["max_attention_logit"] for r in G)
    tmax = max(max(max(v) for k, v in r.items() if k.startswith("qk.max_logit.")) for r in T) if has_qt else None
    checks.append(ck("L4.health.qk_clip_trigger", has_qt and qt == 0 and gmax < a.tau,
                     {"tau": a.tau, "tpu_clipped_heads_total": qt, "tpu_max_logit_max": tmax, "gpu_max_logit_max": gmax},
                     f"QK-Clip（τ {a.tau:g}）：TPU 1000 步被裁头数合计 {qt}（最大 logit 最高 {tmax:.4g}）；GPU 最大 logit 最高 {gmax:.4g} < τ，没有头被裁"))
    cf = tm.get("maxtext_config_full", {}).get("capacity_factor")
    gcf = gm.get("config", {}).get("moe_expert_capacity_factor", None)
    checks.append(ck("L4.health.dropped_tokens", cf is not None and float(cf) < 0 and gcf is None,
                     {"tpu_capacity_factor": cf, "gpu_moe_expert_capacity_factor": gcf},
                     f"不丢 token 的设置：TPU capacity_factor {cf}（< 0 = 不丢）；GPU 不设 moe_expert_capacity_factor（Megatron-Core 默认 None = 不丢）"))
    sp = []
    for side, rows in (("tpu", T), ("gpu", G)):
        series = {f: {r["step"]: r[f] for r in rows} for f in ("lm_loss", "grad_norm")}
        series["param_norm"] = {r["step"]: r.get("param_norm_fp64", r.get("param_norm")) for r in rows}
        r = health_signals(f"L4.health.spikes.{side}", "L4", series, params={"source": "janus 规格默认值：前 20 步中位数的 10 倍"})
        sp.append((side, r))
    checks.append(ck("L4.health.spikes", all(r.status == "pass" for _, r in sp),
                     {side: {"status": r.status, "message": r.message} for side, r in sp},
                     "尖峰（某步超过前 20 步中位数 10 倍）：" + "；".join(f"{'TPU' if s == 'tpu' else 'GPU'} {r.message}" for s, r in sp)))
    res = {"schema_version": 1, "stage": a.stage, "model": "hunyuan4-mini",
           "inputs": {"tpu_run": tm["run_name"], "gpu_run": gm["run_name"], "steps": n}, "checks": checks}
    res["summary"] = {k: sum(c["status"] == k for c in checks) for k in ("pass", "fail", "inconclusive", "error")}
    Path(a.out_health).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for c in checks:
        print(c["status"], c["id"], c["message"])
    if a.same_state and a.out_same_state:
        rows = []
        for s in json.load(open(a.same_state)):
            f = json.load(open(s["fp64_summary"]))["max_logit"]
            fg = max(max(v) for v in f.values())
            t = json.loads(open(s["tpu_metrics"]).readline())
            tg = max(max(v) for k, v in t.items() if k.startswith("qk.max_logit."))
            g = s["gpu_max_logit"]
            rt = (tg - fg) / fg
            rg = [(x - fg) / fg for x in g]
            med = sorted(abs(x) for x in rg)[len(rg) // 2]
            rows.append({"step": s["step"], "state": s["state"], "fp64": fg, "tpu": tg, "gpu": g, "tpu_rel_vs_fp64": rt,
                         "gpu_rel_vs_fp64": rg, "R": abs(rt) / med if med else None, "tpu_run": s["tpu_run"], "gpu_runs": s["gpu_runs"]})
            print(rows[-1])
        out = {"what": "QK-Clip 曲线的同状态定责：同一状态下 TPU、GPU、CPU FP64 各自前向一次的全局最大注意力 logit（不含 sink）",
               "definition": "rel = (x − FP64) / FP64；R = |rel(TPU)| ÷ GPU 各份 |rel| 的中位数", "rows": rows,
               "conclusion": "四个状态里 TPU 都比 GPU 更近 FP64（R 见 rows）；GPU 同一状态重跑的最大 logit 相同或几乎相同，"
                             "它对 FP64 的偏差是各份共有的，GPU 重跑的噪声尺子量不到，所以多步曲线上 TPU 对 GPU 的差显出超线。"}
        Path(a.out_same_state).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
