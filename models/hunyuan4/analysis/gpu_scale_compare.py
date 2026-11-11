"""混元 4 · GPU 扩展（05-gpu-scale）的几项核对，只读运行目录，写一个 JSON。

1. `data`：几份运行每一步吃的是不是同一批数据。按 `meta.json` 的并行设置把每个 rank、每个 microbatch 的输入指纹
   （`data/rank<r>.jsonl` 里 tokens / labels / loss_mask 的 sha256）还原到样本号
   （第 s 步第 m 个 microbatch、DP rank d 吃第 (s−1)·16 + m·DP + d 条；PP 时只看首段的 rank，DP rank = rank mod DP），
   再逐样本比。
2. `pairs`：几份运行两两之间、每条曲线逐步相对差（以第一个参数为分母）的最大单步、带符号均值、MARE；
   用来看一组同设置运行彼此差多少（同 janus loss_series 的三个判定量，不判档）。
3. `groups`：两组运行（如单机 8 份、多机 8 份）对同一份参照的逐步相对差，每份取窗口内带符号均值与最后一步的值，
   比两组的均值差与 Welch t（看换了执行方式有没有系统偏移；两组各自的散布就是噪声）。
4. `summary`：一份运行的概况：步数、步时中位数、显存峰值、指定步的主 loss / MTP loss / grad_norm、全程最大注意力 logit
   与 QK-Clip 触发的步数。
5. `bitwise`：两份运行逐位相同吗。比 `metrics.jsonl` 每一步的每个记录量（去掉计时与显存），`params_sha/` 的最终清单
   （模型参数与 buffer），有 `state_sha/` 时再比全部状态（FP32 主权重、优化器状态）。`--from-step` 只比这一步起的记录量
   （续跑的运行从存档的下一步开始记）。

用法：
    python gpu_scale_compare.py data --root <运行根目录> --runs g3-a mn-a mn-pp2 --json out.json
    python gpu_scale_compare.py pairs --root <运行根目录> --runs mn-a mn-b mn-c --steps 1:300 --json out.json
    python gpu_scale_compare.py groups --root <运行根目录> --ref g3-a --a g3-b … g3-h --b mn-a … mn-h --steps 1:300 --json out.json
    python gpu_scale_compare.py summary --root <运行根目录> --run rw1k-a --at 1,100,500,1000 --json out.json
    python gpu_scale_compare.py bitwise --root <运行根目录> --a ck-a --b ck-d --from-step 11 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os

SKIP = {"step_time", "step_time_s", "peak_alloc_gib"}  # 计时与显存每次都不同，不算数值


def _meta(root, run):
    return json.load(open(os.path.join(root, run, "meta.json")))


def samples(root: str, run: str) -> dict:
    """{样本号: (tokens, labels, loss_mask) 的 sha256}。"""
    par = _meta(root, run)["parallelism"]
    dp, gbs = par["dp"], par["gbs"]
    out = {}
    for r in range(dp):  # PP 外层：rank 0 … DP−1 是首段，DP rank = rank
        f = os.path.join(root, run, "data", f"rank{r}.jsonl")
        for rec in map(json.loads, open(f)):
            idx = (rec["step"] - 1) * gbs + rec["micro"] * dp + r
            key = (rec["tokens"], rec["labels"], rec["loss_mask"])
            assert out.setdefault(idx, key) == key, (run, idx)
    return out


def cmd_data(a):
    ref = samples(a.root, a.runs[0])
    res = {"reference": a.runs[0], "runs": {}}
    for run in a.runs[1:]:
        cur = samples(a.root, run)
        common = sorted(set(ref) & set(cur))
        bad = [i for i in common if ref[i] != cur[i]]
        res["runs"][run] = {
            "parallelism": _meta(a.root, run)["parallelism"],
            "samples": len(cur),
            "compared": len(common),
            "differ": len(bad),
            "first_differ": bad[:5],
            "same": not bad and len(common) == len(cur),
        }
    return res


def cmd_pairs(a):
    lo, hi = (int(x) for x in a.steps.split(":"))
    fields = a.fields.split(",")

    def series(run, f):
        d = {r["step"]: r for r in map(json.loads, open(os.path.join(a.root, run, "metrics.jsonl")))}
        return [d[s][f] for s in range(lo, hi + 1)]

    res = {"steps": [lo, hi], "pairs": {}}
    for i, x in enumerate(a.runs):
        for y in a.runs[i + 1 :]:
            out = {}
            for f in fields:
                sx, sy = series(x, f), series(y, f)
                d = [(v - u) / u for u, v in zip(sx, sy)]
                k = max(range(len(d)), key=lambda j: abs(d[j]))
                out[f] = {
                    "max_step": abs(d[k]),
                    "max_step_at": lo + k,
                    "signed_mean": sum(d) / len(d),
                    "mare": sum(abs(v) for v in d) / len(d),
                }
            res["pairs"][f"{x}|{y}"] = out
    return res


def cmd_groups(a):
    lo, hi = (int(x) for x in a.steps.split(":"))
    fields = a.fields.split(",")

    def series(run, f):
        d = {r["step"]: r for r in map(json.loads, open(os.path.join(a.root, run, "metrics.jsonl")))}
        return [d[s][f] for s in range(lo, hi + 1)]

    def stats(xs):
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        return {"n": n, "mean": m, "std": var**0.5}

    res = {"ref": a.ref, "a": a.a, "b": a.b, "steps": [lo, hi], "fields": {}}
    for f in fields:
        ref = series(a.ref, f)
        per = {}
        for run in a.a + a.b:
            d = [(v - u) / u for u, v in zip(ref, series(run, f))]
            per[run] = {"signed_mean": sum(d) / len(d), "last": d[-1]}
        out = {"runs": per}
        for q in ("signed_mean", "last"):
            sa, sb = stats([per[r][q] for r in a.a]), stats([per[r][q] for r in a.b])
            se = (sa["std"] ** 2 / sa["n"] + sb["std"] ** 2 / sb["n"]) ** 0.5
            out[q] = {
                "a": sa,
                "b": sb,
                "diff_b_minus_a": sb["mean"] - sa["mean"],
                "welch_t": (sb["mean"] - sa["mean"]) / se if se > 0 else None,
            }
        res["fields"][f] = out
    return res


def cmd_summary(a):
    rows = [json.loads(x) for x in open(os.path.join(a.root, a.run, "metrics.jsonl"))]
    by = {r["step"]: r for r in rows}
    times = sorted(r["step_time"] for r in rows if r["step"] >= 11)
    keys = ("lm_loss", "mtp_loss_mt", "indexer_loss", "grad_norm", "param_norm_fp64")
    return {
        "run": a.run,
        "parallelism": _meta(a.root, a.run)["parallelism"],
        "steps": len(rows),
        "step_time_median_from_11": times[len(times) // 2] if times else None,
        "peak_alloc_gib_max": max(r["peak_alloc_gib"] for r in rows),
        "at": {int(s): {k: by[int(s)].get(k) for k in keys} for s in a.at.split(",")},
        "max_attention_logit_max": max(r.get("max_attention_logit", 0.0) for r in rows),
        "qk_clip_steps": sum(1 for r in rows if r.get("qk.clipped_total", 0) > 0),
    }


def cmd_bitwise(a):
    def mets(run):
        return {d["step"]: d for d in map(json.loads, open(os.path.join(a.root, run, "metrics.jsonl")))}

    ma, mb = mets(a.a), mets(a.b)
    steps = sorted(s for s in mb if s >= a.from_step)
    assert all(s in ma for s in steps), "a 缺 b 的步"
    diff, n = [], 0
    for s in steps:
        for k in sorted(set(ma[s]) | set(mb[s])):
            if k in SKIP:
                continue
            n += 1
            if ma[s].get(k) != mb[s].get(k):
                diff.append([s, k, ma[s].get(k), mb[s].get(k)])
    res = {
        "a": a.a,
        "b": a.b,
        "steps": [steps[0], steps[-1]],
        "metrics": {"values": n, "differ": len(diff), "first_differ": diff[:5]},
    }
    for kind in ("params_sha", "state_sha"):
        da, db = os.path.join(a.root, a.a, kind), os.path.join(a.root, a.b, kind)
        if not (os.path.isdir(da) and os.path.isdir(db)):
            continue
        fa = sorted(f for f in os.listdir(da) if "init" not in f)
        fb = sorted(f for f in os.listdir(db) if "init" not in f)
        assert fa == fb, (fa, fb)
        tot = bad = 0
        ex = []
        for f in fa:
            ja, jb = json.load(open(os.path.join(da, f))), json.load(open(os.path.join(db, f)))
            ta, tb = ja.get("tensors", ja), jb.get("tensors", jb)
            for k in sorted(set(ta) | set(tb)):
                tot += 1
                if ta.get(k) != tb.get(k):
                    bad += 1
                    ex.append([f, k])
        res[kind] = {"files": len(fa), "entries": tot, "differ": bad, "first_differ": ex[:5]}
    res["bitwise_identical"] = res["metrics"]["differ"] == 0 and all(
        res[k]["differ"] == 0 for k in ("params_sha", "state_sha") if k in res
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--root", required=True)
    d.add_argument("--runs", nargs="+", required=True, help="第一份是参照")
    d.add_argument("--json", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--root", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="两两比较；每对以前一份为分母")
    p.add_argument("--steps", required=True, help="起:止（含两端）")
    p.add_argument("--fields", default="lm_loss,mtp_loss_mt,indexer_loss,grad_norm,param_norm_fp64")
    p.add_argument("--json", required=True)
    g = sub.add_parser("groups")
    g.add_argument("--root", required=True)
    g.add_argument("--ref", required=True, help="两组共同的参照（分母）")
    g.add_argument("--a", nargs="+", required=True)
    g.add_argument("--b", nargs="+", required=True)
    g.add_argument("--steps", required=True, help="起:止（含两端）")
    g.add_argument("--fields", default="lm_loss,mtp_loss_mt,indexer_loss,grad_norm,param_norm_fp64")
    g.add_argument("--json", required=True)
    m = sub.add_parser("summary")
    m.add_argument("--root", required=True)
    m.add_argument("--run", required=True)
    m.add_argument("--at", default="1", help="逗号分隔的步")
    m.add_argument("--json", required=True)
    b = sub.add_parser("bitwise")
    b.add_argument("--root", required=True)
    b.add_argument("--a", required=True)
    b.add_argument("--b", required=True)
    b.add_argument("--from-step", type=int, default=1)
    b.add_argument("--json", required=True)
    a = ap.parse_args()
    res = {"data": cmd_data, "pairs": cmd_pairs, "groups": cmd_groups, "summary": cmd_summary, "bitwise": cmd_bitwise}[
        a.cmd
    ](a)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
