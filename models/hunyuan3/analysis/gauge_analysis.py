"""案例 01 反向复现与 softmax 规范方向的三项计算（GPU 诊断臂见 frameworks/megatron/rr_arms.py）。

softmax 对「每个 token 的全部 logits 加同一个数」不变：输出层每一行都加同一个向量 c，loss 精确不变，所以输出层
梯度的跨行和 Σ_v g_v = Σ_t (Σ_v dlogit_tv) h_t 在精确算术下是 0（规范方向没有梯度）。这里的五个子命令：

  arms    各臂（300 / 1000 步）的输出层平方和、param_norm、loss 偏移，对同设置 GPU 对照的均值比；1000 步另给 301–1000
          步 loss 带符号均值对 g1k-a 的值与它在 10 次 GPU 运行里的 z
            python gauge_analysis.py arms --ctl <对照 metrics.jsonl>... --band <10 次运行 metrics.jsonl>... \
              --golden <g1k-a metrics.jsonl> --run <名>=<metrics.jsonl> ... [--out json]
  step1   第 1 步输出层 FP32 梯度（裁剪前）：跨行和 r 的大小与方向、只看没出现过的词那些行时的共同分量；并用一个
          microbatch 的 logits 按「FP32 算 (p − y)/N 再舍 BF16」重算 r 的方向，看 GPU 的 r 是不是这一步舍入造成的
            python gauge_analysis.py step1 --test <TPU grad.other.safetensors> --ref <GPU grad.other.safetensors> \
              --logits <logits.safetensors> --hidden <final_norm.safetensors> --batches <golden data/batches 目录> [--out json]
  logits  第 1000 步 rank 0 第一个 microbatch 的 logits 与最后归一化输出：logsumexp、原始 logit 均值（全部 / 没出现过 /
          常见词）、没出现过的词的平均 log 概率与概率质量、最后隐状态 token 均值的范数与几个维度
            python gauge_analysis.py logits --run <名>=<step_1000 目录> ... --counts <label_counts.npz> [--key c1000] [--out json]

  rows    逐行统计按词频分桶：输出层每行 w·u 的和（范数变化的一阶项，Adam 收缩 = −2·lr·Σ w·u）、√v̂ 与 |u| 的均值，
          被测 / 参照 / 第二参照逐步比（--row-stats-every 写的 output_row_wu / output_row_g / output_row_u）
            python gauge_analysis.py rows --test <metrics.jsonl> --ref <metrics.jsonl> [--pair <metrics.jsonl>] \
              --counts <label_counts.npz> [--key c300] [--steps 25,50,100,200,300,400] [--out json]

  probe   残差流探针（两侧 --resid-probe-every）逐层逐步比：每个 resid_mean.* 向量，被测对参照的差的范数、它与第二参照
          对参照之差的比值、在给定维度上的相对差；probe_lse_mean / probe_logit_mean 的差
            python gauge_analysis.py probe --test <metrics.jsonl> --ref <metrics.jsonl> --pair <metrics.jsonl> \
              [--dims 2007,3881,2146,1567,528,14] [--out json]

数字都在 FP64 里算（输入是 BF16 / FP32 原值）。
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def _load_metrics(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["step"]] = d
    return out


def _named(items):
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        out[name] = path
    return out


# ---------------- arms ----------------
def cmd_arms(a):
    ctl = [_load_metrics(p) for p in a.ctl]
    gold = _load_metrics(a.golden) if a.golden else None
    band = [_load_metrics(p) for p in a.band] if a.band else []

    def ctl_mean(step, key):
        v = [c[step][key] for c in ctl if step in c and key in c[step]]
        return st.mean(v) if v else None

    def rel_loss(d, lo, hi, ref):
        xs = []
        for s in range(lo, hi + 1):
            if s in d and ref(s) is not None:
                r = ref(s)
                xs.append((d[s]["lm_loss"] - r) / r)
        return st.mean(xs) if xs else None

    band_off = [rel_loss(b, 301, 1000, lambda s: gold[s]["lm_loss"]) for b in band] if gold else []
    band_off = [x for x in band_off if x is not None]
    res = {"controls": a.ctl, "golden": a.golden, "band": a.band,
           "band_301_1000": ({"mean": st.mean(band_off), "sd": st.stdev(band_off), "min": min(band_off),
                              "max": max(band_off), "n": len(band_off)} if len(band_off) > 1 else None),
           "runs": {}}
    for name, path in _named(a.run).items():
        d = _load_metrics(path)
        last = max(d)
        r = {"last_step": last, "output_sq_minus_ctl": {}, "param_norm_minus_ctl": {}}
        for s in (100, 200, 300, 1000):
            if s in d and "group_sq.output" in d[s] and ctl_mean(s, "group_sq.output") is not None:
                r["output_sq_minus_ctl"][s] = d[s]["group_sq.output"] - ctl_mean(s, "group_sq.output")
            if s in d and ctl_mean(s, "param_norm") is not None:
                r["param_norm_minus_ctl"][s] = d[s]["param_norm"] - ctl_mean(s, "param_norm")
        r["loss_rel_vs_ctl_101_300"] = rel_loss(d, 101, 300, lambda s: ctl_mean(s, "lm_loss"))
        # 路由偏置标准差第 300 步对对照的 z（对照之间的总体标准差为分母）
        r["router_bias_std_z_300"] = {}
        for k in ("router.bias_std.1", "router.bias_std.2", "router.bias_std.3"):
            v = [c[300][k] for c in ctl if 300 in c and k in c[300]]
            if 300 in d and k in d[300] and len(v) > 1 and st.pstdev(v) > 0:
                r["router_bias_std_z_300"][k] = (d[300][k] - st.mean(v)) / st.pstdev(v)
        if last >= 1000 and gold:
            o = rel_loss(d, 301, 1000, lambda s: gold[s]["lm_loss"])
            r["loss_rel_vs_g1k_a_301_1000"] = o
            if res["band_301_1000"]:
                r["z_vs_band"] = (o - res["band_301_1000"]["mean"]) / res["band_301_1000"]["sd"]
            r["param_norm_minus_g1k_a_1000"] = d[1000]["param_norm"] - gold[1000]["param_norm"]
        res["runs"][name] = r
    return res


# ---------------- step1 ----------------
def cmd_step1(a):
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    key = "grad.output_layer.weight"

    def get(p):
        with safe_open(p, "pt") as s:
            return s.get_tensor(key).double()

    T, A = get(a.test), get(a.ref)
    V = T.shape[0]
    seen = np.zeros(V, bool)
    bdir = Path(a.batches)
    for r in range(8):
        b = np.memmap(bdir / f"rank{r}.bin", dtype=np.uint8, mode="r")
        for line in open(bdir / f"rank{r}.jsonl"):
            e = json.loads(line)
            if e["step"] != 1:
                continue
            f = e["fields"]["labels"]
            n = int(np.prod(f["shape"]))
            seen[np.frombuffer(b[f["offset"]:f["offset"] + 8 * n].tobytes(), dtype=np.int64)] = True
    un = torch.from_numpy(~seen)
    cos = torch.nn.functional.cosine_similarity
    h = load_file(a.hidden)["final_norm.out"].double().reshape(-1, T.shape[1])
    hbar = h.mean(0)
    rT, rA = T.sum(0), A.sum(0)
    res = {"rows": V, "unseen_rows_step1": int(un.sum()),
           "row_sum_norm": {"test": float(rT.norm()), "ref": float(rA.norm())},
           "row_sum_cos_hbar": {"test": float(cos(rT, hbar, dim=0)), "ref": float(cos(rA, hbar, dim=0)),
                                "test_minus_ref": float(cos(rT - rA, hbar, dim=0))},
           "row_sum_diff_norm": float((rT - rA).norm()),
           "row_sum_diff_random_expect": float((T - A).norm())}
    D = (T - A)[un]
    md = D.mean(0)
    mA = A[un].mean(0)
    res["unseen_common_mode"] = {"mean_diff_norm": float(md.norm()), "random_expect": float(D.norm() / D.shape[0]),
                                 "cos_with_mean_ref_grad": float(cos(md, mA, dim=0)),
                                 "cos_with_hbar": float(cos(md, hbar, dim=0)),
                                 "rel_to_mean_ref_grad": float(md.norm() / mA.norm())}
    # 用一个 microbatch 的 logits 重算「(p − y)/N 舍 BF16」带来的跨行和方向
    L = load_file(a.logits)["logits"].float().reshape(h.shape[0], -1)
    b = np.memmap(bdir / "rank0.bin", dtype=np.uint8, mode="r")
    lab = None
    for line in open(bdir / "rank0.jsonl"):
        e = json.loads(line)
        if e["step"] == 1 and e["micro"] == 0:
            f = e["fields"]["labels"]
            lab = torch.from_numpy(np.frombuffer(b[f["offset"]:f["offset"] + 8 * h.shape[0]].tobytes(),
                                                 dtype=np.int64).copy())
            break
    p = torch.softmax(L.double(), -1)
    p[torch.arange(h.shape[0]), lab] -= 1.0
    n_glob = float(a.global_tokens)
    s_bf16 = (p / n_glob).float().bfloat16().double().sum(-1) * n_glob
    py = p[torch.arange(h.shape[0]), lab] + 1.0
    r_bf = (s_bf16[:, None] * h).mean(0)
    res["bf16_rounding_of_dlogits"] = {
        "per_token_sum_mean": float(s_bf16.mean()), "per_token_sum_sd": float(s_bf16.std()),
        "frac_p_target_below_2^-9": float((py < 2 ** -9).double().mean()),
        "implied_row_sum_cos_ref": float(cos(r_bf, rA, dim=0)), "implied_row_sum_cos_test": float(cos(r_bf, rT, dim=0)),
        "note": "单个 microbatch（rank 0 第一个）重算，方向可比，大小只作量级参考"}
    return res


# ---------------- logits ----------------
def cmd_logits(a):
    import numpy as np
    import torch
    from safetensors.torch import load_file

    counts = np.load(a.counts)[a.key]
    un = torch.from_numpy(counts == 0)
    fr = torch.from_numpy(counts > 10000)
    dims = [int(x) for x in a.dims.split(",")]
    res = {"counts": a.counts, "key": a.key, "unseen_tokens": int(un.sum()), "runs": {}}
    for name, d in _named(a.run).items():
        x = load_file(f"{d}/logits.safetensors")["logits"].double()
        x = x.reshape(-1, x.shape[-1])
        lse = torch.logsumexp(x, -1, keepdim=True)
        h = load_file(f"{d}/final_norm.safetensors")["final_norm.out"].double()
        h = h.reshape(x.shape[0], -1)
        hb = h.mean(0)
        res["runs"][name] = {
            "lse_mean": float(lse.mean()), "raw_logit_mean": float(x.mean()),
            "raw_logit_mean_unseen": float(x[:, un].mean()), "raw_logit_mean_frequent": float(x[:, fr].mean()),
            "logp_unseen_mean": float((x[:, un] - lse).mean()),
            "p_mass_unseen": float(torch.exp(x[:, un] - lse).sum(-1).mean()),
            "hbar_norm": float(hb.norm()), "hbar_dims": {str(k): float(hb[k]) for k in dims}}
    return res


# ---------------- rows ----------------
def cmd_rows(a):
    import numpy as np

    def rows(path):
        out = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if "output_row_wu" in d:
                    out[d["step"]] = {k: np.asarray(d[k], np.float64) for k in ("output_row_wu", "output_row_g", "output_row_u")
                                      if k in d}
        return out

    c = np.load(a.counts)[a.key]
    buckets = [("没出现过", c == 0), ("1–10 次", (c >= 1) & (c <= 10)), ("11–100", (c > 10) & (c <= 100)),
               ("101–1000", (c > 100) & (c <= 1000)), (">1000", c > 1000)]
    sides = {"test": rows(a.test), "ref": rows(a.ref)}
    if a.pair:
        sides["pair"] = rows(a.pair)
    res = {"counts": a.counts, "key": a.key, "steps": {}}
    for s in [int(x) for x in a.steps.split(",")]:
        if not all(s in v for v in sides.values()):
            continue
        res["steps"][s] = [{"bucket": nm, "rows": int(m.sum()),
                            **{f"sum_wu_{k}": float(v[s]["output_row_wu"][m].sum()) for k, v in sides.items()},
                            **{f"mean_sqrt_v_{k}": float(v[s]["output_row_g"][m].mean()) for k, v in sides.items()},
                            **{f"mean_u_{k}": float(v[s]["output_row_u"][m].mean()) for k, v in sides.items()}}
                           for nm, m in buckets]
    return res


# ---------------- probe ----------------
def cmd_probe(a):
    import numpy as np

    def probes(path):
        out = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if "probe_lse_mean" in d:
                    out[d["step"]] = {k: (np.asarray(v, np.float64) if isinstance(v, list) else v) for k, v in d.items()
                                      if k.startswith("resid_mean.") or k.startswith("probe_") or k == "final_norm_scale"}
        return out

    T, A, B = probes(a.test), probes(a.ref), probes(a.pair)
    dims = [int(x) for x in a.dims.split(",")]
    steps = sorted(set(T) & set(A) & set(B))
    keys = sorted(k for k in T[steps[0]] if k.startswith("resid_mean.")) if steps else []
    res = {"steps": steps, "dims": dims, "by_step": {}}
    for s in steps:
        row = {"lse_diff": T[s]["probe_lse_mean"] - A[s]["probe_lse_mean"],
               "lse_pair_diff": B[s]["probe_lse_mean"] - A[s]["probe_lse_mean"],
               "logit_mean_diff": T[s]["probe_logit_mean"] - A[s]["probe_logit_mean"],
               "logit_mean_pair_diff": B[s]["probe_logit_mean"] - A[s]["probe_logit_mean"], "resid": {}}
        for k in keys:
            if k not in A[s] or k not in B[s]:
                continue
            t, r, q = T[s][k], A[s][k], B[s][k]
            dt, dq = float(np.linalg.norm(t - r)), float(np.linalg.norm(q - r))
            row["resid"][k] = {"diff_norm": dt, "pair_diff_norm": dq, "ratio": dt / dq if dq > 0 else None,
                               "ref_norm": float(np.linalg.norm(r)),
                               "dims_rel": {str(i): float((t[i] - r[i]) / r[i]) if r[i] != 0 else None for i in dims},
                               "dims_rel_pair": {str(i): float((q[i] - r[i]) / r[i]) if r[i] != 0 else None for i in dims}}
        res["by_step"][s] = row
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arms")
    p.add_argument("--ctl", nargs="+", required=True)
    p.add_argument("--band", nargs="*", default=[])
    p.add_argument("--golden", default=None)
    p.add_argument("--run", nargs="+", required=True)
    p = sub.add_parser("step1")
    for k in ("--test", "--ref", "--logits", "--hidden", "--batches"):
        p.add_argument(k, required=True)
    p.add_argument("--global-tokens", type=int, default=65536)
    p = sub.add_parser("rows")
    p.add_argument("--test", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--pair", default=None)
    p.add_argument("--counts", required=True)
    p.add_argument("--key", default="c300")
    p.add_argument("--steps", default="25,50,100,200,300,400")
    p = sub.add_parser("probe")
    p.add_argument("--test", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--dims", default="2007,3881,2146,1567,528,14")
    p = sub.add_parser("logits")
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--counts", required=True)
    p.add_argument("--key", default="c1000")
    p.add_argument("--dims", default="2007,3881,2146,1567,528,14")
    for q in sub.choices.values():
        q.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {"arms": cmd_arms, "step1": cmd_step1, "logits": cmd_logits, "rows": cmd_rows, "probe": cmd_probe}[a.cmd](a)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
