"""路由专家输出的 FP64 参照（案例 01）：同一状态、同一 batch 的两份前向抓数，各自拿自己的输入在 FP64 里重算路由专家输出再比。

用来回答「两侧号称全程 FP32 的程序，路由专家那一段到底是不是 FP32」。只取两侧 top-k 选中的专家集合相同的 token
（路由翻转的 token 本来就不能比），每个 token 用 GPU 抓到的路由概率（[token, 专家数]，已含缩放）加权：
    out_t = Σ_{e ∈ topk(t)} p[t, e] · fc2_e( silu(fc1_e x_t)[:F] ⊙ (fc1_e x_t)[F:] )
x_t 用各侧自己的 mlp_norm.out（这样比的是专家这一段本身，不混进上游的差）。另报「输入与权重全部先舍到 BF16 再算」的误差量级当对照。

  python moe_fp64_ref.py --params <整套参数目录（param_swap 格式）> --layer 1 \\
      --tpu <TPU 运行目录>@<步> --gpu <GPU 运行目录>@<步> [--tokens 48] [--out <json>]

GPU 侧路由专家输出 = mlp.out − mlp.shared.out；TPU 侧直接用 mlp.routed.out。权重按 Megatron 布局读：
rank r 的文件存全局专家 24r…24r+23（本地名 weight<k>）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from janus.store import Run

EXPERTS_PER_RANK = 24


def _spec(x: str):
    r, s = x.rsplit("@", 1)
    return Run.open(r), int(s)


def _bf16(a):
    import ml_dtypes

    return a.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)


def _silu(z):
    return z / (1 + np.exp(-z))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", required=True)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--tpu", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    from safetensors import safe_open

    (T, ts), (G, gs) = _spec(a.tpu), _spec(a.gpu)
    L = a.layer
    f64 = lambda R, s, n: np.asarray(R.tensor(s, n)).astype(np.float64)
    ti = np.asarray(T.tensor(ts, f"layers.{L}.mlp.router.topk_idx"))
    gi = np.asarray(G.tensor(gs, f"layers.{L}.mlp.router.topk_idx"))
    k = ti.shape[-1]
    ti, gi = ti.reshape(-1, k), gi.reshape(-1, k)
    n = ti.shape[0]
    same = np.all(np.sort(ti, 1) == np.sort(gi, 1), 1)
    probs = f64(G, gs, f"layers.{L}.mlp.router.probs").reshape(n, -1)
    xg = f64(G, gs, f"layers.{L}.mlp_norm.out").reshape(n, -1)
    xt = f64(T, ts, f"layers.{L}.mlp_norm.out").reshape(n, -1)
    rg = (f64(G, gs, f"layers.{L}.mlp.out") - f64(G, gs, f"layers.{L}.mlp.shared.out")).reshape(n, -1)
    rt = f64(T, ts, f"layers.{L}.mlp.routed.out").reshape(n, -1)
    toks = np.random.default_rng(a.seed).choice(np.where(same)[0], min(a.tokens, int(same.sum())), replace=False)

    files, cache = {}, {}

    def W(e):
        if e not in cache:
            r = e // EXPERTS_PER_RANK
            if r not in files:
                files[r] = safe_open(str(Path(a.params) / f"rank{r}.safetensors"), "np")
            pre = f"decoder.layers.{L}.mlp.experts.linear_fc"
            cache[e] = tuple(
                files[r].get_tensor(f"{pre}{i}.weight{e % EXPERTS_PER_RANK}").astype(np.float64) for i in (1, 2)
            )
        return cache[e]

    out = {key: [] for key in ("ref_g", "ref_t", "ref_g_bf16", "g", "t")}
    for t in toks:
        for key, x in (("ref_g", xg[t]), ("ref_t", xt[t])):
            acc = np.zeros(x.shape[0])
            accb = np.zeros(x.shape[0])
            for e in gi[t]:
                f1, f2 = W(int(e))
                F = f1.shape[0] // 2
                w = probs[t, e]
                h = f1 @ x
                acc += w * (f2 @ (_silu(h[:F]) * h[F:]))
                if key == "ref_g":
                    h = _bf16(f1) @ _bf16(x)
                    accb += w * (_bf16(f2) @ _bf16(_silu(h[:F]) * h[F:]))
            out[key].append(acc)
            if key == "ref_g":
                out["ref_g_bf16"].append(accb)
        out["g"].append(rg[t])
        out["t"].append(rt[t])
    A = {key: np.array(v) for key, v in out.items()}
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
    res = {
        "layer": L,
        "tokens": int(len(toks)),
        "same_route_tokens": int(same.sum()),
        "all_tokens": int(n),
        "gpu_vs_fp64": rel(A["g"], A["ref_g"]),
        "tpu_vs_fp64": rel(A["t"], A["ref_t"]),
        "fp64_input_diff": rel(A["ref_t"], A["ref_g"]),
        "all_bf16_inputs_vs_fp64": rel(A["ref_g_bf16"], A["ref_g"]),
        "gpu_scale_fit": float((A["g"] * A["ref_g"]).sum() / (A["ref_g"] ** 2).sum()),
        "tpu_scale_fit": float((A["t"] * A["ref_t"]).sum() / (A["ref_t"] ** 2).sum()),
        "tpu": a.tpu,
        "gpu": a.gpu,
    }
    for key, v in res.items():
        print(f"{key:26s} {v:.4e}" if isinstance(v, float) else f"{key:26s} {v}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
