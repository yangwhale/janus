"""v5p 上 L1 单算子回放的正式判定：layer_replay.py --l1 存下的 TPU 算子输出（l1.<层>.<观测点>，输入都是 golden 的）
对 FP64 真值、对 GPU（golden 抓下的同名输出）、对主机上复现的舍入方式比，写成与 `janus run` 同一格式的结果文件（op_replay）。

| 检查 | 输入（golden） | 判据 |
|---|---|---|
| L1.ops.router.gemm | mlp_norm.out；核 = 检查点的路由权重（第 1 步 = HF 的 BF16 值，L0 已逐位） | R-ratio ≤ 1.2（被测离 FP64 不比 GPU 远） |
| L1.ops.router.topk | 被测 logits 过 sigmoid、加 golden 的 expert_bias 取 top-8 | 与 GPU 选中的集合逐行相同（exact） |
| L1.ops.router.precision | 不能用 BF16 表示的探针核 | DEFAULT 精度的结果与「两个操作数先舍 BF16、单遍乘、FP32 累加」逐位相同（exact） |
| L1.ops.rmsnorm | bf16(层输入 + attn.out)（两边的残差加都是 BF16） | R-ratio ≤ 1.2 |
| L1.ops.act | 第 0 层 fc1.out 切成 gate、up | 与主机上的 GPU 写法 bf16(bf16(silu(gate))·up) 逐位相同（exact） |
| L1.ops.attn.core | q_norm.out、k_norm.out（RoPE 之前）、qkv.out 里的 v | R-ratio ≤ 1.2（被测含 RoPE、1/√d 缩放、splash 注意力） |
| L1.ops.moe | mlp_norm.out | R-ratio ≤ 1.2（被测含路由、分组 GEMM、合并、共享专家；真值按 FP64 分数重新选专家） |

R-ratio 门槛 1.2 是推断，和 GPU 侧 phase1-l1 的 L1.rratio 同一个起点；exact 的三条不需要门槛。
真值由 fp64_forward.py 的同一套函数算（全程 FP64，中间不舍入）。

  python l1_replay_results.py --hf <HF 目录> --golden <运行根>/g1k-a --test <运行根>/r-l1-ops --out <结果目录>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import fp64_forward as F  # noqa: E402
import fwd_rratio as R  # noqa: E402

from janus.checks import REGISTRY, Threshold  # noqa: E402
from janus.store import Run  # noqa: E402


def bf16():
    import ml_dtypes
    return ml_dtypes.bfloat16


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="phase2-l1-ops")
    a = ap.parse_args(argv)
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    hf, g, t = F.HF(a.hf), Run.open(a.golden), Run.open(a.test)
    c = hf.cfg
    eps, theta, H, KV, d = c["rms_norm_eps"], c["rope_parameters"]["rope_theta"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    op = REGISTRY["op_replay"]
    rr = Threshold("numeric", values={"r_ratio": R.R_MAX}, source=R.R_SOURCE)
    ex = Threshold("exact")
    tn = set(t.names(1))
    S = t.tensor(1, next(n for n in sorted(tn) if n.endswith("attn.core.out"))).shape[0]
    pos = g.tensor(1, "input.position_ids")[0, :S]
    cases = {k: [] for k in ("gemm", "topk", "precision", "rmsnorm", "act", "attn", "moe")}
    for i in range(c["num_hidden_layers"]):
        L, p = f"layers.{i}", f"model.layers.{i}."
        moe = i >= c["first_k_dense_replace"]
        if moe and f"l1.{L}.mlp.router.logits" in tn:
            x = g.tensor(1, f"{L}.mlp_norm.out")[:S, 0]
            w = hf.get(p + "mlp.router.gate.weight")
            lg = t.tensor(1, f"l1.{L}.mlp.router.logits")
            cases["gemm"].append({"name": f"{L}.mlp.router.logits", "out": lg, "fp64": x.astype(np.float64) @ w.T,
                                  "ref": g.tensor(1, f"{L}.mlp.router.logits")[:S, 0]})
            b = g.tensor(1, f"buffer.decoder.{L}.mlp.router.expert_bias").astype(np.float64)
            sel = F.topk_sets(1.0 / (1.0 + np.exp(-lg.astype(np.float64))) + b, c["num_experts_per_tok"]).astype(np.int32)
            cases["topk"].append({"name": f"{L}.mlp.router.topk_idx", "out": sel, "expect": g.tensor(1, f"{L}.mlp.router.topk_idx")[:S]})
            cases["precision"].append({"name": f"{L}.router.probe", "out": t.tensor(1, f"l1.{L}.router.probe.default"),
                                       "bitref": t.tensor(1, f"l1.{L}.router.probe.bf16_operands"),
                                       "extra": {"kernel": "BF16 核 × (1 + 2^-10·N(0,1))，FP32，不能用 BF16 表示"}})
        if moe and f"l1.{L}.mlp_norm.out" in tn:
            res = (g.tensor(1, f"{L}.in")[:S, 0].astype(np.float32) + g.tensor(1, f"{L}.attn.out")[:S, 0].astype(np.float32)).astype(bf16())
            cases["rmsnorm"].append({"name": f"{L}.mlp_norm.out", "out": t.tensor(1, f"l1.{L}.mlp_norm.out"),
                                     "fp64": F.rmsnorm(res, hf.get(p + "post_attention_layernorm.weight"), eps),
                                     "ref": g.tensor(1, f"{L}.mlp_norm.out")[:S, 0]})
        if not moe and f"l1.{L}.mlp.act.out" in tn:
            fc1 = g.tensor(1, f"{L}.mlp.fc1.out")[:S, 0]
            half = fc1.shape[-1] // 2
            gate, up = fc1[:, :half], fc1[:, half:]
            s64 = F.silu(gate.astype(np.float64))
            gpu = (s64.astype(np.float32).astype(bf16()).astype(np.float32) * up.astype(np.float32)).astype(bf16())
            cases["act"].append({"name": f"{L}.mlp.act.out", "out": t.tensor(1, f"l1.{L}.mlp.act.out"), "emulate": gpu,
                                 "fp64": s64 * up.astype(np.float64)})
        if f"l1.{L}.attn.core.out" in tn:
            qn = g.tensor(1, f"{L}.attn.q_norm.out")[:S, 0].astype(np.float64)
            kn = g.tensor(1, f"{L}.attn.k_norm.out")[:S, 0].astype(np.float64)
            v = g.tensor(1, f"{L}.attn.qkv.out")[:S, 0].reshape(S, KV, (H // KV + 2) * d)[:, :, (H // KV + 1) * d:].astype(np.float64)
            f64 = F.attention(F.rope(qn, pos, theta, d), F.rope(kn, pos, theta, d), v, H, KV)
            cases["attn"].append({"name": f"{L}.attn.core.out", "out": t.tensor(1, f"l1.{L}.attn.core.out"), "fp64": f64,
                                  "ref": g.tensor(1, f"{L}.attn.core.out")[:S, 0]})
            print(f"[attn {i}] 真值已算", flush=True)
        if moe and f"l1.{L}.mlp.out" in tn:
            m = F.moe_forward(hf, i, g.tensor(1, f"{L}.mlp_norm.out")[:S, 0])
            cases["moe"].append({"name": f"{L}.mlp.out", "out": t.tensor(1, f"l1.{L}.mlp.out"), "fp64": m["mlp.out"][:, 0],
                                 "ref": g.tensor(1, f"{L}.mlp.out")[:S, 0]})
            print(f"[moe {i}] 真值已算", flush=True)
    spec = [("L1.ops.router.gemm", "gemm", rr), ("L1.ops.router.topk", "topk", ex), ("L1.ops.router.precision", "precision", ex),
            ("L1.ops.rmsnorm", "rmsnorm", rr), ("L1.ops.act", "act", ex), ("L1.ops.attn.core", "attn", rr), ("L1.ops.moe", "moe", rr)]
    checks = []
    for cid, key, th in spec:
        r = op(cid, "L1", cases[key], th, params={"worst_by": ["r_ratio"]} if th is rr else None)
        checks.append(r)
        print(f"[{cid}] {r.status} {r.metrics}", flush=True)
    env = {"hf_config_sha256": hashlib.sha256((Path(a.hf) / "config.json").read_bytes()).hexdigest(), "numpy": np.__version__,
           "reference": "models/hunyuan3/analysis/fp64_forward.py（FP64，中间不舍入）"}
    return R.write(Path(a.out), checks, [t, g], a.stage, Path(__file__).resolve(), started, env, t.name, g.name)


if __name__ == "__main__":
    sys.exit(main())
