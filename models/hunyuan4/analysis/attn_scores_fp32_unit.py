"""GPU 参照稀疏注意力 q·k 舍 BF16 与留 FP32 两种算法对 FP64 参照的相对 L2 误差（单测 test_attn_scores_fp32 的数入库）。

本机没有 megatron.core，不能 import hy4_mcore；这里用 ast 只抽出 causal_sparse_mask 与 sparse_sink_attention 两个函数执行，
输入与 tests/models/hunyuan4/test_attn_scores_fp32.py 的 _inputs 逐行相同（seed=0, s=64, h=2, d=32, dv=16, k=16），另加种子 1..9 看稳定性。

python models/hunyuan4/analysis/attn_scores_fp32_unit.py [--seeds 10] [--out models/hunyuan4/alignments/01-mini/results/attn-scores-fp32-unit.json]
"""

import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "models" / "hunyuan4" / "frameworks" / "megatron" / "hy4_mcore.py"
TEST = ROOT / "tests" / "models" / "hunyuan4" / "test_attn_scores_fp32.py"
WANT = ("causal_sparse_mask", "sparse_sink_attention")


def load_funcs():
    text = SRC.read_text()
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
    assert sorted(n.name for n in nodes) == sorted(WANT), [n.name for n in nodes]
    mod = ast.Module(body=nodes, type_ignores=[])
    ns = {"torch": torch}
    exec(compile(mod, str(SRC), "exec"), ns)
    return ns["sparse_sink_attention"], hashlib.sha256(text.encode()).hexdigest()


def _inputs(seed=0, s=64, h=2, d=32, dv=16, k=16):  # 与单测 _inputs 相同
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(s, 1, h, d, generator=g) * 3).to(torch.bfloat16)
    kk = (torch.randn(s, 1, h, d, generator=g) * 3).to(torch.bfloat16)
    v = torch.randn(s, 1, h, dv, generator=g).to(torch.bfloat16)
    topk = torch.stack([torch.randperm(s, generator=g)[:k] for _ in range(s)])[None]
    sinks = torch.zeros(h)
    return q, kk, v, topk, sinks


def one(attn, seed):
    q, k, v, topk, sinks = _inputs(seed)
    scale = q.shape[-1] ** -0.5
    ref = attn(q.double(), k.double(), v.double(), topk, sinks.double(), scale, scores_fp32=False)
    bf = attn(q, k, v, topk, sinks, scale, scores_fp32=False).double()
    fp = attn(q, k, v, topk, sinks, scale, scores_fp32=True).double()

    def err(x):
        return float((x - ref).norm() / ref.norm())

    e_bf, e_fp = err(bf), err(fp)
    return {
        "seed": seed,
        "rel_l2_scores_bf16": e_bf,
        "rel_l2_scores_fp32": e_fp,
        "ratio_fp32_over_bf16": e_fp / e_bf,
        "outputs_differ": not torch.equal(bf, fp),
        "fp32_closer": e_fp < e_bf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--out", default=str(ROOT / "models/hunyuan4/alignments/01-mini/results/attn-scores-fp32-unit.json")
    )
    a = ap.parse_args()
    torch.manual_seed(0)
    attn, sha = load_funcs()
    rows = [one(attn, s) for s in range(a.seeds)]
    ratios = [r["ratio_fp32_over_bf16"] for r in rows]
    out = {
        "what": "GPU 参照 sparse_sink_attention：scores_fp32=False（q·k 先舍 BF16）/True（留 FP32）对 FP64 参照的输出相对 L2 误差",
        "source": str(SRC.relative_to(ROOT)),
        "source_sha256": sha,
        "test": str(TEST.relative_to(ROOT)),
        "test_sha256": hashlib.sha256(TEST.read_bytes()).hexdigest(),
        "functions_extracted_by_ast": list(WANT),
        "inputs": {"s": 64, "h": 2, "d": 32, "dv": 16, "k": 16, "q_k_scale": 3, "dtype": "bfloat16", "sinks": "zeros"},
        "torch": torch.__version__,
        "device": "cpu",
        "seed0_same_as_unit_test": rows[0],
        "per_seed": rows,
        "summary": {
            "n_seeds": len(rows),
            "fp32_closer_all": all(r["fp32_closer"] for r in rows),
            "ratio_min": min(ratios),
            "ratio_median": statistics.median(ratios),
            "ratio_max": max(ratios),
            "rel_l2_bf16_median": statistics.median(r["rel_l2_scores_bf16"] for r in rows),
            "rel_l2_fp32_median": statistics.median(r["rel_l2_scores_fp32"] for r in rows),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out["summary"], indent=1))
    print(json.dumps(rows[0], indent=1))


if __name__ == "__main__":
    main()
