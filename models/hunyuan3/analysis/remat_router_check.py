"""整层重算下路由的逐层核对：重算是否沿用了首次前向的专家、首次前向和不重算的基线差在哪、差异来自 top-k 的顺序。

用法（Pod 里，要一张 GPU：top-k 不排序时的返回顺序是 GPU 实现决定的，CPU 复现不了）：
  CUDA_VISIBLE_DEVICES=0 flock /data/gpu.lock python3 models/hunyuan3/analysis/remat_router_check.py \
      --test <开重算的运行目录> --base <不开重算的运行目录> [--json <输出.json>]

两个运行都要有第 1 步抓数（同一批数据、同一份权重）；被测要开 --recompute（有 <名>.recompute）。输出：
  1. 每个 MoE 层：重算那次的 logits / topk_idx / probs 与首次前向是否逐位相同；
  2. 每个 MoE 层：被测首次前向对基线——logits、probs 逐位与否（不同元素占比、rel_L2）、选中集合不同的 token 数；
  3. 第一个 MoE 层（输入与 logits 两边逐位相同时）：用基线抓下的 logits 与 expert_bias，按 Megatron 的 sigmoid + expert_bias
     分支重算 probs，top-k 分别 sorted=True / False，各与哪一边逐位相同（定位「首次前向不建图 → top-k 不排序 → 归一化求和顺序不同」）。
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

TOPK, SCALE, EXPERTS = 8, 2.826, 192  # 混元 3：top-8、缩放 2.826、192 个专家（GPU meta 的 model_config）


class Run:
    def __init__(self, d):
        self.d = Path(d)
        self.idx = {t["name"]: t for t in json.loads((self.d / "step_1" / "index.json").read_text())["tensors"]}
        self.cache = {}

    def has(self, name):
        return name in self.idx

    def get(self, name):
        f = self.d / "step_1" / self.idx[name]["file"]
        if f not in self.cache:
            self.cache[f] = load_file(str(f))
        return self.cache[f][name]


def bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def diff(a, b):
    out = {"bitwise": same(a, b)}
    if not out["bitwise"] and a.shape == b.shape:
        out["differ_frac"] = float((a != b).float().mean())
        x, y = a.double(), b.double()
        out["rel_l2"] = float((x - y).norm() / y.norm()) if y.norm() > 0 else None
    return out


def sigmoid_probs(logits, bias, sorted_):
    """Megatron-Core moe_utils.topk_routing_with_score_function 的 sigmoid + expert_bias 分支（非融合路径）。"""
    s = torch.sigmoid(logits.float())
    _, idx = torch.topk(s + bias.float(), k=TOPK, dim=1, sorted=sorted_)
    sc = torch.gather(s, 1, idx)
    p = (sc / (sc.sum(dim=-1, keepdim=True) + 1e-20) * SCALE).type_as(logits)
    return torch.zeros_like(logits).scatter(1, idx, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    T, B = Run(a.test), Run(a.base)
    layers = sorted({int(n.split(".")[1]) for n in T.idx if n.endswith(".mlp.router.logits")})
    res = {"test": a.test, "base": a.base, "layers": {}}
    for L in layers:
        p = f"layers.{L}.mlp.router"
        r = {
            "recompute_vs_first": {
                n: (same(T.get(f"{p}.{n}.recompute"), T.get(f"{p}.{n}")) if T.has(f"{p}.{n}.recompute") else None)
                for n in ("logits", "topk_idx", "probs")
            }
        }
        r["layer_in_vs_base"] = same(T.get(f"layers.{L}.in"), B.get(f"layers.{L}.in"))
        r["logits_vs_base"] = diff(T.get(f"{p}.logits"), B.get(f"{p}.logits"))
        r["probs_vs_base"] = diff(T.get(f"{p}.probs"), B.get(f"{p}.probs"))
        ti, bi = T.get(f"{p}.topk_idx"), B.get(f"{p}.topk_idx")  # 抓数时已按行排序，比的是集合
        r["topk_set_rows_differ"] = int((ti != bi).any(dim=1).sum())
        r["tokens"] = int(ti.shape[0])
        res["layers"][str(L)] = r
    L0 = layers[0]
    if res["layers"][str(L0)]["logits_vs_base"]["bitwise"]:
        logits = B.get(f"layers.{L0}.mlp.router.logits").cuda().view(-1, EXPERTS)
        bias = B.get(f"buffer.decoder.layers.{L0}.mlp.router.expert_bias").cuda()
        res["topk_order_repro"] = {
            "layer": L0,
            **{
                f"{who}_matches_sorted_{so}": same(
                    sigmoid_probs(logits, bias, so).cpu(), run.get(f"layers.{L0}.mlp.router.probs")
                )
                for who, run in (("base", B), ("test_first_fwd", T))
                for so in (True, False)
            },
        }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
