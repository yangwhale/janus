"""把一份 janus 全量抓数（JanusDump，Megatron-Core 名字）里第 N 步的梯度换成 HF checkpoint 的名字与布局，FP32。

梯度是 `grad.<Megatron 名>`：JanusDump 在 `opt.step()` 之前抓的 main_grad（FP32），即 DP 归约、按全局有效 token 数
缩放之后、裁剪之前，和裁剪看到的是同一份。名字与布局用载入检查点的同一张对照表反过来（`hy4_weights.py`：
GLOBAL / LAYER / ATTN / INDEXER / MOE_FIXED / MTP_EXTRA / layer_plan / mtp_names），所以与 `tensor_sha256.json`
里的名字逐个对得上：
- 稠密层与共享专家的 fc1 = cat(gate_proj, up_proj)，沿第 0 维拆回两半；
- 路由专家：Megatron 的 linear_fc1.weight{e} / linear_fc2.weight{e} 就是 HF 的 gate_up_proj[e] / down_proj[e]，
  按专家号叠回 [E_dump, 2I, D] / [E_dump, D, I]；
- 其余原样（MLA 的 q / kv 上投影、hc_fn 等载入时也是原样拷贝，列序即 HF 的列序）。

全量抓数只在 rank 0 上做：非专家梯度是完整的（DP 归约后各 rank 相同）；路由专家只有 rank 0 本地的那几个
（EP4 时每层 16 个里的 0–3 号）。旁边写一份 JSON：每个张量的形状、FP64 平方和、专家范围、来源。

    python export_hf_grads.py --dump <运行目录> --hf-config <config.json> --step 1 --out step_1_hf_grads.safetensors
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
from types import SimpleNamespace

import numpy as np


def mcore_to_hf(cfg) -> dict:
    """Megatron 名 → (HF 名, 变换)。变换：None 原样；('cat', [名, 名]) 拆两半；('experts', 名) 叠专家。"""
    from hy4_weights import GLOBAL, layer_plan, mtp_names

    out = {m: (h, None) for m, h in GLOBAL}
    for i in range(cfg.num_layers):
        pm, ph = f"decoder.layers.{i}.", f"model.layers.{i}."
        for m, h, tf in layer_plan(cfg, i):
            out[pm + m] = (tuple(ph + x for x in h), "cat") if tf == "cat" else (ph + h, None)
        if cfg.moe_layer_freq[i]:
            out[pm + "mlp.experts.linear_fc1"] = (ph + "mlp.experts.gate_up_proj", "experts")
            out[pm + "mlp.experts.linear_fc2"] = (ph + "mlp.experts.down_proj", "experts")
    if cfg.mtp_num_layers:
        for m, h, tf in mtp_names(cfg):
            if tf == "cat":
                out[m] = (tuple(h), "cat")
            elif tf == "experts":
                out[m + "mlp.experts.linear_fc1"] = (h + "mlp.experts.gate_up_proj", "experts")
                out[m + "mlp.experts.linear_fc2"] = (h + "mlp.experts.down_proj", "experts")
            else:
                out[m] = (h, None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="janus 运行目录（含 step_<N>/grad.*.safetensors）")
    p.add_argument("--hf-config", required=True)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--kind",
        default="grad",
        choices=["grad", "main_param", "main_param_after", "exp_avg", "exp_avg_sq"],
        help="换哪一类张量：梯度（默认）、更新前 / 更新后的 FP32 主权重、Adam 的两个动量（只有走 Adam 的参数有；同一张名字对照表）",
    )
    a = p.parse_args()

    from safetensors import safe_open

    hf = json.load(open(a.hf_config))
    L = hf["num_hidden_layers"]
    cfg = SimpleNamespace(
        num_layers=L,
        hy4_indexer_types=hf["indexer_types"][:L],
        moe_layer_freq=[int(t == "sparse") for t in hf["mlp_layer_types"][:L]],
        mtp_num_layers=int(hf.get("num_nextn_predict_layers", 0)),
        num_moe_experts=hf["n_routed_experts"],
    )
    table = mcore_to_hf(cfg)
    # 只记每个张量在哪个文件、哪个键，写出时才逐个读：T2 一层路由专家的 FP32 主权重约 43 GB，整份读进内存再拼、
    # 再整体序列化会超过 100 GB；逐个读、逐个写，峰值约等于最大的一个输出张量（叠好的 gate_up_proj）。
    src = {}
    for f in sorted(glob.glob(os.path.join(a.dump, f"step_{a.step}", f"{a.kind}.*.safetensors"))):
        with safe_open(f, "np") as h:
            for k in h.keys():
                if k.startswith(a.kind + "."):
                    src[k[len(a.kind) + 1 :]] = (f, k, list(h.get_slice(k).get_shape()))

    plan, unmapped, experts = {}, [], {}  # HF 名 → (取法, 输出形状)
    for name, (f, k, shape) in src.items():
        m = re.match(r"^(.*mlp\.experts\.linear_fc[12])\.weight(\d+)$", name)
        if m:
            experts.setdefault(m.group(1), {})[int(m.group(2))] = (f, k, shape)
            continue
        if name not in table:
            unmapped.append(name)
            continue
        h, tf = table[name]
        if tf == "cat":
            half = shape[0] // 2
            assert shape[0] == 2 * half, (name, shape)
            plan[h[0]] = (("half", f, k, 0, half), [half, *shape[1:]])
            plan[h[1]] = (("half", f, k, 1, half), [half, *shape[1:]])
        else:
            plan[h] = (("one", f, k), shape)
    side = {}
    for base, by_e in experts.items():
        h, tf = table[base]
        assert tf == "experts", base
        ids = sorted(by_e)
        shapes = {tuple(by_e[e][2]) for e in ids}
        assert len(shapes) == 1, (base, shapes)
        plan[h] = (("experts", [by_e[e][:2] for e in ids]), [len(ids), *by_e[ids[0]][2]])
        side[h] = {"experts": ids}
    assert not unmapped, f"对照表里没有：{unmapped[:10]}"

    def load(how):
        def get(f, k):
            with safe_open(f, "np") as hh:
                return np.asarray(hh.get_tensor(k), dtype=np.float32)

        if how[0] == "one":
            return get(how[1], how[2])
        if how[0] == "half":
            g = get(how[1], how[2])
            return np.ascontiguousarray(g[: how[4]] if how[3] == 0 else g[how[4] :])
        parts = how[1]
        first = get(*parts[0])
        out = np.empty((len(parts), *first.shape), np.float32)
        out[0] = first
        del first
        for i, (f, k) in enumerate(parts[1:], 1):
            out[i] = get(f, k)
        return out

    def sumsq64(g):  # 按第 0 维分块转 FP64，免得整份 FP64 副本
        flat = g.reshape(g.shape[0], -1) if g.ndim > 1 else g.reshape(1, -1)
        step = max(1, (1 << 26) // max(1, flat.shape[1]))
        return float(sum(np.square(flat[i : i + step].astype(np.float64)).sum() for i in range(0, flat.shape[0], step)))

    names = sorted(plan)
    header, off = {}, 0
    for n in names:
        nbytes = 4 * int(np.prod(plan[n][1], dtype=np.int64))
        header[n] = {"dtype": "F32", "shape": plan[n][1], "data_offsets": [off, off + nbytes]}
        off += nbytes
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(a.out, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for n in names:
            g = load(plan[n][0])
            assert list(g.shape) == plan[n][1], (n, g.shape, plan[n][1])
            side.setdefault(n, {}).update({"shape": list(g.shape), "sumsq_fp64": sumsq64(g)})
            fo.write(memoryview(np.ascontiguousarray(g)).cast("B"))
            del g
    out = plan
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    summary = {
        "source_run": meta.get("run_name"),
        "step": a.step,
        "what": "main_grad FP32 before opt.step (after DP reduction and ÷ global valid tokens, before clipping)",
        "names": "HF checkpoint names (same as tensor_sha256.json)",
        "rank": 0,
        "routed_experts": "the experts present in the dump (listed per tensor under 'experts'): rank 0's local ones, or all of them when dumped with gather_experts; all non-expert grads complete",
        "n_tensors": len(out),
        "sumsq_fp64_total_exported": float(sum(v["sumsq_fp64"] for v in side.values())),
        "tensors": side,
    }
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print("EXPORTED", len(out), "tensors", summary["sumsq_fp64_total_exported"])


if __name__ == "__main__":
    main()
