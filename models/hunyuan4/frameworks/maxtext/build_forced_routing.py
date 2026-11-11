"""把 GPU 抓到的路由选择（每行一个 safetensors：各 MoE 层与 MTP 层的 mlp.gate.routing_map / probs，加 input.tokens）
整理成 MaxText 路由回放（开关 debug_forced_routing_deepseek）用的一个小文件，供 train_hy4.py --forced-routing 读。

输出（safetensors）：
- forced_routed_experts：int32 [行, 序列, 主干层数 + MTP 层数, top-k]，第 L 个槽位 = HF 第 L 层（MTP 层排在主干层之后）；
  稠密层的槽位填 -1（MaxText 不读）。同一 token 的 k 个专家按 GPU 的路由权重（probs）降序排列
  （TPU 自己选时按加偏置后的分数排；偏置比分数小两个量级，个别 token 顺序可能不同，只影响 k 项求和的先后）；
- gpu_weights：float32，同形状，GPU 的路由权重（已归一并乘 routed_scaling_factor），留给「权重也灌」的变体；
- tokens：int32 [行, 序列]，train_hy4.py 用它把每一行对上本侧 batch 的行（逐 token 相等）；
- 给了 --indexer 时另写 forced_indexer_topk：int16 [行, 序列, 主干层数 + MTP 层数, indexer_topk]，各层
  self_attn.topk（DSA 索引器选中的 key 位置，-1 = 空位）；供 MaxText 开关 debug_forced_indexer_topk
  （train_hy4.py --forced-indexer），只有自带索引器的层的槽位会被读到。
核对：每个 token 正好 k 个专家；routing_map 与 probs 的非零位置一致（probs 只在选中的专家上非零，所以文件里
看不出 GPU 加偏置前的分数，也就核不了「是前 k 名」），不符直接报错。

  python build_forced_routing.py --choices <目录，含 step_1_row_*.safetensors> --out forced_routing_step1.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def _layers(keys) -> tuple[list[int], list[int]]:
    main = sorted(
        {int(m.group(1)) for k in keys if (m := re.match(r"hf\.model\.layers\.(\d+)\.mlp\.gate\.routing_map$", k))}
    )
    mtp = sorted(
        {int(m.group(1)) for k in keys if (m := re.match(r"hf\.model\.mtp_layers\.(\d+)\.mlp\.gate\.routing_map$", k))}
    )
    return main, mtp


def build(
    files: list[Path], num_decoder_layers: int | None = None, indexer: bool = False
) -> tuple[dict[str, np.ndarray], dict]:
    from janus.store import read_safetensors_header, read_safetensors_tensor

    rows_ids, rows_w, toks, srcs, rows_idx = [], [], [], [], []
    main = mtp = None
    k = None
    for f in files:
        header, _ = read_safetensors_header(f)
        keys = [x for x in header if x != "__metadata__"]

        def get(name, f=f):
            return read_safetensors_tensor(f, name)[0]

        m_, t_ = _layers(keys)
        if main is None:
            main, mtp = m_, t_
            n_main = num_decoder_layers if num_decoder_layers is not None else max(main) + 1
        elif (m_, t_) != (main, mtp):
            raise SystemExit(f"{f.name}：层集合与前面的文件不同")
        tok = get("input.tokens").reshape(-1)
        S = tok.shape[0]
        w = None
        for slot, name in [(L, f"layers.{L}") for L in main] + [(n_main + j, f"mtp_layers.{j}") for j in mtp]:
            rm = get(f"hf.model.{name}.mlp.gate.routing_map").astype(bool)
            p = get(f"hf.model.{name}.mlp.gate.probs").astype(np.float32)
            cnt = rm.sum(1)
            if k is None:
                k = int(cnt[0])
            if not np.all(cnt == k):
                raise SystemExit(f"{f.name} {name}：每个 token 的专家数不全是 {k}")
            if not np.array_equal(rm, p != 0):
                raise SystemExit(f"{f.name} {name}：routing_map 与 probs 的非零位置不一致")
            if w is None:
                ids = np.full((S, n_main + len(mtp), k), -1, np.int32)
                w = np.zeros((S, n_main + len(mtp), k), np.float32)
            order = np.argsort(-np.where(rm, p, -np.inf), axis=1, kind="stable")[:, :k]
            ids[:, slot] = order
            w[:, slot] = np.take_along_axis(p, order, axis=1)
        if indexer:
            topk = None
            for slot, name in [(L, f"layers.{L}") for L in range(n_main)] + [
                (n_main + j, f"mtp_layers.{j}") for j in mtp
            ]:
                key = f"hf.model.{name}.self_attn.topk"
                if key not in header:
                    continue
                t = get(key).reshape(S, -1)
                if t.max() >= 2**15:
                    raise SystemExit(f"{f.name} {name}：top-k 位置超出 int16")
                if topk is None:
                    topk = np.full((S, n_main + len(mtp), t.shape[1]), -1, np.int16)
                topk[:, slot] = t.astype(np.int16)
            if topk is None:
                raise SystemExit(f"{f.name}：没有 self_attn.topk")
            rows_idx.append(topk)
        rows_ids.append(ids)
        rows_w.append(w)
        toks.append(tok.astype(np.int32))
        srcs.append({"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    out = {"forced_routed_experts": np.stack(rows_ids), "gpu_weights": np.stack(rows_w), "tokens": np.stack(toks)}
    if indexer:
        out["forced_indexer_topk"] = np.stack(rows_idx)
    info = {
        "rows": len(files),
        "top_k": k,
        "main_moe_layers": main,
        "mtp_layers": mtp,
        "slots": int(out["forced_routed_experts"].shape[2]),
        "order_within_token": "GPU 路由权重降序",
        "indexer_topk": int(out["forced_indexer_topk"].shape[3]) if indexer else None,
        "sources": srcs,
    }
    return out, info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--choices", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-decoder-layers", type=int, default=None, help="主干层数（默认 = 最大 MoE 层号 + 1）")
    p.add_argument("--indexer", action="store_true", help="另写 forced_indexer_topk（各层 self_attn.topk）")
    p.add_argument("--step", type=int, default=None, help="只取 step_<k>_row_*（目录里有多步时必须给）")
    a = p.parse_args(argv)
    files = sorted(Path(a.choices).glob(f"step_{a.step if a.step is not None else '*'}_row_*.safetensors"))
    steps = {f.name.split("_")[1] for f in files}
    if len(steps) > 1:
        raise SystemExit(f"{a.choices} 里有多步（{sorted(steps)}），用 --step 选一步")
    if not files:
        raise SystemExit(f"{a.choices} 下没有 step_*_row_*.safetensors")
    out, info = build(files, a.num_decoder_layers, a.indexer)
    from janus.store import write_safetensors

    write_safetensors(Path(a.out), out, metadata={"info": json.dumps(info, ensure_ascii=False)})
    print(json.dumps({k: v for k, v in info.items() if k != "sources"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
