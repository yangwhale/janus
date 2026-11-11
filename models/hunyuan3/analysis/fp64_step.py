"""混元 3（截 4 层）第 1 步整网梯度的 FP64 参照：同一份 BF16 权重、golden 第 1 步的全局 batch（16 条 × 4096），
全程 FP64 前向 + 自动求导，得到 GPU 第 1 步梯度的「精确算术」版本。

用途：L3 第 1 步（grads / optim_state / updated / bias_after）和 L2 逐层参数梯度。GPU 自己重跑第 1 步路由从不翻转，
TPU 只改执行方式的重跑也不翻转、梯度只差 1e-7，两把执行噪声的尺子都量不出「两种硬件各自的 BF16 舍入顺序与近似平局路由」这一份。
和前向同一个原则换尺子：R = rel_l2(被测, FP64) / rel_l2(GPU, FP64)，被测不比 GPU 离真值远就算对齐（fwd_rratio.py --mode step）。

语义（推断，靠 GPU 梯度对 FP64 梯度的投影系数核对，见输出的 scale_fit）：
- 每个 microbatch 的 loss = Σ(逐 token 交叉熵 · loss_mask) / Σ loss_mask，整步梯度 = 16 个 microbatch 的平均（Megatron 默认：
  microbatch 内按 token 平均、梯度累积除以 microbatch 数、数据并行取平均）；
- 路由按 FP64 分数 + 第 1 步的 expert_bias 选专家（选谁不可微），权重 = 选中 sigmoid 分数归一化 × 缩放；无辅助损失（moe_aux_loss_coeff 0）；
- 只对 GPU 抓了的参数求梯度（专家只抓了 rank 0 的本地 0、1 号，即全局 0、1 号专家），其余专家权重当常数。
同时累计每层每个专家第 1 步选中的 token 数（FP64 路由），给 bias_after 的真值用。

  python fp64_step.py --hf <HF 目录> --golden-data <golden 运行目录（含 data/batches）> --gpu <运行根>/g1k-a --out <目录> [--micro 0:16]
输出 <out>/grads.safetensors（grad.<Megatron 名>，FP64）、<out>/meta.json（scale_fit、counts、loss）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "frameworks" / "maxtext"))
import fp64_backward as B  # noqa: E402
import fp64_forward as F  # noqa: E402

from janus.dump.name_map import NameMap  # noqa: E402
from janus.store import Run  # noqa: E402

NAME_MAP = HERE.parent / "model" / "name_map.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--golden-data", required=True)
    ap.add_argument("--gpu", required=True, help="GPU 第 1 步全量状态读的运行（取要算的梯度名单与 expert_bias）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--micro", default="", help="只算全局 batch 的这些行（a:b），分进程跑后用 --merge 合并")
    ap.add_argument("--merge", nargs="*", help="把几个 --micro 的输出目录加起来")
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import load_file, save_file
    g = Run.open(a.gpu)
    if a.merge:
        acc, counts, meta0 = {}, None, None
        for d in a.merge:
            t = load_file(str(Path(d) / "grads.safetensors"))
            m = json.loads((Path(d) / "meta.json").read_text())
            for k, v in t.items():
                acc[k] = acc.get(k, 0) + v
            counts = m["counts"] if counts is None else {L: (np.add(counts[L], cc)).tolist() for L, cc in m["counts"].items()}
            meta0 = m if meta0 is None else {**meta0, "rows": meta0["rows"] + m["rows"], "seconds": meta0["seconds"] + m["seconds"],
                                             "loss_per_micro": meta0["loss_per_micro"] + m["loss_per_micro"]}
        meta0["counts"] = counts
        meta0["merged_from"] = list(a.merge)
        meta0["scale_fit"] = scale_fit(g, acc)
        save_file(acc, str(out / "grads.safetensors"))
        (out / "meta.json").write_text(json.dumps(meta0, ensure_ascii=False, indent=1))
        print(json.dumps(meta0["scale_fit"], ensure_ascii=False))
        return 0

    torch.set_num_threads(a.threads)
    from replay import GoldenBatches
    hf = F.HF(a.hf)
    c = hf.cfg
    gb = GoldenBatches(a.golden_data)
    batch = gb.global_batch(1)
    rows = list(range(gb.global_rows))
    if a.micro:
        lo, hi = (int(x) for x in a.micro.split(":"))
        rows = rows[lo:hi]
    n_total = gb.global_rows
    nm = NameMap.load(NAME_MAP)
    want = {n[len("grad."):] for n in g.names(1, "grad.*")}
    ents = [e for e in nm.entries if not e.buffer and nm.megatron_name(e) in want and nm.expert_owner(e) in (None, 0)]   # 专家只抓了 rank 0 的
    missing = want - {nm.megatron_name(e) for e in ents}
    if missing:
        raise SystemExit(f"GPU 抓了但映射表里找不到：{sorted(missing)[:5]}")
    need_grad = {h for e in ents for h in e.hf}
    print(f"[fp64-step] 行 {rows[0]}..{rows[-1]}（共 {n_total}），要梯度的 HF 张量 {len(need_grad)} 个", flush=True)

    # 要梯度的：FP64 叶子；其余（大部分专家）：BF16 原样缓存，用时转 FP64
    leaves, cache = {}, {}
    for name in hf.where:
        if name in need_grad:
            leaves[name] = torch.as_tensor(hf.get(name)).requires_grad_(True)
        elif ".mlp.experts." in name:
            cache[name] = torch.as_tensor(hf.get(name).astype(np.float32)).to(torch.bfloat16)

    def W(n):
        if n in leaves:
            return leaves[n]
        if n in cache:
            return cache[n].to(torch.float64)
        leaves[n] = torch.as_tensor(hf.get(n))     # 不要梯度的非专家张量（如 expert_bias），常驻 FP64
        return leaves[n]

    nl, fd = c["num_hidden_layers"], c["first_k_dense_replace"]
    bias = {}
    for i in range(fd, nl):
        b_hf = hf.get(f"model.layers.{i}.mlp.expert_bias")
        b_g = g.tensor(1, f"buffer.decoder.layers.{i}.mlp.router.expert_bias").astype(np.float64)
        if not np.array_equal(b_hf, b_g):
            raise SystemExit(f"第 {i} 层：HF 的 expert_bias 与 GPU 第 1 步的不一致")
        bias[i] = b_hf
    counts = {i: np.zeros(c["num_experts"], np.int64) for i in bias}
    losses = []
    t0 = time.time()
    for r in rows:
        tok = torch.as_tensor(batch["tokens"][r].astype(np.int64))
        lab = torch.as_tensor(batch["labels"][r].astype(np.int64))
        lm = torch.as_tensor(batch["loss_mask"][r].astype(np.float64))
        cos, sin = B.rope_tables(c, batch["position_ids"][r])
        x = W("model.embed_tokens.weight")[tok]
        for i in range(nl):
            # 逐层重算（checkpoint）：只留层边界的激活，内存从约 150 GB 降到几十 GB；重算时不再累计选中数
            first = [True]

            def fn(x_, i=i, first=first):
                cnt = counts.get(i) if first[0] else None
                first[0] = False
                return B.layer_torch(c, W, i, x_, cos, sin, bias.get(i), cnt)
            x = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
        eps = c["rms_norm_eps"]
        yn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * W("model.norm.weight")
        logits = yn @ W("lm_head.weight").T
        ce = torch.logsumexp(logits, -1) - logits.gather(1, lab[:, None])[:, 0]
        loss = (ce * lm).sum() / lm.sum()
        (loss / n_total).backward()
        losses.append(float(loss.detach()))
        del x, yn, logits, ce, loss
        print(f"[fp64-step] 行 {r} loss {losses[-1]:.6f}（累计 {time.time() - t0:.0f}s）", flush=True)

    grads = {}
    for e in ents:
        arrs = [leaves[h].grad.numpy() for h in e.hf]
        grads["grad." + nm.megatron_name(e)] = np.ascontiguousarray(nm.to_megatron(e, arrs))
    meta = {"rows": len(rows), "row_range": [rows[0], rows[-1] + 1], "global_rows": n_total, "loss_per_micro": losses,
            "counts": {str(i): v.tolist() for i, v in counts.items()}, "seconds": time.time() - t0,
            "semantics": "每个 microbatch 按 token 平均，整步除以 microbatch 总数；FP64 路由；无辅助损失"}
    if len(rows) == n_total:
        meta["scale_fit"] = scale_fit(g, grads)
        print(json.dumps(meta["scale_fit"], ensure_ascii=False), flush=True)
    save_file(grads, str(out / "grads.safetensors"))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


def scale_fit(g: Run, grads: dict) -> dict:
    """GPU 梯度对 FP64 梯度的最小二乘投影系数 s = ⟨g, f⟩/⟨f, f⟩（整体与逐张量的范围）。约定对上时 s ≈ 1。"""
    num = den = 0.0
    per = []
    for k, f in grads.items():
        gv = g.tensor(1, k).astype(np.float64).reshape(f.shape)
        a, b = float((gv * f).sum()), float((f * f).sum())
        num += a
        den += b
        if b > 0:
            per.append(a / b)
    return {"overall": num / den, "per_tensor_min": min(per), "per_tensor_max": max(per), "n": len(per)}


if __name__ == "__main__":
    sys.exit(main())
