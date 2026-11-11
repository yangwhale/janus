"""索引器 KD 的 GPU 偏差定位：同一行先跑 CPU FP64 参照，再跑一份 GPU 写法的 BF16 前向，交换 KD 的输入看偏差从哪来。

为什么：真实宽度三层（06-rw3）第 1 步索引器 KD 对 CPU FP64，GPU 第 0 层 +1.53e-3、第 1 层 +2.20e-3（8 份逐位
相同）、MTP 层约 +2e-4。KD 本身的 FP32 算术只差 1e-7 级（`kd_input_rounding.py`），偏差在输入。这里逐行：

1. FP64 参照：`step1_ref.py` 同一套建法（ForceDtype 把写死的低精度改写成 FP64，路由专家 BF16 存储、FP64 计算）；
2. BF16 前向：同一套 Megatron-Core 代码（local 层，CPU），参数 BF16、iHC / sink / k_norm 等留 FP32（Float16Module，
   同 `train_hy4mg.py`），代码里写死的 `.float()` 照原样是 FP32 ——这是 GPU 的写法，但矩阵乘的累加顺序、TE 的
   融合算子与 GPU 不同（推断：噪声量级相同，逐位不同）；
3. 每个产出 KD 的层抓两份 KD 输入（教师 query / key，索引器 q / k / w，DSA top-k），用原版
   `compute_dsa_indexer_loss` 算若干组合：全 FP64（参照）、全 BF16 前向的（模拟 GPU）、FP64 底上只换一个输入
   （换成 BF16 前向的那份）、BF16 底上只把一个输入换回 FP64，以及只把 FP64 的输入舍一次 BF16；
4. 索引器分开看输入与内部算子：输入（x、qr）取 FP64 / 舍一次 / BF16 前向的 × 索引器用 FP64（ForceDtype）或 BF16
   写法算；BF16 写法下再只换 RoPE 的精度（`install_rope_emulation`：exact / f32freq / cs16 / ops16 / full16）。

逐行结果写进 `<out>/rows.jsonl`；`--summarize` 汇总成 per-layer 的相对差（口径同 summary 的 indexer_loss_per_layer：
逐行 KD 相加）。

用法（CPU，Megatron-Core 源码在 PYTHONPATH；内存约 FP64 参照 + 一份 BF16 模型，D3 估约 200 GiB，用 cgroup 限）：
    python kd_bf16_vs_fp64.py --hf <HF 目录> --tokens tokens.npy --out <目录> --layers 3 --mtp on --rows 0,1,2,3
    python kd_bf16_vs_fp64.py --summarize <目录>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MEG = os.path.join(HERE, "..", "frameworks", "megatron")
sys.path.insert(0, MEG)
sys.path.insert(0, HERE)

INPUTS = ("query", "key", "q", "k", "w", "topk")
COEFF = 0.1 / 3
SCALE = 256**-0.5
EPS = 1e-10  # 原版 INDEXER_LOSS_EPS
ROPE = 64  # 索引器每头后 64 维是 RoPE（HF 布局，qk_rope_head_dim）


def _valid(topk, s0, s1, sk):
    """top-k ∩ 因果，[c, sk] bool（同原版：scatter 进 -inf 掩码再与因果掩码取交）。"""
    pos = torch.arange(s0, s1).unsqueeze(1)
    causal = torch.arange(sk).unsqueeze(0) <= pos  # sq == sk：key 位置 ≤ query 位置
    return _topk_mask(topk[0, s0:s1], sk) & causal


def _topk_mask(t, sk):
    """[c, k] 的下标（-1 = 空位）→ [c, sk] bool。空位不能直接 scatter（会把第 0 列写回 False），同原版用 amax 归约。"""
    t = t.long()
    m = torch.zeros(t.shape[0], sk, dtype=torch.float32)
    m.scatter_reduce_(1, t.clamp_min(0), (t >= 0).float(), reduce="amax", include_self=True)
    return m > 0


def teacher(query, key, topk, dt, chunk=256):
    """教师目标：每头 softmax（只在 top-k ∩ 因果上）→ 头求和 → L1 归一（clamp 1e-10）。返回 [sq, sk]，dt 精度。"""
    sq, sk = query.shape[0], key.shape[0]
    qh = query[:, 0].to(dt).permute(1, 0, 2)  # [h, sq, d]
    kh = key[:, 0].to(dt).permute(1, 2, 0)  # [h, d, sk]
    out = torch.empty(sq, sk, dtype=dt)
    for s0 in range(0, sq, chunk):
        s1 = min(sq, s0 + chunk)
        v = _valid(topk, s0, s1, sk)
        sc = torch.bmm(qh[:, s0:s1], kh) * SCALE
        sc = sc.masked_fill(~v, torch.finfo(dt).min)
        p = torch.exp(sc - sc.max(-1, keepdim=True).values).masked_fill(~v, 0.0)
        p = p / p.sum(-1, keepdim=True).clamp_min(EPS)
        t = p.sum(0)
        out[s0:s1] = t / t.sum(-1, keepdim=True).clamp_min(EPS)
    return out


def student(q, k, w, topk, dt, chunk=256):
    """学生：Σ_头 relu(q·k) · w → log_softmax（只在 top-k ∩ 因果上，其余记 0）。返回 [sq, sk]。"""
    sq, sk = q.shape[0], k.shape[0]
    qq, kk, ww = q[:, 0].to(dt), k[:, 0].to(dt), w[:, 0].to(dt)
    out = torch.empty(sq, sk, dtype=dt)
    for s0 in range(0, sq, chunk):
        s1 = min(sq, s0 + chunk)
        v = _valid(topk, s0, s1, sk)
        sc = torch.einsum("shd,td->sht", qq[s0:s1], kk).relu_()
        sc = (sc * ww[s0:s1].unsqueeze(-1)).sum(1)
        sc = sc.masked_fill(~v, torch.finfo(dt).min)
        out[s0:s1] = torch.log_softmax(sc, -1).masked_fill(~v, 0.0)
    return out


def kl_sum(target, logp, topk):
    """Σ_行 Σ_key target · (log max(target, 1e-10) − logp)，只在 top-k ∩ 因果上；× 逐层系数（同原版 per-token）。"""
    sq, sk = target.shape
    v = _valid(topk, 0, sq, sk)
    terms = (target * (torch.log(target.clamp_min(EPS)) - logp)).masked_fill(~v, 0.0)
    return float(terms.sum(-1).sum().double()) * COEFF


def tap(cores, store):
    """包住每个 full 层索引器的 forward_before_topk，记下 q / k / w。"""
    for name, c in cores:
        if not c.is_full:
            continue
        c.capture = True
        orig = c.indexer.forward_before_topk

        def fbt(*args, _o=orig, _n=name, **kw):
            out = _o(*args, **kw)
            store[_n] = tuple(t.detach().clone() for t in out)
            return out

        c.indexer.forward_before_topk = fbt


def collect(cores, idx_store):
    out = {}
    for name, c in cores:
        if not c.is_full:
            continue
        q, k, w = idx_store[name]
        x, qr, query, key = c.last_kd_inputs
        out[name] = {"query": query, "key": key, "q": q, "k": k, "w": w, "topk": c.last_topk.clone(), "x": x, "qr": qr}
    return out


def cores_of(gpt, mtp):
    cores = [(str(i), layer.self_attention.core_attention) for i, layer in enumerate(gpt.decoder.layers)]
    if mtp:
        cores.append(("mtp0", gpt.mtp.layers[0].mtp_model_layer.self_attention.core_attention))
    return cores


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def topk_overlap(a, b):
    """两组 top-k 作为集合的重合比例（只算有效位置）。"""
    sq = a.shape[1]
    ma, mb = _topk_mask(a[0], sq), _topk_mask(b[0], sq)
    return float((ma & mb).sum() / ma.sum())


def mix(base, other, sl):
    """base 的最后一维 sl 那段换成 other 的。"""
    out = base.clone()
    out[..., sl] = other[..., sl]
    return out


ROPE_MODE = {"mode": None, "freqs64": None}


def install_rope_emulation(H):
    """诊断：把 hy4_mcore 里索引器 `_apply_rope` 用的 apply_rotary_pos_emb 换成可切换精度的版本（只影响索引器，
    MLA 主注意力用的是 Megatron-Core 自己导入的那份）。mode：
    - None：原版（BF16 输入时 cos / sin 先转 BF16，乘、加各舍一次 BF16）；
    - full16：照原版的舍入点在 FP64 里重做（核对用，应与原版一致）；
    - exact：FP64 角度、FP64 cos / sin、FP64 运算，结果只舍一次 BF16；
    - f32freq：同 exact 但角度用原版传进来的 FP32 角度（位置 × FP32 inv_freq）；
    - cs16：同 exact 但 cos / sin 舍成 BF16；
    - ops16：同 exact 但 t·cos、rot(t)·sin、两者之和各舍一次 BF16。
    """
    orig = H.apply_rotary_pos_emb

    def rnd(t):
        return t.to(torch.bfloat16).to(torch.float64)

    def emul(t, freqs, config=None, cu_seqlens=None, mscale=1.0, cp_group=None, mla_rotary_interleaved=False, **kw):
        mode = ROPE_MODE["mode"]
        if mode is None or t.dtype != torch.bfloat16:
            return orig(
                t,
                freqs,
                config=config,
                cu_seqlens=cu_seqlens,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=mla_rotary_interleaved,
                **kw,
            )
        assert cu_seqlens is None and not config.rotary_interleaved
        fr = freqs if mode == "f32freq" else ROPE_MODE["freqs64"]
        fr = fr.to(torch.float64)
        if fr.dim() == t.dim() + 1 and fr.size(-2) == 1:
            fr = fr.squeeze(-2)
        rot_dim = fr.shape[-1]
        assert rot_dim == t.shape[-1]
        x = t.to(torch.float64)
        if mla_rotary_interleaved:
            x = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
        cos, sin = torch.cos(fr) * mscale, torch.sin(fr) * mscale
        if mode in ("cs16", "full16"):
            cos, sin = rnd(cos), rnd(sin)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        if mode in ("ops16", "full16"):
            y = rnd(rnd(x * cos) + rnd(rot * sin))
        else:
            y = x * cos + rot * sin
        return y.to(torch.bfloat16)

    H.apply_rotary_pos_emb = emul


def layer_variants(name, x64, x16, c64, c16, force, kd_one, a):
    """一层的全部组合。教师 T_*、学生 S_* 分别算一次再两两配 KL；FP64 算术、FP64 的 top-k，
    只有 bf16run（全用 BF16 前向的输入与 top-k、FP32 算术，原版函数）是整条 GPU 写法的对照。"""

    f64 = torch.float64
    b16 = {k: v.to(f64) if v.is_floating_point() else v for k, v in x16.items()}
    r16 = {k: v.to(torch.bfloat16).to(f64) if v.is_floating_point() else v for k, v in x64.items()}
    topk = x64["topk"]
    lay = {
        "dtype_bf16run": {k: str(v.dtype) for k, v in x16.items()},
        "rel_l2_bf16run": {k: rel_l2(b16[k], x64[k]) for k in x64 if k != "topk"},
        "rel_l2_round": {k: rel_l2(r16[k], x64[k]) for k in x64 if k != "topk"},
        "topk_overlap": topk_overlap(x64["topk"], x16["topk"]),
    }

    # 索引器：输入（x, qr）取 FP64 / 舍一次 / BF16 前向的，内部算子 FP64（ForceDtype）或 BF16 前向那套（GPU 写法）
    def idx64(x, qr):
        torch.set_default_dtype(f64)
        with force, torch.no_grad():
            return tuple(t.detach().clone() for t in c64.indexer.forward_before_topk(x.to(f64), qr.to(f64), None))

    def idx16(x, qr):
        torch.set_default_dtype(torch.float32)
        with torch.no_grad():
            out = c16.indexer.forward_before_topk(x.to(torch.bfloat16), qr.to(torch.bfloat16), None)
        return tuple(t.detach().to(f64) for t in out)

    studs = {
        "S_ref": (x64["q"], x64["k"], x64["w"]),
        "S_all16": (b16["q"], b16["k"], b16["w"]),
        "S_q16": (b16["q"], x64["k"], x64["w"]),
        "S_k16": (x64["q"], b16["k"], x64["w"]),
        "S_w16": (x64["q"], x64["k"], b16["w"]),
        "S_qrope16": (mix(x64["q"], b16["q"], slice(-ROPE, None)), x64["k"], x64["w"]),
        "S_qnope16": (mix(x64["q"], b16["q"], slice(0, -ROPE)), x64["k"], x64["w"]),
        "S_krope16": (x64["q"], mix(x64["k"], b16["k"], slice(-ROPE, None)), x64["w"]),
        "S_knope16": (x64["q"], mix(x64["k"], b16["k"], slice(0, -ROPE)), x64["w"]),
        "S_round": (r16["q"], r16["k"], x64["w"]),
    }
    torch.set_default_dtype(f64)
    chk = idx64(x64["x"], x64["qr"])
    lay["idx64_reproduces_fp64"] = max(rel_l2(c, o) for c, o in zip(chk, studs["S_ref"]))
    chk = idx16(x16["x"], x16["qr"])
    lay["idx16_reproduces_bf16run"] = max(rel_l2(c, o) for c, o in zip(chk, studs["S_all16"]))
    studs["S_in16_op64"] = idx64(b16["x"], b16["qr"])  # 输入 BF16 前向的，内部 FP64
    studs["S_inr16_op64"] = idx64(r16["x"], r16["qr"])  # 输入只舍一次，内部 FP64
    studs["S_inr16_op16"] = idx16(r16["x"], r16["qr"])  # 输入只舍一次，内部 BF16 写法
    for mode in ("full16", "exact", "f32freq", "cs16", "ops16"):  # 输入只舍一次、索引器 BF16 写法，只换 RoPE 的精度
        ROPE_MODE["mode"] = mode
        studs[f"S_inr16_op16_rope_{mode}"] = idx16(r16["x"], r16["qr"])
    ROPE_MODE["mode"] = None
    studs["S_x16_op64"] = idx64(b16["x"], x64["qr"])  # 只 x（k、w 的输入）换成 BF16 前向的
    studs["S_qr16_op64"] = idx64(x64["x"], b16["qr"])  # 只 qr（q 的输入）换成 BF16 前向的
    torch.set_default_dtype(f64)
    teach = {
        "T_ref": (x64["query"], x64["key"]),
        "T_all16": (b16["query"], b16["key"]),
        "T_key16": (x64["query"], b16["key"]),
        "T_query16": (b16["query"], x64["key"]),
        "T_keyr16": (x64["query"], r16["key"]),
    }
    T = {n: teacher(qq, kk, topk, f64, a.chunk) for n, (qq, kk) in teach.items()}
    Sd = {n: student(qq, kk, ww, topk, f64, a.chunk) for n, (qq, kk, ww) in studs.items()}
    kd = {"ref": kl_sum(T["T_ref"], Sd["S_ref"], topk)}
    for n in T:
        if n != "T_ref":
            kd[n] = kl_sum(T[n], Sd["S_ref"], topk)
    for n in Sd:
        if n != "S_ref":
            kd[n] = kl_sum(T["T_ref"], Sd[n], topk)
    kd["T_all16+S_all16"] = kl_sum(T["T_all16"], Sd["S_all16"], topk)
    # 整条 GPU 写法（原版函数、FP32、BF16 前向的 top-k）与原版函数的 FP64 参照（核对上面拆开写的 KL）
    inp = {k: x16[k] for k in ("query", "key", "q", "k", "w")}
    kd["bf16run"], _ = kd_one({k: v.to(torch.float32) for k, v in inp.items()}, x16["topk"], torch.float32, a.chunk)
    kd["ref_orig_fn"], _ = kd_one({k: x64[k] for k in inp}, topk, f64, a.chunk)
    lay["kd"] = kd
    return lay


def run(a):
    import numpy as np
    import step1_ref as S

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    f64 = torch.float64
    force = S.ForceDtype(f64)
    S.cpu_megatron_setup()
    import hy4_mcore as H
    import hy4_weights as W
    from kd_input_rounding import kd_one

    orig_st = {nm: getattr(W.Safetensors, nm) for nm in ("get", "get_slice")}
    S.relax_mask_dtype(f64)

    # ---- FP64 参照（同 step1_ref：--expert-store ram、只前向）
    t0 = time.time()
    torch.set_default_dtype(f64)
    with force:
        for nm, fn in orig_st.items():
            setattr(W.Safetensors, nm, S._unforced(fn))
        ns = SimpleNamespace(
            hf=a.hf,
            layers=a.layers,
            mtp=a.mtp,
            indexer_loss=0.1,
            head_chunk=1024,
            log_max_logit=False,
            expert_store="ram",
        )
        hf, cfg64, m64, L, mtp, coeff = S.build(ns, f64)
        m64.train()
        S.install_expert_store(m64, S.ExpertGradStore(None))
    for nm, fn in orig_st.items():
        setattr(W.Safetensors, nm, fn)
    print("BUILT_FP64", round(time.time() - t0, 1), flush=True)

    # ---- BF16 前向（GPU 写法：BF16 参数 + Float16Module 留 FP32 标记的参数；local 层）
    t0 = time.time()
    torch.set_default_dtype(torch.float32)
    from megatron.core.transformer.module import Float16Module

    kw = H.hy4_config_kwargs(hf, num_layers=L)
    kw["moe_token_dispatcher_type"] = "allgather"
    extra = dict(
        params_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=False,
        transformer_impl="local",
        moe_grouped_gemm=False,
        use_cpu_initialization=True,
        hy4_train_indexer=True,
        dsa_indexer_loss_coeff=coeff,
        dsa_indexer_use_sparse_loss=True,
        calculate_per_token_loss=True,
        hy4_attn_scores_fp32=True,
        hy4_head_chunk=1024,
        cross_entropy_loss_fusion=False,
        gradient_accumulation_fusion=False,
    )
    if mtp:
        extra.update(mtp_num_layers=1, mtp_loss_scaling_factor=float(hf.get("mtp_loss_factor", 0.1)))
    cfg16 = H.Hy4Config(**{**kw, **extra})
    m16 = Float16Module(
        cfg16, H.build_model(cfg16, use_te=False, vocab_size=hf["vocab_size"], max_seq=1 << 20, mtp=mtp)
    )
    W.load_hf_into_mcore(m16.module, cfg16, a.hf)
    m16.train()
    dts = {}
    for _n, p in m16.named_parameters():
        dts[str(p.dtype)] = dts.get(str(p.dtype), 0) + p.numel()
    print("BUILT_BF16", round(time.time() - t0, 1), json.dumps(dts), flush=True)

    c64, c16 = cores_of(m64, mtp), cores_of(m16.module, mtp)
    s64, s16 = {}, {}
    tap(c64, s64)
    tap(c16, s16)
    install_rope_emulation(H)
    ix64 = next(c for _, c in c64 if c.is_full).indexer
    ix16 = next(c for _, c in c16 if c.is_full).indexer
    torch.set_default_dtype(f64)
    with force:
        ROPE_MODE["freqs64"] = ix64.rotary_pos_emb(a.seq).detach().clone()
    torch.set_default_dtype(torch.float32)
    f32fr = ix16.rotary_pos_emb(a.seq).detach()
    print(
        "ROPE_FREQS",
        ROPE_MODE["freqs64"].dtype,
        f32fr.dtype,
        tuple(f32fr.shape),
        "max|f32-f64|",
        float((f32fr.double() - ROPE_MODE["freqs64"]).abs().max()),
        flush=True,
    )
    toks = np.load(a.tokens, mmap_mode="r")
    eod = int(hf["eos_token_id"])
    pos = torch.arange(a.seq).unsqueeze(0)
    done = set()
    rows_path = os.path.join(a.out, "rows.jsonl")
    if os.path.exists(rows_path):
        done = {json.loads(x)["row"] for x in open(rows_path)}
    for r in [int(x) for x in a.rows.split(",")]:
        if r in done:
            continue
        row = np.asarray(toks[r, : a.seq + 1]).astype(np.int64)
        ids = torch.from_numpy(row[:-1])[None]
        labels = torch.from_numpy(row[1:])[None]
        rec = {"row": r, "layers": {}}
        t0 = time.time()
        torch.set_default_dtype(f64)
        with force, torch.no_grad():
            lm = (ids != eod).to(f64)
            o = m64(ids, pos, attention_mask=None, labels=labels, loss_mask=lm)
            rec["lm_sum_fp64"] = float((o * lm).sum())
        x64 = collect(c64, s64)
        rec["fwd64_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        torch.set_default_dtype(torch.float32)
        with torch.no_grad():
            lm = (ids != eod).float()
            o = m16(ids, pos, attention_mask=None, labels=labels, loss_mask=lm)
            rec["lm_sum_bf16"] = float((o.float() * lm).sum())
        x16 = collect(c16, s16)
        rec["fwd16_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        for name in x64:
            rec["layers"][name] = layer_variants(
                name, x64[name], x16[name], dict(c64)[name], dict(c16)[name], force, kd_one, a
            )
        rec["kd_s"] = round(time.time() - t0, 1)
        with open(rows_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        brief = {
            n: {v: f"{lay['kd'][v] / lay['kd']['ref'] - 1:+.2e}" for v in ("bf16run", "T_key16", "S_all16")}
            for n, lay in rec["layers"].items()
        }
        print("ROW", r, rec["fwd64_s"], rec["fwd16_s"], rec["kd_s"], json.dumps(brief), flush=True)


def summarize(d):
    recs = [json.loads(x) for x in open(os.path.join(d, "rows.jsonl"))]
    layers = list(recs[0]["layers"])
    out = {"rows": [r["row"] for r in recs], "per_layer": {}, "all": {}}
    names = list(recs[0]["layers"][layers[0]]["kd"])
    tot = dict.fromkeys(names, 0.0)
    for L in layers:
        s = {v: sum(r["layers"][L]["kd"][v] for r in recs) for v in names}
        for v in names:
            tot[v] += s[v]
        out["per_layer"][L] = {
            "rel_vs_ref": {v: s[v] / s["ref"] - 1 for v in names if v != "ref"},
            "ref_kd_sum": s["ref"],
            "rel_l2_bf16run_mean": {
                k: sum(r["layers"][L]["rel_l2_bf16run"][k] for r in recs) / len(recs)
                for k in recs[0]["layers"][L]["rel_l2_bf16run"]
            },
            "rel_l2_round_mean": {
                k: sum(r["layers"][L]["rel_l2_round"][k] for r in recs) / len(recs)
                for k in recs[0]["layers"][L]["rel_l2_round"]
            },
            "topk_overlap_mean": sum(r["layers"][L]["topk_overlap"] for r in recs) / len(recs),
        }
    out["all"] = {v: tot[v] / tot["ref"] - 1 for v in names if v != "ref"}
    out["lm_rel_bf16_vs_fp64"] = sum(r["lm_sum_bf16"] for r in recs) / sum(r["lm_sum_fp64"] for r in recs) - 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf")
    p.add_argument("--tokens")
    p.add_argument("--out")
    p.add_argument("--rows", default="0")
    p.add_argument("--layers", type=int, default=0)
    p.add_argument("--mtp", type=lambda s: s.lower() in ("1", "true", "on"), default=None)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--threads", type=int, default=40)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--summarize", default="")
    a = p.parse_args()
    if a.summarize:
        print(json.dumps(summarize(a.summarize), indent=1))
        return
    run(a)


if __name__ == "__main__":
    main()
