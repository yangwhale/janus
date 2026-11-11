"""TPU 索引器 KD（第 1 步，第 0 行）：在 TPU 自己 sow 出的输入上用 CPU FP64 重算，按项拆开 TPU 记下的 KD 偏在哪。

输入是 `debug_sow_indexer=true` 的 TPU 诊断运行写出的 diag_kd_inputs.safetensors（每个产出层第 0 行的教师 q / k、
学生分数与掩码、TPU 图里的教师分布 p、学生 log q、逐行 KL）。对每层：
- KL 行均值：TPU 图里的值 vs FP64 在同一份输入上重算（原版 Megatron-Core 写法：L1 归一化夹 1e-10、精确 log_softmax、
  log(max(p, 1e-10))、p = 0 的项不计）；
- 交换：FP64 公式里只换 TPU 的 p、只换 TPU 的 log q、两个都换；
- 两处系统偏差：TPU 的 p 每行之和 − 1、TPU 的 exp(log q) 每行之和 − 1 与 log q 的平均误差；
  KL ≈ 真 KL + (Σp − 1) − ε（ε 为 log q 的整体平移），所以推算的相对偏差 = (Σp − 1) ÷ KL、−ε ÷ KL。

    python models/hunyuan4/analysis/kd_sow_vs_fp64.py <diag_kd_inputs.safetensors> <out.json>
"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

f = safe_open(sys.argv[1], 'np')
S = 4096
layers = sorted({k.rsplit('/debug_kd', 1)[0] for k in f.keys()})
out = {}
for L in layers:
    g = lambda n: f.get_tensor(f'{L}/debug_kd_{n}/0')
    q = torch.from_numpy(g('query').reshape(S, 8, 256).astype(np.float64))
    k = torch.from_numpy(g('key').reshape(S, 8, 256).astype(np.float64))
    s = torch.from_numpy(g('indexer_score').reshape(S, S).astype(np.float64))
    mk = torch.from_numpy(g('indexer_mask').reshape(S, S).astype(np.float64))
    m = (mk > -1e20) & torch.ones(S, S, dtype=torch.bool).tril()
    tp = torch.from_numpy(g('target').reshape(S, S).astype(np.float64))
    tl = torch.from_numpy(g('log_indexer').reshape(S, S).astype(np.float64))
    tr = torch.from_numpy(g('kl_rows').reshape(S).astype(np.float64))
    att = torch.zeros(S, S, dtype=torch.float64)
    for h in range(8):
        sc = q[:, h] @ k[:, h].T
        att += torch.nan_to_num(torch.softmax(sc.masked_fill(~m, float('-inf')), -1), nan=0.0)
    p = att / att.sum(-1, keepdim=True).clamp_min(1e-10)
    lq = torch.log_softmax(s.masked_fill(~m, float('-inf')), -1)
    def kl(pp, ll):
        return torch.where(m & (pp > 0), pp * (torch.log(pp.clamp_min(1e-10)) - ll), torch.zeros_like(pp)).sum(-1)
    k64 = kl(p, lq)
    r = dict(
        kl_row_mean_fp64=float(k64.mean()), kl_row_mean_tpu=float(tr.mean()),
        rel_tpu_vs_fp64=float(tr.mean() / k64.mean() - 1),
        fp64_formula_on_tpu_p_and_logq_rel=float(kl(tp, tl).mean() / k64.mean() - 1),
        swap_tpu_p_only_rel=float(kl(tp, lq).mean() / k64.mean() - 1),
        swap_tpu_logq_only_rel=float(kl(p, tl).mean() / k64.mean() - 1),
        target_rowsum_minus1_mean=float((tp.sum(-1) - 1).mean()),
        target_rowsum_minus1_frac_pos=float(((tp.sum(-1) - 1) > 0).double().mean()),
        logq_exp_rowsum_minus1_mean=float((torch.where(m, tl.exp(), torch.zeros_like(tl)).sum(-1) - 1).mean()),
        target_rel_l2=float((tp[m] - p[m]).norm() / p[m].norm()),
        logq_mean_err=float((tl[m] - lq[m]).mean()),
    )
    # 只改一项的效应（推算）：p 整行放大 (1+δ) 使 KL 增加约 δ；log q 整体偏 ε 使 KL 减少约 ε·Σp
    r['pred_from_rowsum_rel'] = r['target_rowsum_minus1_mean'] / r['kl_row_mean_fp64']
    r['pred_from_logq_shift_rel'] = -r['logq_mean_err'] / r['kl_row_mean_fp64']
    out[L] = r
    print(L.split('/')[1] if '/' in L else L, json.dumps({a: (f'{b:+.3e}' if isinstance(b, float) else b) for a, b in r.items()}))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
