"""输出头反向的数值：同一份最后隐状态 h（golden 第 1 步 rank 0 第一个 microbatch），比各实现的 dh 在 ŵ̄（输出层行均值方向）上的分量。
参照：用该实现自己的 logits（BF16）在 FP64 里算 softmax − onehot，再 FP64 乘输出层 BF16 权重。
用法：python head_bwd_ref.py <golden step 目录> [<TPU head 回放 step 目录>]
"""
import sys

import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
torch.set_num_threads(64)
G = sys.argv[1]
TOT = 65536.0
f = safe_open(f"{G}/final_norm.safetensors", "pt")
dhG = f.get_tensor("final_norm.out.grad")[:, 0].double()
fl = safe_open(f"{G}/logits.safetensors", "pt")
lgG = fl.get_tensor("logits")[:, 0].double()
dlG = fl.get_tensor("logits.grad")[:, 0].double()
lab = safe_open(f"{G}/input.safetensors", "pt").get_tensor("input.labels")[0]

W = None
for path in (f"{G}/param.other.safetensors", f"{G}/main_param.other.safetensors", "/tmp/jg/g1k-a/step_1/main_param.other.safetensors"):
    try:
        ff = safe_open(path, "pt")
    except Exception:
        continue
    for k in ff.keys():
        if k.endswith("output_layer.weight"):
            W = ff.get_tensor(k).bfloat16().double()     # GPU 前向用 BF16 参数
            print("W from", path, k, tuple(W.shape))
            break
    if W is not None:
        break
wbar = W.mean(0)
u = wbar / wbar.norm()
T = dhG.shape[0]


def ref_from_logits(lg):
    p = torch.softmax(lg, -1)
    d = p.clone()
    d[torch.arange(T), lab[:T]] -= 1.0
    return d / TOT


def report(tag, dh, dl, lg):
    d_ref = ref_from_logits(lg)
    dh_ref = d_ref @ W
    scale = float((dh * dh_ref).sum() / (dh_ref * dh_ref).sum())
    print(f"== {tag}: 与参照的尺度比 {scale:.6f}（按它把参照缩放后再比）")
    dh_ref = dh_ref * scale
    d_ref = d_ref * scale
    e = dh - dh_ref
    print(f"  dh rel_l2 {float(e.norm() / dh_ref.norm()):.3e}")
    print(f"  Σ_t dh·û: impl {float((dh @ u).sum()):+.6e}  ref {float((dh_ref @ u).sum()):+.6e}  "
          f"err {float((e @ u).sum()):+.4e}  err/|ref| {float((e @ u).sum() / (dh_ref @ u).abs().sum()):+.3e}")
    if dl is not None:
        s_impl = dl.sum(-1)
        print(f"  行和 s_t 均值 impl {float(s_impl.mean()):+.4e}（参照 0）  s_t·|w̄| 的 Σ_t {float(s_impl.sum() * wbar.norm()):+.4e}")
        ed = dl - d_ref
        # 去掉行和那份后，dlogits 误差经 W 在 û 上的贡献
        edc = ed - ed.mean(-1, keepdim=True)
        print(f"  dlogits 误差：行和部分经 W 在 û 上 {float((ed.sum(-1) * wbar.norm()).sum() * 1.0):+.4e}；"
              f"去行和后经 W 在 û 上 {float(((edc @ W) @ u).sum()):+.4e}")
        print(f"  dh 误差里 dlogits 解释不了的（矩阵乘本身）在 û 上 {float(((e - ed @ W) @ u).sum()):+.4e}")


report("GPU", dhG, dlG, lgG)
if len(sys.argv) > 2:
    R = sys.argv[2]
    fh = safe_open(f"{R}/head.safetensors", "pt")
    dhT = fh.get_tensor("head.dh")[:, 0].double()
    dlT = fh.get_tensor("head.dlogits")[:, 0].double()
    lgT = fh.get_tensor("head.logits")[:, 0].double()
    report("TPU", dhT, dlT, lgT)
    report("TPU（dlogits 按 RNE 舍到 BF16）", dhT, fh.get_tensor("head.dlogits")[:, 0].bfloat16().double(), lgT)
