"""路由翻转定责（PROTOCOL 第 0 节「路由翻转定责」，2026-11-12 事前登记，只用于之后还没看过结果的判定）。

一个张量（或一项统计量）超出 GPU 噪声、R = rel(TPU, FP64 自选) ÷ GPU 中位数 rel(GPU, FP64 自选) > 1.2 时，
把 TPU 自己的离散选择（路由专家、DSA / 索引器 top-k）灌进 CPU FP64 参照重算，得到「FP64 灌 TPU 选择」：

    R_flip = rel(TPU, FP64 灌 TPU 选择) ÷ 中位数_i rel(GPU_i, FP64 自选)

分母沿用原参照的口径（GPU 对 FP64 自选，与 R 的分母相同），只把 TPU 一侧的参照换成灌它自己选择的那一份。
判为「已定责：路由翻转」要同时满足：
1. R > 1.2（本来就是要定责的格子）；
2. R_flip ≤ 1.2；
3. rel(TPU, FP64 灌 TPU 选择) < rel(TPU, FP64 自选)（灌了 TPU 的选择以后参照确实靠向 TPU）。
不满足的照旧计入未定责。正式判定只灌 TPU 自己的选择；灌 GPU 的选择只作诊断（PROTOCOL 2026-11-09）。
rel 的分母都是 ‖FP64 自选‖（单值统计量用逐元素向量的同一写法）。
"""

from __future__ import annotations

R_LIMIT = 1.2


def route_flip_check(rel_tpu_self: float, rel_tpu_inj: float, gpu_median_self: float, r_limit: float = R_LIMIT) -> dict:
    """→ {R, R_flip, closer, attributed, reason}。输入都是 rel_L2（分母 ‖FP64 自选‖）。"""
    if gpu_median_self <= 0:
        return {
            "R": float("nan"),
            "R_flip": float("nan"),
            "closer": False,
            "attributed": False,
            "reason": "GPU 对参照的距离为 0，R 无定义，不适用本条",
        }
    r = rel_tpu_self / gpu_median_self
    rf = rel_tpu_inj / gpu_median_self
    closer = rel_tpu_inj < rel_tpu_self
    if r <= r_limit:
        why, ok = "R ≤ 1.2，本来就已定责，不适用本条", False
    elif rf > r_limit:
        why, ok = "灌 TPU 选择后 R_flip > 1.2：计入未定责", False
    elif not closer:
        why, ok = "灌 TPU 选择后参照没有靠向 TPU：计入未定责", False
    else:
        why, ok = "已定责：路由翻转", True
    return {"R": r, "R_flip": rf, "closer": closer, "attributed": ok, "reason": why}
