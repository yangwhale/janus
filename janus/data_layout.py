"""数据加载器的「样本 → 设备 / rank / microbatch」布局：Megatron 与 MaxText 各自怎么分，以及两者之间的换序。

两边从同一条全局样本流（流下标 i = 第 i 个训练样本）里按步取 gbs 个样本，但一步之内分给谁不一样：

- Megatron（MegatronPretrainingSampler，dataloader_type=single）：每个 microbatch 依次取 dp×mbs 个，
  rank r 拿其中第 [r×mbs, (r+1)×mbs) 个。第 m 个 microbatch、rank r、第 k 个样本 = 步内第 m×dp×mbs + r×mbs + k 个。
- MaxText（Grain 批 + GA）：一次读整个全局批（gbs = dp×mbs×ga 行，第 g 行 = 步内第 g 个样本），按批维连续切给设备，
  设备 d 持有第 [d×mbs×ga, (d+1)×mbs×ga) 行；梯度累积再按 reshape(gbs//ga, ga) + swapaxes 切 microbatch
  （maxtext/utils/gradient_accumulation.py 的 reshape_to_microbatch_accumulations），
  第 m 个 microbatch 的第 j 行 = 全局第 j×ga + m 行，落在设备 j // mbs 上。
  所以设备 d、microbatch m、第 k 个样本 = 全局第 (d×mbs + k)×ga + m 行。

多 host 时 MaxText 先按 host 切样本流再成批（grain_data_processing 的 dataset[host_index::host_count]），
host h 的本地批是步内第 h、h+H、h+2H… 个样本；全局批按 host 拼接（host h 的设备在 mesh 批维上连续时），
所以全局第 g 行 = 步内第 (g mod B/H)×H + g div (B/H) 个样本（B = 全局批，H = host 数）。

ga = 1、单 host 时两边一致；ga > 1 或多 host 时同一个样本落在不同的设备 / microbatch 上。`maxtext_perm_for_megatron`
给出把 MaxText 全局批重排成 Megatron 布局的行置换：batch[perm] 之后，MaxText 设备 d、microbatch m 拿到的正是
Megatron rank d、microbatch m 的样本。
"""
from __future__ import annotations

import numpy as np


def megatron_slot(rank: int, micro: int, k: int, dp: int, mbs: int) -> int:
    """Megatron：rank、microbatch、组内第 k 个 → 步内样本序号。"""
    return micro * dp * mbs + rank * mbs + k


def maxtext_row(device: int, micro: int, k: int, mbs: int, ga: int) -> int:
    """MaxText：设备、microbatch、组内第 k 个 → 全局批里的行号（单 host、设备沿批维连续排）。"""
    return (device * mbs + k) * ga + micro


def maxtext_loaded_slot(row: int, gbs: int, hosts: int = 1) -> int:
    """MaxText 读进来的全局批第 row 行是步内第几个样本（多 host 时按 host 交错）。"""
    if gbs % hosts:
        raise ValueError(f"全局批 {gbs} 不能被 host 数 {hosts} 整除")
    local = gbs // hosts
    return (row % local) * hosts + row // local


def maxtext_micro_split(batch: np.ndarray, ga: int) -> np.ndarray:
    """照抄 MaxText 梯度累积的切法：[gbs, ...] → [ga, gbs//ga, ...]，第 m 个就是第 m 个 microbatch。"""
    gbs = batch.shape[0]
    if gbs % ga:
        raise ValueError(f"全局批 {gbs} 不能被 ga={ga} 整除")
    return np.swapaxes(batch.reshape((gbs // ga, ga) + batch.shape[1:]), 0, 1)


def maxtext_perm_for_megatron(dp: int, mbs: int, ga: int, hosts: int = 1) -> np.ndarray:
    """返回 perm（长 gbs）：全局批按 batch[perm] 重排后，MaxText 设备 d / microbatch m 拿到 Megatron rank d / microbatch m 的样本。"""
    gbs = dp * mbs * ga
    row_of_slot = np.empty(gbs, dtype=np.int64)
    for g in range(gbs):
        row_of_slot[maxtext_loaded_slot(g, gbs, hosts)] = g
    perm = np.empty(gbs, dtype=np.int64)
    for d in range(dp):
        for m in range(ga):
            for k in range(mbs):
                perm[maxtext_row(d, m, k, mbs, ga)] = row_of_slot[megatron_slot(d, m, k, dp, mbs)]
    return perm


def per_device(batch: np.ndarray, dp: int, mbs: int, ga: int) -> dict:
    """MaxText 全局批 → {(设备, microbatch): [mbs, ...]}，按训练时的实际切法。"""
    micro = maxtext_micro_split(batch, ga)
    return {(d, m): micro[m][d * mbs:(d + 1) * mbs] for d in range(dp) for m in range(ga)}
