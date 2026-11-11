"""TPU 训练状态 → HF 命名与布局的张量（参数、Adam 两个动量）。混元 4 的 GPU golden 是 HF 实现（FSDP2），
参数在 GPU 上就是 HF 的名字和形状，所以 TPU 一侧的参数清单、全量抓数都换成 HF 名字，和 golden 同名可比。

反向变换用 models/hunyuan4/model/hy4_map.py（与 L0 逐位核对同一段代码：L0 证明了「MaxText 检查点 → HF」逐位回到原件）。
形状用 make_tiny_hf.hf_shapes（按 HF 配置推出；L0 的 verify_ckpt.py names 逐个核过真实检查点的头）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "model"))

from hy4_map import STAGES, assemble_hf, hf_dtype, iter_hf_sources, mapping_and_hooks, stage_hf_config  # noqa: E402
from make_tiny_hf import hf_shapes  # noqa: E402


class HFView:
    """config：MaxText 配置（已按阶段覆盖）；hf_cfg_full：截好的 HF config.json（4 层）。"""

    def __init__(self, config, hf_cfg_full: dict, stage: str):
        layers, mtp = STAGES[stage]
        self.hf_cfg = stage_hf_config(hf_cfg_full, layers)
        self.mapping, _ = mapping_and_hooks(self.hf_cfg, config)
        self.shapes = hf_shapes(self.hf_cfg, layers, mtp)
        self.rope = int(self.hf_cfg["qk_rope_head_dim"])
        self.dein = bool(getattr(config, "checkpoint_rope_interleave", False))
        self.scan_axis = int(config.param_scan_axis)
        by_hf: dict[str, list] = {}
        for mt_key, i, h in iter_hf_sources(self.mapping):
            by_hf.setdefault(h, []).append((mt_key, i))
        self.by_hf = dict(sorted(by_hf.items()))
        missing = sorted(set(self.shapes) - set(self.by_hf)) + sorted(set(self.by_hf) - set(self.shapes))
        if missing:
            raise ValueError(f"HF 形状表与映射不一致：{missing[:5]}")

    @staticmethod
    def leaf_path(mt_key: str) -> str:
        """分支映射的键（'params-decoder-…'）→ 抽象树路径（'params/decoder/…'）。路径里本身没有 '-'。"""
        return mt_key.replace("-", "/")

    def state_key(self, flat: dict, mt_key: str, kind: str) -> str | None:
        """训练状态里的键。param：'<集合>/model/<路径>'；mu / nu：以 '/mu/<路径>'、'/nu/<路径>' 结尾的那个
        （冻结 indexer 后优化器状态是 multi_transform 的嵌套结构，不写死前缀）。冻住的参数没有动量，返回 None。"""
        col, rest = self.leaf_path(mt_key).split("/", 1)
        if kind == "param":
            return f"{col}/model/{rest}"
        if col != "params":
            return None
        # 冻住的参数在 multi_transform 的状态里是 optax 的 MaskedNode（不是数组），跳过
        hits = [k for k in flat if k.endswith(f"/{kind}/{rest}") and "/opt_state/" in k and hasattr(flat[k], "dtype")]
        if len(hits) > 1:
            raise ValueError(f"{mt_key} 的 {kind} 在状态里有 {len(hits)} 个：{hits}")
        return hits[0] if hits else None

    def is_buffer(self, hf_name: str) -> bool:
        """路由偏置：不是可训练参数（MaxText 的 MoEBiasVar，GPU 上是 buffer），没有 Adam 动量。"""
        return hf_name.endswith("e_score_correction_bias")

    @staticmethod
    def stored_fp32(hf_name: str) -> bool:
        """HF 检查点里存 FP32 的张量（hy4_map.hf_dtype）：参数清单按 FP32 原样，其余按 BF16 舍入。"""
        return hf_dtype(hf_name) == "float32"

    def tensors(self, flat: dict, to_host, kind: str = "param", keep=None):
        """逐个 yield (HF 名, FP32 数组)。keep(hf 名) 为假的跳过（不从设备取）。一次只缓存当前张量用到的叶子。"""
        cache: dict[str, np.ndarray] = {}
        for h, srcs in self.by_hf.items():
            if keep is not None and not keep(h):
                continue
            pieces = {}
            ok = True
            for mt_key, i in srcs:
                sk = self.state_key(flat, mt_key, kind)
                if sk is None:
                    ok = False
                    break
                if sk not in cache:
                    cache.clear() if len(cache) > 4 else None
                    cache[sk] = np.asarray(to_host(flat[sk]))
                full = cache[sk]
                if i is None:
                    pieces[mt_key] = full
                else:
                    ax = 0 if mt_key.startswith("MoEBiasVar") else (self.scan_axis if full.ndim > self.scan_axis else 0)
                    if full.ndim == 0 or full.shape[ax] <= i:
                        raise ValueError(f"{h}：{sk} 形状 {full.shape}（{type(flat[sk])}），取不出第 {i} 层（轴 {ax}）")
                    pieces[mt_key] = np.take(full, i, axis=ax)
            if not ok:
                continue
            yield h, assemble_hf(pieces, self.shapes[h], self.rope, self.dein).astype(np.float32)
