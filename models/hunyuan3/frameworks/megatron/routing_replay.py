"""整层激活重算时，反向里的重算沿用首次前向选中的专家（路由回放），不改 Megatron 源码。

为什么要它：整层重算（recompute_granularity=full）下，首次前向在 no_grad 里跑、只算 loss，反向前再建图重跑一遍。
不回放时重算会重跑 top-k：Megatron 的 `torch.topk(..., sorted=torch.is_grad_enabled())` 在首次前向（no_grad）里不排序、在重算里排序，
选中集合即使相同，顺序也不同，top-k 权重归一化的求和顺序跟着变，probs 差 1 ULP 量级，重算出的专家输出与首次前向不逐位
（第一阶段 `phase1-remat`：3 个 MoE 层的专家输出、MLP 输出、层输出不同）；分数打平或贴近时选中集合本身也可能变。
回放把首次前向的下标连同顺序交给重算，重算与首次前向逐位相同（03 `remat-replay`）。
TPU 侧（`hy3-4layers.yml`）用 `remat_policy=custom` + `moe_routing_maps=device`，top-k 专家下标跨重算边界保存，反向不重跑 top-k。

Megatron-Core 自带路由回放（`megatron/core/transformer/moe/router_replay.py`，配置 `moe_enable_routing_replay`），
但训练循环不会自己切换它的动作：只有推理引擎会设 RECORD。这里按 microbatch 切换：
- 模型前向开始（GPTModel 的 forward pre-hook）：清空上一 microbatch 的记录，动作设 RECORD——首次前向照常选专家并记下 top-k 下标；
- 模型前向结束（forward hook，loss 已算出、反向还没开始）：把记下的下标交给各层（`set_target_indices`），动作设 REPLAY_BACKWARD——
  反向里每层重算时从队列里取出这一份，不再重新选。

回放的只有 top-k 下标。路由概率不回放：重算时由重算出的 logits 重新算 sigmoid 分数，在回放的下标上取值、归一化、乘缩放系数
（`moe_utils.topk_routing_with_score_function` 的 sigmoid + expert_bias 分支：`compute_topk` 的返回值只取下标，
`scores = gather(sigmoid(logits), top_indices)`）。负载均衡偏置的计数只在建图的前向里累加（`router.py` 的 `torch.is_grad_enabled()`），
整层重算下就是重算那一次，回放后它数的就是首次前向实际用的那组专家。

自检：每个 microbatch 开始、每步结束时，各层回放队列必须是空的（反向恰好取走了一份）；不空说明重算没走到回放，直接报错。
`--routing-replay` 要求 `--recompute`（不重算时没有可回放的地方）。
"""

from __future__ import annotations

import logging

log = logging.getLogger("janus.routing_replay")


class RoutingReplay:
    """Megatron-Bridge 回调：on_train_start 在模型上挂两个钩子，on_train_step_end 自检并累计次数。"""

    def __init__(self):
        self.handles = []
        self.microbatches = 0  # 本步进入 RECORD 的 microbatch 数
        self.stats = {"steps": 0, "microbatches": 0, "replays_queued": 0}

    @staticmethod
    def _rr():
        from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

        return RouterReplay, RouterReplayAction

    def _check_drained(self, where):
        RouterReplay, _ = self._rr()
        left = [len(r.replay_backward_list) for r in RouterReplay.global_router_replay_instances]
        if any(left):
            raise RuntimeError(f"路由回放：{where}时还有没被重算取走的下标（每层剩 {left}）——反向重算没有走到回放")

    def on_train_start(self, context):
        RouterReplay, Action = self._rr()
        n = len(RouterReplay.global_router_replay_instances)
        if n == 0:
            raise RuntimeError("路由回放：没有 RouterReplay 实例（moe_enable_routing_replay 没生效？）")
        model = context.model[0]
        model = getattr(model, "module", model)
        while hasattr(model, "module") and not hasattr(model, "decoder"):
            model = model.module

        def pre(module, args):
            self._check_drained("新 microbatch 开始")
            RouterReplay.clear_global_indices()
            RouterReplay.set_global_router_replay_action(Action.RECORD)
            self.microbatches += 1

        def post(module, args, output):
            recorded = RouterReplay.get_recorded_data()
            if any(t is None for t in recorded):
                raise RuntimeError("路由回放：首次前向后有层没记下 top-k 下标")
            for r, t in zip(RouterReplay.global_router_replay_instances, recorded):
                r.set_target_indices(t)
            RouterReplay.set_global_router_replay_action(Action.REPLAY_BACKWARD)
            self.stats["replays_queued"] += len(recorded)

        self.handles.append(model.register_forward_pre_hook(pre))
        self.handles.append(model.register_forward_hook(post))
        log.info("routing replay: %d 个 MoE 层", n)

    def on_train_step_end(self, context):
        self._check_drained("一步结束")
        self.stats["steps"] += 1
        self.stats["microbatches"] += self.microbatches
        self.microbatches = 0

    def on_train_end(self, context):
        import os

        if int(os.environ.get("RANK", 0)) == 0:
            print(f"[janus] routing replay {self.stats}", flush=True)
        RouterReplay, _ = self._rr()
        RouterReplay.clear_global_indices()
        RouterReplay.clear_global_router_replay_action()
        for h in self.handles:
            h.remove()
