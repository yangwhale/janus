"""把 Megatron 的 TensorBoard 标量转成 janus 的 metrics.jsonl（float32 原值，不截断）。
用法：python tb2jsonl.py <tb 目录> <输出 jsonl>
"""
import json
import math
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MAP = {"lm loss": "lm_loss", "learning-rate": "lr", "grad-norm": "grad_norm",
       "num-zeros": "num_zeros", "params-norm": "param_norm", "loss-scale": "loss_scale"}
ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()
rows = {}
for tag in ea.Tags()["scalars"]:
    name = MAP.get(tag)
    if name is None:
        continue
    for e in ea.Scalars(tag):
        rows.setdefault(e.step, {})[name] = e.value
with open(sys.argv[2], "w") as f:
    for s in sorted(rows):
        rec = {"step": s}
        for k, v in rows[s].items():
            rec[k] = v if math.isfinite(v) else ("nan" if math.isnan(v) else ("inf" if v > 0 else "-inf"))
        f.write(json.dumps(rec) + "\n")
print(len(rows), "steps;", sorted({k for r in rows.values() for k in r}))
