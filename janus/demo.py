"""`janus demo`：在本地造一组小的合成运行目录和一份检查项目录，不要 GPU / TPU、不要任何外部数据。

用来几秒钟内看懂 janus 怎么判：6 份参照运行（side 都是 golden）两两之间的差（15 对）当噪声参照，
其中两份列为 golden 给 L4 的多 golden loss 对比；两份被测运行——

- ``t-good``：只带和 golden 同量级的随机差，应当全部 pass；
- ``t-bad``：埋了一个系统错（第 1 层输出整体放大 0.3%，loss 从第 2 步起整体偏高 0.05%），
  应当只在 ``L2.layer1.fwd`` 与 ``L4.loss`` 报 fail，第 0 层仍然 pass——这就是逐层比较能定位的意思。

用法::

    janus demo /tmp/janus-demo
    janus run /tmp/janus-demo/catalog.yaml --out /tmp/janus-demo/out-good
    janus run /tmp/janus-demo/catalog.yaml --out /tmp/janus-demo/out-bad --test t-bad

数据格式与真实运行完全相同（docs/reference/dump-format.md），只是张量很小、模型是假的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from janus.store import RunWriter

# 6 份 → 15 对噪声参照（docs/reference/judging.md「参照要几对」的建议值）
GOLDEN = ["g-1", "g-2", "g-3", "g-4", "g-5", "g-6"]
STEPS = 100
NOISE = 1e-4  # 每次运行相对 golden 的随机差（模拟 GPU 浮点原子加顺序不同带来的差）
FAULT_ACT = 3e-3  # t-bad：第 1 层输出整体放大的比例
FAULT_LOSS = 5e-4  # t-bad：loss 从第 2 步起整体偏高的比例

META = {
    "schema_version": 1,
    "framework": "demo",
    "model": "demo-2layers",
    "code": {"repo": "janus", "commit": "demo"},
    "image": "demo@sha256:0",
    "versions": {},
    "hardware": {"accelerator": "cpu", "count": 1, "topology": "-"},
    "parallelism": {"dp": 1, "fsdp": 1, "tp": 1, "ep": 1, "cp": 1, "pp": 1},
    "config": {"path": "-", "sha256": "0"},
    "checkpoint": {"path": "-", "manifest_sha256": "0"},
    "data": {"manifest": "-", "manifest_sha256": "0", "seed": 0},
    "steps": {"first": 1, "last": STEPS, "convention": "golden_step = step"},
    "determinism": {},
    "dtypes": {},
    "created_at": "2026-11-04T00:00:00+08:00",
}


def _write_run(root: Path, name: str, side: str, seed: int, fault: bool = False) -> None:
    base = np.random.default_rng(0)  # 所有运行共用的「真值」：同一份权重、同一批输入
    weight = base.standard_normal((16, 16)).astype(np.float32)
    acts = [base.standard_normal((32, 16)).astype(np.float32) for _ in range(2)]
    grads = [base.standard_normal((16, 16)).astype(np.float32) * 0.01 for _ in range(2)]
    rng = np.random.default_rng(seed)  # 每次运行自己的随机差

    def jitter(x: np.ndarray) -> np.ndarray:
        return (x * (1 + NOISE * rng.standard_normal(x.shape))).astype(np.float32)

    w = RunWriter(root / name, {**META, "run_name": name, "side": side})
    for s in range(1, STEPS + 1):
        loss = 3.0 * np.exp(-s / 80) + 1.0
        if s > 1:  # 第 1 步前向确定，两边逐位相同；第 2 步起才有随机差
            loss *= 1 + NOISE * rng.standard_normal()
            if fault:
                loss *= 1 + FAULT_LOSS
        w.log_metrics(s, lm_loss=loss, lr=1e-4)
    w.write_tensors(1, "params", {"param.layers.0.w": weight}, kind="param", point="before_update")
    out = [jitter(a) for a in acts]
    if fault:
        out[1] = (out[1] * (1 + FAULT_ACT)).astype(np.float32)
    w.write_tensors(1, "layers", {f"layers.{i}.out": a for i, a in enumerate(out)}, kind="activation", point="forward")
    w.write_tensors(
        1, "grads", {f"grad.layers.{i}.w": jitter(g) for i, g in enumerate(grads)}, kind="grad", point="backward"
    )
    w.done()


def _catalog(runs_root: Path) -> dict:
    pairs = [[a, b] for i, a in enumerate(GOLDEN) for b in GOLDEN[i + 1 :]]
    tensor = {"kind": "tensor", "step": 1}
    return {
        "schema_version": 1,
        "model": "demo-2layers",
        "stage": "demo",
        "description": f"janus demo 造的合成运行：{len(GOLDEN)} 份 golden（{len(pairs)} 对噪声参照），被测 t-good / t-bad",
        "runs": {"test": "t-good", "golden": GOLDEN[:2], "noise": pairs, "root": str(runs_root)},
        "checks": [
            {
                "id": "L0.weights",
                "type": "bitwise_equal",
                "level": "L0",
                "select": {**tensor, "name": "param.*"},
                "threshold": "exact",
            },
            {
                "id": "L2.layer{i}.fwd",
                "type": "tensor_tolerance",
                "level": "L2",
                "foreach": {"i": [0, 1]},
                "metrics": ["rel_l2", "cosine", "norm_ratio"],
                "select": {**tensor, "name": "layers.{i}.out"},
                "threshold": "calibrated",
            },
            {
                "id": "L3.step1.grads",
                "type": "tensor_tolerance",
                "level": "L3",
                "metrics": ["rel_l2", "cosine", "norm_ratio"],
                "select": {**tensor, "name": "grad.*"},
                "threshold": "calibrated",
            },
            {
                "id": "L3.step1.coverage",
                "type": "coverage",
                "level": "L3",
                "select": {**tensor, "name": "*"},
            },
            {
                "id": "L4.loss",
                "type": "multi_golden",
                "level": "L4",
                "select": {"kind": "metrics", "field": "lm_loss", "window": f"1:{STEPS + 1}"},
                "threshold": "calibrated",
            },
        ],
    }


def make_demo(out: Path) -> Path:
    """在 out 下写 runs/<运行名>/ 与 catalog.yaml，返回目录文件路径。out 已有 demo 时报错，不覆盖。"""
    out = Path(out).resolve()
    runs = out / "runs"
    if runs.exists():
        raise FileExistsError(f"{runs} 已存在；换一个目录，或先删掉它")
    runs.mkdir(parents=True)
    for i, name in enumerate(GOLDEN, 1):
        _write_run(runs, name, f"golden-{i}", seed=i)
    _write_run(runs, "t-good", "tpu", seed=101)
    _write_run(runs, "t-bad", "tpu", seed=102, fault=True)
    cat = out / "catalog.yaml"
    cat.write_text(yaml.safe_dump(_catalog(runs), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="janus demo", description="造一组合成运行目录与检查项目录，演示 janus 怎么判。")
    ap.add_argument("out", type=Path, help="输出目录（会建 runs/ 与 catalog.yaml）")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        cat = make_demo(args.out)
    except FileExistsError as e:
        print(f"janus demo：{e}", file=sys.stderr)
        return 2
    print(f"已写 {cat}（运行：{'、'.join(GOLDEN)}、t-good、t-bad）。接着跑：")
    print(f"  janus run {cat} --out {args.out}/out-good                # 期望全部 pass，退出码 0")
    print(f"  janus run {cat} --out {args.out}/out-bad --test t-bad    # 期望 L2.layer1.fwd 与 L4.loss fail，退出码 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
