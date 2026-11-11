"""第 1 步前向的 loss 对 CPU FP64 参照：GPU 与 TPU 各离精确值多远，谁更近（R = |TPU − FP64| ÷ |GPU − FP64|）。

参照来自 `frameworks/megatron/step1_ref.py --forward-only` 每一行打印的 MICRO 记录（行的 loss 和、有效 token 数、MTP 均值），
按训练入口的口径合成全局量：主 loss = Σ loss 和 ÷ Σ 有效 token 数；MTP（MaxText 口径，梯度累积 1 时）=
0.1 × Σ(MTP 均值 × 有效 token 数) ÷ Σ 有效 token 数。GPU、TPU 取各运行 metrics.jsonl 第 1 步的 `lm_loss`、`mtp_loss_mt`。

用法：
    python step1_loss_vs_fp64.py --micro <MICRO 行的 jsonl> --gpu rw1k-a=<运行目录> … --tpu trw-a=<运行目录> --json out.json
"""

from __future__ import annotations

import argparse
import json
import os


def step1(run_dir):
    for line in open(os.path.join(run_dir, "metrics.jsonl")):
        r = json.loads(line)
        if r["step"] == 1:
            return r
    raise SystemExit(f"{run_dir} 没有第 1 步")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", required=True)
    ap.add_argument("--gpu", nargs="+", required=True, help="名字=运行目录；第一个当 GPU 代表")
    ap.add_argument("--tpu", nargs="+", required=True, help="名字=运行目录")
    ap.add_argument("--mtp-factor", type=float, default=0.1)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    rows = []
    for line in open(a.micro):
        line = line.strip()
        if line.startswith("MICRO "):
            line = line[6:]
        if line:
            rows.append(json.loads(line))
    n = sum(r["n_tokens"] for r in rows)
    ref = {
        "rows": len(rows),
        "lm_loss": sum(r["loss_sum"] for r in rows) / n,
        "mtp_loss_mt": a.mtp_factor * sum(r["mtp_mean"] * r["n_tokens"] for r in rows) / n,
    }
    out = {"fp64": ref, "runs": {}, "R": {}}
    for side, specs in (("gpu", a.gpu), ("tpu", a.tpu)):
        for spec in specs:
            name, path = spec.split("=", 1)
            m = step1(path)
            out["runs"][name] = {"side": side}
            for k in ("lm_loss", "mtp_loss_mt"):
                if m.get(k) is not None:
                    out["runs"][name][k] = m[k]
                    out["runs"][name][k + ".rel_vs_fp64"] = (m[k] - ref[k]) / ref[k]
    g = a.gpu[0].split("=", 1)[0]
    for spec in a.tpu:
        t = spec.split("=", 1)[0]
        out["R"][t] = {
            k: abs(out["runs"][t][k + ".rel_vs_fp64"]) / abs(out["runs"][g][k + ".rel_vs_fp64"])
            for k in ("lm_loss", "mtp_loss_mt")
            if k + ".rel_vs_fp64" in out["runs"][t] and k + ".rel_vs_fp64" in out["runs"][g]
        }
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
