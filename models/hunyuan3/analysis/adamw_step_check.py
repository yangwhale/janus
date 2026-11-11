"""用一个运行目录第 N 步抓下的 main_param / exp_avg / exp_avg_sq（更新后的两个动量）按 Megatron 的 AdamW 公式
重算更新量，与同一步的 main_param_after 比：解耦权重衰减只给二维、名字不含 norm 的参数；偏差修正；ε 加在根号外。
另把衰减掩码反过来算一遍，看哪种更吻合。

  python adamw_step_check.py <运行目录> [--step 1] [--lr 1e-5 --b1 0.9 --b2 0.95 --eps 1e-8 --wd 0.033] [--out json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from janus.store import Run  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--b1", type=float, default=0.9)
    p.add_argument("--b2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--wd", type=float, default=0.033)
    p.add_argument("--out")
    a = p.parse_args(argv)
    r, t = Run.open(a.run), a.step
    ns = set(r.names(t))
    rows = []
    for n in sorted(x for x in ns if x.startswith("main_param.")):
        base = n[len("main_param."):]
        need = [f"main_param_after.{base}", f"exp_avg.{base}", f"exp_avg_sq.{base}"]
        if not all(x in ns for x in need):
            continue
        pv, pa, m, v = (r.tensor(t, x).astype(np.float64) for x in [n] + need)
        upd = (m / (1 - a.b1 ** t)) / (np.sqrt(v / (1 - a.b2 ** t)) + a.eps)
        decay = pv.ndim >= 2 and "norm" not in base
        row = {"name": base, "ndim": pv.ndim, "decay": decay}
        for tag, w in (("rel_err", a.wd if decay else 0.0), ("rel_err_mask_flipped", 0.0 if decay else a.wd)):
            d_exp = -a.lr * (upd + w * pv)
            row[tag] = float(np.linalg.norm((pa - pv) - d_exp) / max(np.linalg.norm(d_exp), 1e-30))
        rows.append(row)
        print(f"{base[:70]:70s} decay={decay!s:5s} rel_err={row['rel_err']:.2e} flipped={row['rel_err_mask_flipped']:.2e}")
    if a.out:
        Path(a.out).write_text(json.dumps({"run": a.run, "step": t, "params": vars(a), "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
