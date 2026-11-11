"""打印 MaxText 混元 4 的 Muon 分区表：每个参数走 Muon 还是 Adam、Newton-Schulz 在哪个二维矩阵上做。

    JAX_PLATFORMS=cpu python muon_partition.py --out <结果前缀> [--model hy4-tiny] [覆盖 …]

用 MaxText 自己的 `muon_utils.get_muon_weight_dimension_numbers`（optax 的 MuonDimensionNumbers）判每个参数，
不另写规则。表里每行：参数路径、形状、走哪个优化器、归约轴与输出轴、NS 实际作用的矩阵形状（归约轴的乘积 × 输出轴的乘积）、
有几个这样的矩阵（其余轴，如层、专家，是批维，各自独立正交化）。
写 <前缀>.json 与 <前缀>.md。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="hy4-tiny")
    p.add_argument("overrides", nargs="*", help="MaxText 命令行覆盖，如 base_num_decoder_layers=11")
    a = p.parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    from flax import nnx
    from maxtext.configs import pyconfig
    from maxtext.utils import maxtext_utils, model_creation_utils, muon_utils
    from maxtext.utils.globals import MAXTEXT_PKG_DIR

    argv_cfg = [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), f"model_name={a.model}",
                "scan_layers=True", "attention=dot_product", "opt_type=muon", "muon_include_routers=True",
                "skip_jax_distributed_system=True", *a.overrides]
    config = pyconfig.initialize(argv_cfg)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    _, model = model_creation_utils.create_nnx_abstract_model(config, mesh)
    _, params, _ = nnx.split(model, nnx.Param, ...)
    shapes = jax.tree_util.tree_map_with_path(lambda path, x: x, nnx.to_pure_dict(params))
    mdn = nnx.to_pure_dict(muon_utils.get_muon_weight_dimension_numbers(model, config))

    rows = []

    def walk(sh, md, path):
        if isinstance(sh, dict):
            for k in sh:
                walk(sh[k], md.get(k) if isinstance(md, dict) else None, path + (str(k),))
            return
        shape = tuple(getattr(sh, "shape", ()))
        r = {"path": "/".join(path), "shape": list(shape), "dtype": str(getattr(sh, "dtype", "")),
             "numel": int(math.prod(shape)) if shape else 1}
        if md is None:
            r.update(opt="adam")
        else:
            nd = len(shape)
            red = [ax % nd for ax in md.reduction_axis]
            outp = [ax % nd for ax in md.output_axis]
            batch = [i for i in range(nd) if i not in red + outp]
            r.update(opt="muon", reduction_axis=list(md.reduction_axis), output_axis=list(md.output_axis),
                     ns_matrix=[int(math.prod(shape[i] for i in red)), int(math.prod(shape[i] for i in outp))],
                     n_matrices=int(math.prod(shape[i] for i in batch)) if batch else 1,
                     batch_axes=batch)
        rows.append(r)

    walk(shapes, mdn, ())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"model": a.model, "overrides": a.overrides, "scan_layers": True,
               "muon_params": sum(r["numel"] for r in rows if r["opt"] == "muon"),
               "adam_params": sum(r["numel"] for r in rows if r["opt"] == "adam"),
               "n_muon": sum(r["opt"] == "muon" for r in rows), "n_adam": sum(r["opt"] == "adam" for r in rows)}
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    md = [f"# Muon 分区表：{a.model}（scan_layers=True）", "",
          f"由 `muon_partition.py` 写出，覆盖：{' '.join(a.overrides) or '无'}。Muon {summary['n_muon']} 个张量 / "
          f"{summary['muon_params']:,} 个参数；Adam {summary['n_adam']} 个张量 / {summary['adam_params']:,} 个参数。", "",
          "| 参数 | 形状 | 优化器 | 归约轴 → 输出轴 | NS 矩阵 | 个数（批维） |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["opt"] == "muon":
            md.append(f"| `{r['path']}` | {tuple(r['shape'])} | Muon | {tuple(r['reduction_axis'])} → {tuple(r['output_axis'])} "
                      f"| {r['ns_matrix'][0]} × {r['ns_matrix'][1]} | {r['n_matrices']}（{tuple(r['batch_axes'])}） |")
        else:
            md.append(f"| `{r['path']}` | {tuple(r['shape'])} | Adam | — | — | — |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
