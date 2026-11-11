"""诊断用的单独前向：用转换好的 MaxText 检查点，对 golden 第 1 步 rank 0 第一个 microbatch 的前 T 个 token 跑一次前向，
按 name_map 的激活观测点抓下来，和 golden 第 1 步同名张量的前 T 行比。

这是 docs/reference/dump.md 说的「诊断版」：只有前向、单独编译，不是训练步的生产图；结论只用来核对转换与名字、布局
（错一处转置或拼接，rel_l2 是 O(1)；只差平台舍入时是 BF16 末位的量级），不当作 TPU 对 GPU 的精度结论。
因果注意力 + 不丢 token 的 MoE 下，前 T 个 token 的输出不依赖后面的 token，所以截短不改变比较对象。

  JAX_PLATFORMS=cpu python probe_forward.py --load /data/hy3-4l-maxtext/0/items --golden /data/golden/g1k-a --tokens 256
  # golden 目录需要 step_1/ 下的 input、embed、layers.*、final_norm 等文件（只读要比的那些）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_common  # noqa: E402

from janus import metrics as M  # noqa: E402
from janus.dump import maxtext as jd  # noqa: E402
from janus.store import Run  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--out", default=None, help="把比较结果写成 JSON")
    p.add_argument("--names", default="*", help="只比名字匹配这个通配的张量")
    p.add_argument("maxtext_args", nargs="*")
    a = p.parse_args(argv)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.utils import model_creation_utils

    T = a.tokens
    g = Run.open(a.golden)
    toks = g.tensor(1, "input.tokens")[0, :T].astype(np.int32)[None]
    pos = g.tensor(1, "input.position_ids")[0, :T].astype(np.int32)[None]
    config = mt_common.make_config(["hardware=cpu", "skip_jax_distributed_system=True", "run_name=probe",
                                    "attention=dot_product", "sparse_matmul=False", "megablox=False",
                                    f"max_target_length={T}", "per_device_batch_size=1", "ici_fsdp_parallelism=1",
                                    f"load_parameters_path={a.load}", *a.maxtext_args])
    mesh = mt_common.make_mesh(config)
    model = model_creation_utils.from_pretrained(config, mesh)
    tagged = jd.tag_model(model)
    print(f"打标记的模块 {len(tagged)} 个", flush=True)

    @nnx.jit
    def fwd(m, x, ps):
        with jd.capturing(rows=1):
            logits = m(decoder_input_tokens=x, decoder_positions=ps, decoder_segment_ids=jnp.ones_like(x),
                       enable_dropout=False)
            inter = nnx.pop(m, nnx.Intermediate)
        return logits, inter.to_pure_dict()

    with jax.set_mesh(mesh):
        _, inter = fwd(model, jnp.asarray(toks), jnp.asarray(pos))
    act = {k: np.asarray(v) for k, v in jd.collect_sown(inter).items()}
    nm = mt_common.name_map_for(config)
    got = jd.assemble_activations(act, dict(nm.dims, routed_scaling=float(config.routed_scaling_factor)), rows=1)

    import fnmatch
    rows = []
    for name, x in sorted(got.items()):
        if not fnmatch.fnmatchcase(name, a.names) or name not in g.names(1):
            continue
        ref = g.tensor(1, name)
        if name.endswith(".counts"):
            continue                                          # GPU 计数是整条 4096 × 2 个 microbatch 的，不可比
        if name.endswith(".topk_idx"):
            r = M.topk_agreement(x[:T], ref[:T])
            rows.append({"name": name, "set_frac": r.set_frac})
            continue
        if name.endswith(".probs"):
            ref = ref[:T]
            x = x[:T]
        else:
            ref = ref[:T]                                     # [S, 1, …] 取前 T 行
        rows.append({"name": name, "shape": list(x.shape), "rel_l2": M.rel_l2(x, ref), "cosine": M.cosine(x, ref),
                     "max_abs": M.max_abs(x, ref)[0]})
    for r in rows:
        print("  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    missing = sorted(n for n in g.names(1) if n.split(".")[0] not in ("grad", "param", "main_param", "main_param_after", "exp_avg", "exp_avg_sq", "buffer", "buffer_after",
                      "input", "loss") and not n.endswith(".grad") and n not in got)
    print(f"golden 第 1 步有、这里没抓到的激活：{missing}")
    if a.out:
        Path(a.out).write_text(json.dumps({"tokens": T, "rows": rows, "missing": missing}, indent=1))


if __name__ == "__main__":
    main()
