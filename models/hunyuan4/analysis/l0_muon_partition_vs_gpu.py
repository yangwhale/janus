"""L0 Muon / Adam 分区逐参数对拍：TPU（MaxText 的 muon_utils 判的分区表，muon_partition.py）对 GPU（Megatron-Core 实际的判法，
GPU 运行目录里的 optimizer_partition.json）。

对法：MaxText 每个参数叶子经权重转换的映射（convert_hf_to_maxtext.build → mapping，同 L0 往返用的那张）得到它装的 HF 张量；
HF 张量再按 GPU 载入的对照表（frameworks/megatron/hy4_weights.py）得到 Megatron 参数名。于是 GPU 清单里的每个参数都能找到
TPU 上装它的叶子与那个叶子走的优化器，逐个比「Muon 还是 Adam」。一个 GPU 参数由几片拼成（linear_fc1 = gate | up）时，
几片的叶子必须同一个优化器；一个叶子装了几个 GPU 参数（层堆叠、专家堆叠）时，每个都比。路由偏置不是参数（GPU 上是 buffer、
TPU 上在 MoEBiasVar 集合），不在两边的分区里。

另外记：两边各自的 Muon / Adam 参数总数（元素个数）；GPU 路由专家只列本 rank 的 4 个（EP 4），数总数时按 16 个专家补齐。

  JAX_PLATFORMS=cpu <mt-venv>/bin/python l0_muon_partition_vs_gpu.py --hf <HF 目录> --tpu-partition <muon_partition.py 的 .json> \
      --gpu-partition <GPU 运行目录>/optimizer_partition.json --out <结果 json> model_name=hy4-mini
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE.parent / "frameworks" / "megatron"))

import verify_ckpt as vc  # noqa: E402
from hy4_map import iter_hf_sources  # noqa: E402


def hf_to_mcore(cfg: dict, ep: int = 4, ep_rank: int = 0) -> dict:
    """{Megatron 名: [HF 名, …]}（专家给 HF 名加 [e] 后缀）；GPU 一侧对照表。"""
    from types import SimpleNamespace

    import hy4_weights as hw
    L = int(cfg["num_hidden_layers"])
    c = SimpleNamespace(num_layers=L, hy4_indexer_types=cfg["indexer_types"],
                        moe_layer_freq=[1 if t == "sparse" else 0 for t in cfg["mlp_layer_types"]],
                        num_moe_experts=int(cfg["n_routed_experts"]), mtp_num_layers=int(cfg.get("num_nextn_predict_layers", 0)))
    out = {m: [h] for m, h in hw.GLOBAL}
    n_local = c.num_moe_experts // ep
    lo = ep_rank * n_local
    for i in range(L):
        pm, ph = f"decoder.layers.{i}.", f"model.layers.{i}."
        for m, h, tf in hw.layer_plan(c, i):
            out[pm + m] = [ph + x for x in h] if tf == "cat" else [ph + h]
        if c.moe_layer_freq[i]:
            for e in range(n_local):
                out[hw.expert_param_name(pm, "linear_fc1", e, True)] = [f"{ph}mlp.experts.gate_up_proj[{lo + e}]"]
                out[hw.expert_param_name(pm, "linear_fc2", e, True)] = [f"{ph}mlp.experts.down_proj[{lo + e}]"]
    if c.mtp_num_layers:
        for m, h, tf in hw.mtp_names(c):
            if tf == "cat":
                out[m] = list(h)
            elif tf == "experts":
                for e in range(n_local):
                    out[hw.expert_param_name(m, "linear_fc1", e, True)] = [f"{h}mlp.experts.gate_up_proj[{lo + e}]"]
                    out[hw.expert_param_name(m, "linear_fc2", e, True)] = [f"{h}mlp.experts.down_proj[{lo + e}]"]
            else:
                out[m] = [h]
    return {k: v for k, v in out.items() if not k.endswith("router.expert_bias")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf", required=True)
    p.add_argument("--stage", default="mini")
    p.add_argument("--tpu-partition", required=True)
    p.add_argument("--gpu-partition", required=True)
    p.add_argument("--ep", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("maxtext_args", nargs="*")
    a = p.parse_args()
    hfd = vc.HF(a.hf)
    _config, abstract, mapping, _, _hf_cfg, slash_of = vc._mt_setup(a)
    tpu = json.loads(Path(a.tpu_partition).read_text())
    tpu_opt = {r["path"]: r for r in tpu["rows"]}
    # HF 名 → (MaxText 叶子路径, 优化器)；专家张量整块在一个叶子里，[e] 都指向它
    hf_leaf = {}
    for mt_key in abstract:
        path = slash_of[mt_key]
        if not path.startswith("params/"):
            continue                           # MoEBiasVar（路由偏置）不是参数
        leaf = path[len("params/"):]
        if leaf not in tpu_opt:
            raise SystemExit(f"分区表里没有叶子 {leaf}")
        for _k, _i, h in iter_hf_sources({mt_key: mapping[mt_key]}):
            hf_leaf.setdefault(h, set()).add(leaf)
    gpu = json.loads(Path(a.gpu_partition).read_text())
    m2h = hf_to_mcore(hfd.config, a.ep, 0)
    rows, bad, missing = [], [], []
    for name, g in sorted(gpu.items()):
        if name not in m2h:
            missing.append(name)
            continue
        leaves = set()
        for h in m2h[name]:
            base = re.sub(r"\[\d+\]$", "", h)
            if base not in hf_leaf:
                missing.append(f"{name}（HF {base} 不在 MaxText 映射里）")
                continue
            leaves |= hf_leaf[base]
        opts = {tpu_opt[x]["opt"] for x in leaves}
        t = opts.pop() if len(opts) == 1 else "混合"
        ok = t == g["opt"]
        rows.append({"gpu": name, "gpu_opt": g["opt"], "tpu_opt": t, "tpu_leaves": sorted(leaves), "equal": ok})
        if not ok:
            bad.append(f"{name}：GPU {g['opt']}，TPU {t}（{sorted(leaves)}）")
    gpu_unmapped_leaves = sorted({x for h, ls in hf_leaf.items() for x in ls} - {x for r in rows for x in r["tpu_leaves"]})
    # 两边元素总数（GPU 专家按 16 个补齐：每个本地专家代表 E/本地数 个）
    E = int(hfd.config["n_routed_experts"])
    tot = {"gpu": {"muon": 0, "adam": 0}, "tpu": {"muon": tpu["summary"]["muon_params"], "adam": tpu["summary"]["adam_params"]}}
    for name, g in gpu.items():
        n = 1
        for s in g["shape"]:
            n *= s
        if ".experts.linear_fc" in name:
            n *= E // (E // a.ep)
        tot["gpu"][g["opt"]] += n
    status = "pass" if not bad and not missing and rows else "fail"
    res = {"schema_version": 1, "stage": "t1-l0-muon-partition", "model": "hunyuan4-mini",
           "inputs": {"tpu_partition": Path(a.tpu_partition).name, "tpu_partition_overrides": tpu["summary"]["overrides"],
                      "gpu_partition": "/".join(Path(a.gpu_partition).parts[-2:]), "ep": a.ep},
           "checks": [{"id": "L0.optim.partition", "type": "config_diff", "level": "L0", "status": status,
                       "threshold": {"kind": "exact"},
                       "metrics": {"gpu_params": len(gpu), "compared": len(rows), "differ": len(bad), "missing": len(missing),
                                   "gpu_muon": sum(r["gpu_opt"] == "muon" for r in rows),
                                   "gpu_adam": sum(r["gpu_opt"] == "adam" for r in rows),
                                   "numel": tot, "tpu_leaves_not_in_gpu_list": len(gpu_unmapped_leaves)},
                       "message": f"GPU optimizer_partition.json 的 {len(gpu)} 个参数逐个找到 TPU 上装它的叶子：优化器不同 {len(bad)}、对不上 {len(missing)}；"
                                  f"元素总数 Muon GPU {tot['gpu']['muon']:,} / TPU {tot['tpu']['muon']:,}，Adam GPU {tot['gpu']['adam']:,} / TPU {tot['tpu']['adam']:,}",
                       "details_inline": {"differ": bad, "missing": missing, "tpu_leaves_not_in_gpu_list": gpu_unmapped_leaves, "rows": rows}}]}
    res["summary"] = {k: sum(c["status"] == k for c in res["checks"]) for k in ("pass", "fail", "inconclusive", "error")}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(res["checks"][0]["status"], res["checks"][0]["message"])
    for b in (bad + missing)[:20]:
        print("  ", b)
    print("TPU 叶子没被 GPU 清单引用：", gpu_unmapped_leaves[:10])


if __name__ == "__main__":
    main()
