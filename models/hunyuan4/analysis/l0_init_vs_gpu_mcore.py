"""L0 跨侧初始权重（T1 起 GPU golden 是 Megatron-Core 程序）：TPU 载入的 MaxText 检查点 → GPU 参数清单的写法，逐项逐位比。

两步，都是全量不抽样：
  ① 「TPU 载入后」= 这份检查点：把检查点反向取回成 HF 张量（verify_ckpt.roundtrip，同 L0 往返用的那套），按 TPU 训练
     入口写清单的规则（hy4_map.gpu_name、FP32、hy4_map.gpu_rank_pieces 分 8 份）复算 sha256，对 TPU 运行
     params_sha/rank*.init.json（train_hy4.py 在载入之后、第 1 步之前写的）逐项比。
  ② 同一组 HF 张量按 GPU 载入的对照表（frameworks/megatron/hy4_weights.py 的 layer_plan / mtp_names，
     gate/up 沿第 0 维拼、路由专家按 EP rank 取连续的 E/EP 个）换成 Megatron 名字，转成 GPU 清单里写的 dtype
     （FP32 → BF16 只在值是 BF16 精确值时做，否则记为不一致），按 GPU 的写法算 sha256（BF16 按 int16 位模式），
     对 GPU 运行 params_sha/rank*.init.json 逐 rank 逐项比（GPU 每个 rank 记完整张量；专家只记本 rank 的）。
     EP rank = rank % EP（Megatron 在 TP = 1 时 EP 组取相邻 rank）。

结果写成 janus results.json 的格式（两条 checkpoint_equal 类检查）。

  JAX_PLATFORMS=cpu <mt-venv>/bin/python l0_init_vs_gpu_mcore.py --ckpt <检查点>/0/items --hf <HF 目录> --stage mini \
      --tpu-run ~/janus-runs/t1zg-a --gpu-run <g3-a 运行目录> --out <结果 json> model_name=hy4-mini
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE.parent / "frameworks" / "megatron"))

import verify_ckpt as vc  # noqa: E402
from hy4_map import gpu_name, gpu_rank_pieces  # noqa: E402


def sha_np(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_gpu(a: np.ndarray) -> str:
    """janus/dump/megatron.py 的 _sha：BF16 按 int16 位模式取字节，其余按原 dtype。"""
    import ml_dtypes
    a = np.ascontiguousarray(a)
    if a.dtype == ml_dtypes.bfloat16:
        a = a.view(np.int16)
    return sha_np(a)


def mcore_tensors(hf: dict, cfg: dict, ep: int, ep_rank: int):
    """HF 张量 → {Megatron 名: (数组, 来源说明)}；对照表取自 GPU 一侧的 hy4_weights。"""
    import hy4_weights as hw
    L = int(cfg["num_hidden_layers"])
    c = SimpleNamespace(num_layers=L, hy4_indexer_types=cfg["indexer_types"],
                        moe_layer_freq=[1 if t == "sparse" else 0 for t in cfg["mlp_layer_types"]],
                        num_moe_experts=int(cfg["n_routed_experts"]), mtp_num_layers=int(cfg.get("num_nextn_predict_layers", 0)))
    out = {}
    for m, h in hw.GLOBAL:
        out[m] = (hf[h], h)
    E = c.num_moe_experts
    n_local = E // ep
    lo = ep_rank * n_local
    for i in range(L):
        pm, ph = f"decoder.layers.{i}.", f"model.layers.{i}."
        for m, h, tf in hw.layer_plan(c, i):
            if tf == "cat":
                out[pm + m] = (np.concatenate([hf[ph + x] for x in h], axis=0), "+".join(ph + x for x in h))
            else:
                out[pm + m] = (hf[ph + h], ph + h)
        if c.moe_layer_freq[i]:
            out[f"buffer.{pm}mlp.router.expert_bias"] = out.pop(pm + "mlp.router.expert_bias")
            for e in range(n_local):
                out[hw.expert_param_name(pm, "linear_fc1", e, True)] = (hf[ph + "mlp.experts.gate_up_proj"][lo + e], f"{ph}mlp.experts.gate_up_proj[{lo + e}]")
                out[hw.expert_param_name(pm, "linear_fc2", e, True)] = (hf[ph + "mlp.experts.down_proj"][lo + e], f"{ph}mlp.experts.down_proj[{lo + e}]")
    if c.mtp_num_layers:
        for m, h, tf in hw.mtp_names(c):
            if tf == "cat":
                out[m] = (np.concatenate([hf[x] for x in h], axis=0), "+".join(h))
            elif tf == "experts":
                for e in range(n_local):
                    out[hw.expert_param_name(m, "linear_fc1", e, True)] = (hf[h + "mlp.experts.gate_up_proj"][lo + e], f"{h}mlp.experts.gate_up_proj[{lo + e}]")
                    out[hw.expert_param_name(m, "linear_fc2", e, True)] = (hf[h + "mlp.experts.down_proj"][lo + e], f"{h}mlp.experts.down_proj[{lo + e}]")
            else:
                out[m] = (hf[h], h)
        k = "mtp.layers.0.mtp_model_layer.mlp.router.expert_bias"
        out["buffer." + k] = out.pop(k)
    return out


def to_dtype(x: np.ndarray, want: str):
    """转成 GPU 清单写的 dtype；FP32 → BF16 不是精确值时返回 None。"""
    import ml_dtypes
    if want == "bfloat16":
        y = x.astype(ml_dtypes.bfloat16)
        return y if np.array_equal(y.astype(np.float32), x.astype(np.float32)) else None
    return np.ascontiguousarray(x.astype(np.dtype(want)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hf", required=True)
    p.add_argument("--stage", default="mini")
    p.add_argument("--tpu-run", required=True)
    p.add_argument("--gpu-run", required=True)
    p.add_argument("--ep", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("maxtext_args", nargs="*")
    a = p.parse_args()
    hfd = vc.HF(a.hf)
    got, _where, _config, dein = vc.roundtrip(a, hfd)
    hf32 = {k: np.asarray(v) for k, v in got.items()}
    cfg = hfd.config
    world = 8

    # ① TPU 载入后的清单 = 检查点复算
    tpu_bad, tpu_n, tpu_only = [], 0, []
    for r in range(world):
        man = json.loads((Path(a.tpu_run) / "params_sha" / f"rank{r}.init.json").read_text())
        mine = {}
        for name, x in hf32.items():
            if name.endswith("e_score_correction_bias"):
                continue
            gn = gpu_name(name)
            y = np.ascontiguousarray(x, np.float32)
            mine[gn] = sha_np(np.ascontiguousarray(gpu_rank_pieces(gn, y)[r]))
        tpu_only += [f"r{r}:{k}" for k in sorted(set(man) ^ set(mine))]
        for k in sorted(set(man) & set(mine)):
            tpu_n += 1
            if man[k]["sha256"] != mine[k]:
                tpu_bad.append(f"r{r}:{k}")
    bufman = json.loads((Path(a.tpu_run) / "params_sha" / "buffers.init.json").read_text())
    buf_bad = [k for k, v in bufman.items() if sha_np(np.ascontiguousarray(hf32[k[3:]], np.float32)) != v["sha256"]]
    tpu_n += len(bufman)
    tpu_bad += buf_bad

    # ② 换成 GPU 的写法，对 GPU 清单
    gpu_bad, gpu_n, gpu_only, inexact, rows = [], 0, [], [], []
    for r in range(world):
        gman = json.loads((Path(a.gpu_run) / "params_sha" / f"rank{r}.init.json").read_text())
        mt = mcore_tensors(hf32, cfg, a.ep, r % a.ep)
        gpu_only += [f"r{r}:{k}" for k in sorted(set(gman) ^ set(mt))]
        for k in sorted(set(gman) & set(mt)):
            x, src = mt[k]
            want = gman[k]["dtype"]
            if list(x.shape) != list(gman[k]["shape"]):
                gpu_bad.append(f"r{r}:{k}（形状 {list(x.shape)} ≠ {gman[k]['shape']}）")
                continue
            y = to_dtype(x, want)
            gpu_n += 1
            if y is None:
                inexact.append(f"r{r}:{k}")
                gpu_bad.append(f"r{r}:{k}（FP32 值不是 BF16 精确值）")
                continue
            h = sha_gpu(y)
            eq = h == gman[k]["sha256"]
            if not eq:
                gpu_bad.append(f"r{r}:{k}")
            if r == 0:
                rows.append({"name": k, "hf": src, "dtype": want, "shape": list(x.shape), "equal": eq})
    st = lambda bad, only: "pass" if not bad and not only else "fail"  # noqa: E731
    res = {
        "schema_version": 1, "stage": "t1-l0-init-vs-gpu-mcore", "model": "hunyuan4-mini",
        "inputs": {"ckpt": "/".join(Path(a.ckpt).parts[-3:]), "hf": Path(a.hf).name, "tpu_run": Path(a.tpu_run).name, "gpu_run": Path(a.gpu_run).name,
                   "checkpoint_rope_interleave": dein, "ep": a.ep, "ep_rank_rule": "rank % ep"},
        "checks": [
            {"id": "L0.init.tpu_loaded_eq_ckpt", "type": "checkpoint_equal", "level": "L0",
             "status": st(tpu_bad, tpu_only), "threshold": {"kind": "exact"},
             "metrics": {"entries": tpu_n, "differ": len(tpu_bad), "only_one_side": len(tpu_only)},
             "message": f"TPU 载入后写的清单（8 份 × 参数 + 路由偏置）{tpu_n} 项对检查点复算：不同 {len(tpu_bad)}、只在一边 {len(tpu_only)}",
             "details_inline": {"differ": tpu_bad[:50], "only_one_side": tpu_only[:50]}},
            {"id": "L0.init.tpu_vs_gpu_mcore", "type": "checkpoint_equal", "level": "L0",
             "status": st(gpu_bad, gpu_only), "threshold": {"kind": "exact"},
             "metrics": {"entries": gpu_n, "ranks": world, "params_per_rank": len(rows), "differ": len(gpu_bad),
                         "only_one_side": len(gpu_only), "bf16_inexact": len(inexact)},
             "message": f"TPU 载入的权重换成 GPU 的名字与 dtype，对 GPU 初始参数清单 8 个 rank 共 {gpu_n} 项：不同 {len(gpu_bad)}、只在一边 {len(gpu_only)}",
             "details_inline": {"differ": gpu_bad[:50], "only_one_side": gpu_only[:50], "rank0": rows}},
        ],
    }
    res["summary"] = {k: sum(c["status"] == k for c in res["checks"]) for k in ("pass", "fail", "inconclusive", "error")}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for c in res["checks"]:
        print(c["status"], c["id"], c["message"])


if __name__ == "__main__":
    main()
