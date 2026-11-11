"""逐个开关单独量（项目负责人 2026-11-08 11:39）：以 T1 现行全开配置为底，每次只关一个开关，量它碰到的算子。

两部分，每个开关一节，按四要素写（比什么、参照、GPU 值、TPU 开、TPU 关，两个 R，TPU 对 GPU 开 / 关）：
- 前向：L1 mini 单算子回放（op_replay_mini.py --sweep，`sweeps/switch_isolation.json`），每个算子吃 GPU 抓到的同一个输入，
  TPU / GPU 输出各对 GPU 一侧的 FP64 重算；比较单元是算子输出（逐算子取最差）；
- 反向：第 1 步梯度（train_hy4.py --diag-step1，不更新参数）。不用 TPU 自己的 FP32 参照（被测一侧产出的参照，项目负责人
  2026-11-08 15:03 的规矩）：只报开 / 关两组 TPU 对 GPU（g2s-a / b / c 三份取中位数）的 rel_L2、GPU 三份之间的 rel_L2（噪声尺子）
  与开关前后梯度是否逐位相同；R 等 CPU 上用 GPU 侧代码跑的 FP64 第 1 步参照。比较单元是参数梯度张量。
数字只从原始输出里取。

  python models/hunyuan4/analysis/switch_isolation.py --runs <TPU 运行目录的上一级> --gpu-grads <g2sgrad 目录> \\
      --fwd <tw-sw-fwd 目录，含各组 cmp.<名字>.json> --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

R_LIMIT = 1.2
NOTES = {
    "M13": "第 1 步的输出头权重与输入都在 BF16 网格上（检查点存的是 BF16 值；输入是最后那个 norm 的 BF16 输出转成 FP32），"
    "highest 与 default 的乘积在 FP32 累加里没有差别，所以第 1 步单算子逐位相同；换上离开网格的 FP32 主权重后才看得到（见 "
    "forward_with_trained_weights）。",
    "M27": "同 M13：第 1 步 weights_proj 的权重在 BF16 网格上，权重先舍成 BF16 是恒等变换，第 1 步逐位相同；换上训练后的权重才看得到。",
    "M8": "同 M13：第 1 步路由权重在 BF16 网格上，第 1 步逐位相同；换上训练后的权重才看得到。",
    "M9": "已核实（results/t1-hlo-probe-tpu.json，TPU tw-hlo-a / tw-hlo-b）：默认写法的优化后 HLO 里有 FP32→BF16→FP32 的往返，"
    "但 TPU 后端在 excess precision 下把它省掉了，结果 99.9997% 逐位等于单次舍入；--xla_allow_excess_precision=false 时默认写法 "
    "99.9987% 逐位等于两次舍入（对 FP64 的 rel_L2 2.33e-3 对单次舍入的 1.67e-3），开关才起作用。整网编译里 y 跨融合边界以 BF16 落盘时"
    "往返不会被省（推断），所以第 1 步的 RMSNorm scale 梯度开关前后不同。",
    "M36": "推断（未单独验证）：开关修的是整网编译时 XLA 在 excess precision 下省掉 FP32→BF16→FP32 往返的情形；单个算子单独编译时"
    "这个往返没有被省掉，所以开关前后逐位相同。",
}
ON_VALUES = {
    "M13": "highest",
    "M26": "custom（moe_routing_maps: device）",
    "M27": "true",
    "M23": "true",
    "M9": "true",
    "M8": "true",
    "M10": "true",
    "M12": "true",
    "M28": "true",
    "M36": "true",
    "M25": "true",
}
# 开关 → (MaxText 键, 关时的取值, 运行名, 前向算子组 {组名: 算子名正则}, 反向张量组 {组名: HF 名正则}, 说明)
SWITCHES = [
    (
        "M13",
        "logits_matmul_precision",
        "default",
        "tw-sw-m13",
        {"输出头 logits": r"^lm_head$"},
        {
            "输出头权重梯度（wgrad）": r"^lm_head\.weight$",
            "输出头的 dgrad 流到的第一处参数（最后的 norm、hc_head）": r"^model\.(norm\.weight|hc_head\.)",
        },
        "highest ↔ default（只开 logits_dot_in_fp32 时 TPU 默认精度只做一遍 BF16）",
    ),
    (
        "M26",
        "remat_policy",
        "full",
        "tw-sw-m26",
        {},
        {"MoE 层参数梯度（第 1–6 层与 MTP 的路由、专家、共享专家）": r"\.mlp\.", "全部参数": r"."},
        "custom（存路由索引）↔ full（反向重算，重选路由）",
    ),
    (
        "M27",
        "indexer_weights_proj_bf16_weight",
        "false",
        "tw-sw-m27",
        {"注意力整块（含索引器 weights_proj）": r"^(\d|mtp0)\.self_attn$"},
        {"索引器 weights_proj 权重梯度": r"indexer\.weights_proj\.weight$"},
        "weights_proj 乘 BF16 舍入后的权重（FP32 输出）↔ 直接乘 FP32 权重",
    ),
    (
        "M23",
        "indexer_fp32_scores",
        "false",
        "tw-sw-m23",
        {"注意力整块（含索引器打分）": r"^(\d|mtp0)\.self_attn$"},
        {"索引器参数梯度": r"\.indexer\."},
        "索引器打分与 KD 教师 q·k 真 FP32 ↔ 一遍 BF16",
    ),
    (
        "M9",
        "rmsnorm_single_rounding",
        "false",
        "tw-sw-m9",
        {"RMSNorm 输出": r"(layernorm|norm)$"},
        {"RMSNorm scale 梯度": r"(layernorm|norm)\.weight$"},
        "FP32 里乘完 scale 只舍一次 ↔ 两次",
    ),
    (
        "M8",
        "routed_gate_bf16_rounding",
        "false",
        "tw-sw-m8",
        {"路由 logits": r"mlp\.gate\.logits$", "MoE 整块（自选路由）": r"mlp\(own routing\)$"},
        {"路由权重梯度": r"\.mlp\.gate\.weight$"},
        "路由 GateLogit 权重先舍成 BF16 ↔ FP32 权重",
    ),
    (
        "M10",
        "mlp_activation_round_per_op",
        "false",
        "tw-sw-m10",
        {"稠密 MLP": r"^0\.mlp$", "共享专家": r"shared_experts$", "路由专家": r"routed_experts$"},
        {"MLP / 专家权重梯度": r"\.mlp\.(gate_proj|up_proj|down_proj|experts|shared_experts)"},
        "SwiGLU：silu 舍一次、乘积再舍一次 ↔ 不按算子舍",
    ),
    (
        "M12",
        "dense_fp32_accumulation",
        "false",
        "tw-sw-m12",
        {
            "输出门投影": r"gate_proj$",
            "注意力整块": r"^(\d|mtp0)\.self_attn$",
            "稠密 MLP": r"^0\.mlp$",
            "共享专家": r"shared_experts$",
            "MTP eh_proj": r"eh_proj$",
        },
        {"线性层权重梯度（全部 .weight，除 norm）": r"(proj|lm_head|experts|eh_proj).*weight$"},
        "DenseGeneral：FP32 累加、输出舍一次 ↔ 默认",
    ),
    (
        "M28",
        "ihc_stream_sum_unrolled",
        "false",
        "tw-sw-m28",
        {"iHC 混合": r"_hc\.out$", "iHC 写回": r"residual$", "hc_head": r"^hc_head$"},
        {"iHC 参数梯度": r"hc_"},
        "iHC 四路流加权和逐路 FP32 乘加 ↔ HIGHEST einsum",
    ),
    (
        "M36",
        "ihc_force_bf16_rounding",
        "false",
        "tw-sw-m36",
        {"iHC 混合": r"_hc\.out$", "iHC 写回": r"residual$", "hc_head": r"^hc_head$"},
        {"iHC 参数梯度": r"hc_"},
        "iHC 分支输入与流 reduce_precision 后舍 BF16 ↔ 可能被 XLA 省掉的 astype 往返",
    ),
    (
        "M25",
        "qk_clip_max_logit_exclude_sink",
        "false",
        "tw-sw-m25",
        {},
        {"全部参数": r"."},
        "QK-Clip 最大 logit 不含 sink ↔ 含（只作统计）",
    ),
]


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def load_grads(d: Path) -> dict:
    from safetensors.numpy import load_file

    return load_file(str(d / "diag_grads.safetensors"))


def bwd_section(groups, on, off, gpu):
    out = {}
    pairs = (("g2s-b", "g2s-a"), ("g2s-c", "g2s-a"), ("g2s-c", "g2s-b"))
    for gname, pat in groups.items():
        rows = []
        for h in sorted(set(on) & set(off) & set(gpu["g2s-a"])):
            if not re.search(pat, h):
                continue
            g0 = gpu["g2s-a"][h]
            n = g0.shape[0]
            t_on, t_off = on[h][:n], off[h][:n]  # 路由专家：GPU 只有 rank 0 本地的几个
            if t_on.shape != g0.shape:
                continue
            rows.append(
                {
                    "tensor": h,
                    "tpu_vs_gpu_on": float(np.median([rel(t_on, gpu[x][h]) for x in gpu])),
                    "tpu_vs_gpu_off": float(np.median([rel(t_off, gpu[x][h]) for x in gpu])),
                    "gpu_vs_gpu": float(np.median([rel(gpu[x][h], gpu[y][h]) for x, y in pairs])),
                    "tpu_on_vs_off": rel(t_on, t_off),
                    "bit_identical_on_off": bool(np.array_equal(t_on.view(np.uint32), t_off.view(np.uint32))),
                    "R": "等 CPU FP64 参照",
                }
            )
        if not rows:
            continue
        worst = max(rows, key=lambda r: max(r["tpu_vs_gpu_on"], r["tpu_vs_gpu_off"]))
        out[gname] = {
            "metric": "第 1 步参数梯度 · TPU 对 GPU 的 rel_L2（GPU 三份取中位数）；gpu_vs_gpu 是 GPU 三份两两之间的中位数（噪声尺子）",
            "n_tensors": len(rows),
            "n_bit_identical_on_off": sum(r["bit_identical_on_off"] for r in rows),
            "worst": worst,
            "median": {
                k: float(np.median([r[k] for r in rows]))
                for k in ("tpu_vs_gpu_on", "tpu_vs_gpu_off", "gpu_vs_gpu", "tpu_on_vs_off")
            },
        }
    return out


def fwd_section(groups, on_cmp, off_cmp):
    out = {}
    if not on_cmp or not off_cmp:
        return out
    A = {o["op"]: o for o in on_cmp["ops"] if "R" in o}
    B = {o["op"]: o for o in off_cmp["ops"] if "R" in o}
    for gname, pat in groups.items():
        ops = [k for k in B if re.search(pat, k) and k in A]
        if not ops:
            continue
        w = max(ops, key=lambda k: max(A[k]["R"], B[k]["R"]))
        out[gname] = {
            "metric": "算子输出 · rel_L2 对 FP64（同一 GPU 输入）；逐算子取最差",
            "n_ops": len(ops),
            "worst_op": w,
            "gpu": A[w]["rel_l2_gpu_vs_fp64"],
            "tpu_on": A[w]["rel_l2_tpu_vs_fp64"],
            "tpu_off": B[w]["rel_l2_tpu_vs_fp64"],
            "R_on": A[w]["R"],
            "R_off": B[w]["R"],
            "tpu_vs_gpu_on": A[w].get("rel_l2_tpu_vs_gpu"),
            "tpu_vs_gpu_off": B[w].get("rel_l2_tpu_vs_gpu"),
            "R_max_over_ops": {"on": max(A[k]["R"] for k in ops), "off": max(B[k]["R"] for k in ops)},
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--gpu-grads", required=True)
    p.add_argument("--fwd", required=True)
    p.add_argument("--base", default="tw-sw-base")
    p.add_argument("--ovr", default=None, help="weight_override_refs.py 的输出（权重替换后的 M13、M27、M8）")
    p.add_argument(
        "--rd", default="tw-sw-rd-custom,tw-sw-rd-full", help="debug_routing_digest 的两次诊断（custom、full）"
    )
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from safetensors.numpy import load_file

    runs = Path(a.runs)
    gpu = {r: load_file(f"{a.gpu_grads}/{r}.safetensors") for r in ("g2s-a", "g2s-b", "g2s-c")}
    base = load_grads(runs / a.base)
    base_json = json.loads((runs / a.base / "diag_step1.json").read_text())
    fwd = Path(a.fwd)
    ovr = json.loads(Path(a.ovr).read_text()) if a.ovr else None
    cmp = {
        n: json.loads((fwd / f"cmp.{n}.json").read_text())
        for n in ["base"] + [s[0] for s in SWITCHES]
        if (fwd / f"cmp.{n}.json").exists()
    }
    sections = []
    for sw, key, off_val, run, fgroups, bgroups, what in SWITCHES:
        sec = {
            "switch": sw,
            "key": key,
            "on_value_in_T1": "（T1 yml 现行值）",
            "off_value": off_val,
            "what": what,
            "tpu_off_run": run,
        }
        if (runs / run / "diag_grads.safetensors").exists():
            off = load_grads(runs / run)
            sec["backward"] = bwd_section(bgroups, base, off, gpu)
            off_json = json.loads((runs / run / "diag_step1.json").read_text())
            sec["loss"] = {"on": base_json["loss"], "off": off_json["loss"]}
            n_eq = sum(np.array_equal(base[h].view(np.uint32), off[h].view(np.uint32)) for h in base if h in off)
            sec["all_grads_bit_identical"] = {"n_identical": int(n_eq), "n_tensors": len(base)}
            if sw == "M26":
                rd = {}
                for tag, rn in zip(("custom", "full"), a.rd.split(",")):
                    f = runs / rn / "diag_step1.json"
                    if f.exists():
                        rd[tag] = json.loads(f.read_text()).get("routing_reselection")
                if rd:
                    sec["routing_reselection_direct"] = {
                        "how": "MaxText debug_routing_digest：每次 top-k 记到主机，反向重算的那次配上它重复的前向，数选得不同的 (token, 专家) 槽位；"
                        "只有 JAX 进程 0 的 4 个设备（一半的 token）",
                        **rd,
                    }
                # 反向重算重选路由的槽位数：没有直接计数；梯度逐位相同时重选必为 0（重选会改变路由专家的梯度）
                same = n_eq == len(base)
                sec["reselected_slots"] = (
                    {
                        "count": 0,
                        "fraction": 0.0,
                        "how": "full 与 custom 的全部参数梯度逐位相同，反向重算选出的路由与前向相同（推断：任何重选都会改变路由专家的梯度）",
                    }
                    if same
                    else {"count": "待查", "how": "梯度不逐位相同；没有直接计数的工具"}
                )
            if sw == "M25":
                sec["qk_max_logit"] = {"on": base_json.get("qk_max_logit"), "off": off_json.get("qk_max_logit")}
        else:
            sec["backward"] = "待跑"
        sec["forward"] = fwd_section(fgroups, cmp.get("base"), cmp.get(sw)) if fgroups else {}
        if ovr and sw in ("M13", "M27", "M8"):
            sec["forward_with_trained_weights"] = {
                "metric": "算子输出 · rel_L2 对 FP64；两种参照：权重取 FP32 主权重 / 同一份权重舍成 BF16",
                "weights": ovr["weights"],
                "inputs": ovr["inputs"],
                "gpu": "无（GPU 的这几个算子用的是第 1 步权重）",
                "ops": {op: {"on": v["base"], "off": v[sw]} for op, v in ovr["ops"].items() if "base" in v and sw in v},
            }
        if sw in NOTES:
            sec["note"] = NOTES[sw]
        sections.append(sec)
    out = {
        "what": "逐个开关单独量：以 T1 现行全开配置（MaxText 3741e818，hy4-mini-t1.yml，同 t1zu-a）为底，每次只关一个开关",
        "findings": [
            "第 1 步的权重都能用 BF16 精确表示（检查点存的就是 BF16 值），只作用在权重一侧的开关（M13 输出头精度、M27 weights_proj 权重舍 BF16、"
            "M8 路由权重舍 BF16）第 1 步必然逐位不变；它们的效果要等 FP32 主权重离开 BF16 网格之后才出现。这三个另用 GPU g2o-a 第 8 步的"
            "FP32 主权重重放（forward_with_trained_weights）。",
            "M13 的输入一侧也在 BF16 网格上：最后那个 norm 的输出是 BF16，转成 FP32 进输出头。",
            "反向不报对 TPU 自己的 FP32 参照的 R（项目负责人 2026-11-08 15:03 的规矩），只报 TPU 对 GPU 与开关前后是否逐位相同。",
        ],
        "reference": {
            "forward": "GPU 一侧在 CPU 上的 FP64 重算（mini-l1/fp64）",
            "backward": "等 CPU FP64 第 1 步参照（不用 TPU 自己的 FP32 参照）；现在只报 TPU 对 GPU",
        },
        "gpu": {"forward": "g7-l1-a 第 1 步 rank 0 micro 0", "backward": "g2s-a / b / c 第 1 步梯度（三份中位数）"},
        "rule": f"前向 R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64) ≤ {R_LIMIT} 过（PROTOCOL.md 第 5 节）；本文件只报，不另判",
        "tpu_on_run": {"forward": "tw-sw-fwd（base 组）", "backward": a.base},
        "sections": sections,
    }
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(out, ensure_ascii=False, indent=1, default=float))
    Path(a.out).write_text(text + "\n")
    for s in sections:
        print(
            s["switch"],
            json.dumps(
                {
                    k: (
                        v
                        if not isinstance(v, dict)
                        else {
                            g: {x: v[g].get(x) for x in ("R_on", "R_off")}
                            if isinstance(v[g], dict) and "R_on" in v[g]
                            else "…"
                            for g in v
                        }
                    )
                    for k, v in s.items()
                    if k in ("forward",)
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
