"""混元 4 对齐总览：一份总览（每列一个尺寸的现行比较）、历代比较与各尺寸的检查清单。

用法：python models/hunyuan4/dashboard/build.py   （在仓库根目录跑）；生成逻辑在 janus/report/overview.py（与混元 3 共用），
本文件只放混元 4 的说明文字、配置，以及把原始 JSON 换成检查的几个函数（见下）。
产出：models/hunyuan4/dashboard/dashboard.md 与 detail.{md,html}（总览与总览细节，骨架 dashboard/items.yaml，只放现行比较）；
      models/hunyuan4/dashboard/history.md（历代与诊断代的 L4、L3，按下面的 HISTORY）；
      每个尺寸的 alignments/<尺寸>/CHECKS.md（检查清单，骨架在 alignments/<尺寸>/dashboard/checks.yaml）；
      mini 另有 alignments/01-mini/gpu-noise/README.md 里的逐级大表（总览的 gpu 列）。各尺寸的 README 不由这里生成（手写）。
      总览引用的每条检查都要在某个尺寸的检查清单里有一行（check_claimed）。

原始 JSON 换成检查（virtual）：单算子回放、第 1 步诊断、确定性等结果没有写成 janus 的 results.json，
下面的函数从原文件现读数字、按写明出处的规则判，生成和 results.json 同格式的检查（证据链接指向原文件）。
判定规则有三种：
- R 比（rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照)）≤ 1.2 过，> 1.2 不过：PROTOCOL.md 第 5 节「L1、L2：对 FP64 的 R 比 ≤ 1.2 过」；
- 第 1 步逐张量梯度（L3）：超出 GPU 第 1 步噪声的张量对参照定责，R 用同一条 1.2 的线（PROTOCOL.md 第 5 节；细则见 _step1_judge）；
- 诊断（只报不判）：判定写成待定 🟡，说明里写实测。
逐位类（确定性、负对照、数据指纹）按原文件里的 ok / 逐位相同 判。
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：import janus

from janus.report.overview import Overview  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALIGN = ROOT / "alignments"
TINY = ALIGN / "00-tiny"
MINI = ALIGN / "01-mini"
RES = MINI / "results"
R_LIMIT = 1.2  # PROTOCOL.md 第 5 节


def _short(t):
    """消息里的科学计数指数不补零：1.53e-04 → 1.53e-4（混元 4 的数值格式 a e-n）。"""
    t = re.sub(r"e([+-])0+(\d)", r"e\1\2", t)
    return re.sub(r"(?<![\w.])-(\d)", r"−\1", t)


R_SOURCE = "PROTOCOL.md 第 5 节：对 FP64 参照（CPU 上算，与 TPU 不同源）的 R 比 ≤ 1.2 过"

# hy4-a-l1-op-tpu-fix.json 的算子 → 检查 id 的后缀、中文名。
L1_OPS = {
    "attn_hc.out": ("ihc_attn", "iHC 混合（注意力前）"),
    "ffn_hc.out": ("ihc_ffn", "iHC 混合（FFN 前）"),
    "input_layernorm": ("input_norm", "输入 RMSNorm"),
    "self_attn.gate_proj": ("gate_proj", "输出门投影"),
    "self_attn(own topk)": ("attn_block", "注意力整块（含索引器、自选 top-k、o_proj）"),
    "post_attention_layernorm": ("post_attn_norm", "FFN 前 RMSNorm"),
    "mlp": ("dense_mlp", "稠密 MLP"),
    "hc_head": ("hc_head", "hc_head"),
    "final_norm": ("final_norm", "最后的 RMSNorm"),
    "lm_head": ("lm_head", "输出头"),
}


def _load(name):
    return json.loads((RES / name).read_text())


def _rcheck(cid, typ, level, r, msg, extra=None, status=None, gpu=None, tpu=None, ref=None):
    """R 比类检查：r ≤ 1.2 过、否则不过。gpu / tpu：两边对参照的 rel_L2（决定 R 的那个用例），格子里和 R 并排写。"""
    st = status or ("pass" if r <= R_LIMIT else "fail")
    src = R_SOURCE
    if gpu is not None and tpu is not None:
        extra = {"rel_l2_gpu_vs_ref": float(gpu), "rel_l2_tpu_vs_ref": float(tpu), **(extra or {})}
    return {
        "id": cid,
        "type": typ,
        "level": level,
        "status": st,
        "threshold": {
            "kind": "numeric",
            "values": {"r_ratio": R_LIMIT},
            "source": src,
            **({"ref": ref} if ref else {}),
        },
        "metrics": {"r_ratio": r, **(extra or {})},
        "message": msg,
    }


def _diag(cid, typ, level, msg, metrics=None):
    return {
        "id": cid,
        "type": typ,
        "level": level,
        "status": "inconclusive",
        "threshold": {},
        "metrics": metrics or {},
        "message": msg,
    }


def _exact(cid, typ, level, ok, unit, n, bad, msg):
    return {
        "id": cid,
        "type": typ,
        "level": level,
        "status": "pass" if ok and bad == 0 else "fail",
        "threshold": {"kind": "exact"},
        "metrics": {unit: n, "differ": bad},
        "message": msg,
    }


def l1_ops():
    """阶段 A（真实权重第 0 层，ga1-a 第 1 步的输入）逐算子隔离回放，注入覆盖全部输入：TPU 对 FP64 ÷ GPU（HF 参照模型）对 FP64。"""
    d = _load("hy4-a-l1-op-tpu-fix.json")
    out = []
    for o in d["ops"]:
        slug, zh = L1_OPS[o["op"]]
        msg = f"{zh}：rel_L2(TPU, FP64) {o['rel_l2_maxtext_vs_fp64']:.3e} ÷ rel_L2(GPU, FP64) {o['rel_l2_gpu_vs_fp64']:.3e}"
        out.append(
            _rcheck(
                f"L1.ops.op.{slug}",
                "op_replay",
                "L1",
                float(o["R"]),
                msg,
                {"cases": 1, "vs_ref_rel_l2": o["rel_l2_vs_gpu"], "vs_ref_cases": 1, "vs_ref_worst": o["op"]},
                gpu=o["rel_l2_gpu_vs_fp64"],
                tpu=o["rel_l2_maxtext_vs_fp64"],
            )
        )
    return out


def kd_attribution():
    """T1 第 1 步第 0 层索引器 KD：KD 的输入对 CPU FP64 的 R（t1z 设置），以及 KD 值本身的诊断。"""
    d = _load("t1-layer0-kd-attribution.json")
    f64 = _load("t1-l1-kd-qkclip-vs-fp64.json")["kd_inputs"]["0"]
    gpu, on = f64["gpu_g1p-kdd"], f64["t1zdiag2"]
    names = ("teacher_q", "teacher_k", "indexer_score")
    rs = {n: on["R_vs_gpu"][n] for n in names}
    worst = max(rs, key=rs.get)
    v = d["kd_value_row0_layer0"]
    lse = v["tpu_logsumexp_minus_fp64_in_f32_ulp"]
    return [
        _rcheck(
            "L1.indexer_kd.inputs",
            "op_replay",
            "L1",
            float(rs[worst]),
            "KD 输入对 CPU FP64（t1z 设置 `t1zdiag2`，第 0 层第一行）：教师 q / 教师 k / 索引器分数 R "
            + " / ".join(f"{rs[n]:.2f}" for n in names)
            + "；rel_L2 对 FP64 GPU / TPU："
            + "，".join(f"{n} {gpu[n]:.2e} / {on[n]:.2e}" for n in names),
            {"cases": len(names)},
            ref="CPU FP64",
            gpu=gpu[worst],
            tpu=on[worst],
        ),
        _diag(
            "L1.indexer_kd.value",
            "op_replay",
            "L1",
            # GPU 一侧按原始字段现算（结果文件的 gpu_logged_rel 四舍五入过头，见其 correction_2026_11_07）
            f"KD 对 FP64：TPU {v['tpu_in_graph_rel']:+.2e}、"
            f"GPU {(v['gpu_logged'] - v['fp64_on_gpu_inputs']) / v['fp64_on_gpu_inputs']:+.2e}；"
            f"TPU 的 lse 低 {abs(lse['mean']):.2f} 格",
            {"cases": 1},
        ),
    ]


# 第 1 步全部参数梯度对 GPU 第 1 步噪声、超出的对参照定责（PROTOCOL.md 第 5 节）。逐张量：
#   r = TPU 对最近一份 GPU 的 rel_L2 ÷ GPU 第 1 步参照各份两两的最大 rel_L2（mini 现行 6 份 15 对，之前各代 3 份 3 对）；
#   R = rel_L2(TPU, CPU FP64) ÷ rel_L2(GPU, CPU FP64)：CPU 上的 FP64 整网重算（*-step1-grad-vs-fp64.json，
#   frameworks/megatron/step1_ref.py，GPU golden 同一套 Megatron-Core 代码，与 TPU 不同源）；
#   *-step1-grad-vs-ref.json 是对同设置 TPU FP32 版的比较，不作判定。
# 超出噪声（r > 1）的张量要定责：R ≤ 1.2 即 TPU 不比 GPU 离精确算术远，算已定责。
#   有 r > 2 且 R > 1.2 的张量 → 不过（待定责）；只有 1 < r ≤ 2 且 R > 1.2 的 → 待定；否则过。
#   GPU 各份逐位相同（N = 0）的张量量不出噪声：两边对参照的 rel_L2 都小于 u = 2^-8 的记「低于分辨率（按下限过）」，
#   其余看 R ≤ 1.2（PROTOCOL.md 第 5 节「分辨率下限」）。
STEP1_SOURCE = "PROTOCOL.md 第 5 节：超出 GPU 第 1 步噪声的张量对参照定责（R ≤ 1.2）"

# 按算子分组的张量名（HF 名字）。主干各层与 MTP 层分开；索引器含 MTP 层的。
STEP1_OPS = {
    "embed": r"^model\.embed_tokens\.",
    "ihc": r"\.hc_(attn|mlp)_layer\.hc_pre\.",
    "input_norm": r"^model\.layers\.\d+\.input_layernorm\.",
    "q_a": r"^model\.layers\.\d+\.self_attn\.q_a_proj\.",
    "q_norm": r"^model\.layers\.\d+\.self_attn\.q_a_layernorm\.",
    "q_b": r"^model\.layers\.\d+\.self_attn\.q_b_proj\.",
    "kv_a": r"^model\.layers\.\d+\.self_attn\.kv_a_proj_with_mqa\.",
    "kv_norm": r"^model\.layers\.\d+\.self_attn\.kv_a_layernorm\.",
    "kv_b": r"^model\.layers\.\d+\.self_attn\.kv_b_proj\.",
    "indexer_wq_b": r"\.self_attn\.indexer\.wq_b\.",
    "indexer_wk": r"\.self_attn\.indexer\.wk\.",
    "indexer_k_norm": r"\.self_attn\.indexer\.k_norm\.",
    "indexer_weights_proj": r"\.self_attn\.indexer\.weights_proj\.",
    "sink": r"^model\.layers\.\d+\.self_attn\.learnable_sink_param",
    "gate": r"^model\.layers\.\d+\.self_attn\.linear_gate\.",
    "o_proj": r"^model\.layers\.\d+\.self_attn\.o_proj\.",
    "post_norm": r"^model\.layers\.\d+\.post_attention_layernorm\.",
    "dense_mlp": r"^model\.layers\.\d+\.mlp\.(gate|up|down)_proj\.",
    "router": r"^model\.layers\.\d+\.mlp\.gate\.weight",
    "experts": r"^model\.layers\.\d+\.mlp\.experts\.",
    "shared": r"^model\.layers\.\d+\.mlp\.shared_experts\.",
    "hc_head": r"^model\.hc_head\.",
    "final_norm": r"^model\.norm\.",
    "lm_head": r"^lm_head\.",
    "mtp_norms": r"^model\.mtp_layers\.\d+\.(enorm|hnorm)\.",
    "mtp_eh_proj": r"^model\.mtp_layers\.\d+\.eh_proj\.",
    "mtp_layer": r"^model\.mtp_layers\.\d+\.(?!enorm|hnorm|eh_proj|self_attn\.indexer)",
}


RES_FLOOR = 2.0**-8  # BF16 单位舍入；PROTOCOL.md 变更记录 2026-11-08「分辨率下限」


def _below_floor(r):
    """GPU 第 1 步噪声 N = 0，且两边对参照的 rel_L2 都小于 u = 2^-8：记「低于分辨率（按下限过）」，不用 R 判。"""
    return r["gpu_noise_max"] == 0 and r["tpu_vs_ref"] < RES_FLOOR and r["gpu_vs_ref"] < RES_FLOOR


def _step1_judge(rows):
    """→ (状态, 超 2N 未定责, 1–2N 未定责, N = 0 且 R > 1.2（低于分辨率的不算）, 超 2N 的, 超 N 的)。"""
    zero = [r for r in rows if r["gpu_noise_max"] == 0 and not _below_floor(r)]
    pos = [(r, r["tpu_vs_gpu_min"] / r["gpu_noise_max"]) for r in rows if r["gpu_noise_max"] > 0]
    over2 = [r for r, x in pos if x > 2]
    over1 = [r for r, x in pos if x > 1]
    un2 = [r for r in over2 if r["R_vs_ref"] > R_LIMIT]
    un1 = [r for r, x in pos if 1 < x <= 2 and r["R_vs_ref"] > R_LIMIT]
    zbad = [r for r in zero if r["R_vs_ref"] > R_LIMIT]
    st = "fail" if un2 or zbad else ("inconclusive" if un1 else "pass")
    return st, un2, un1, zbad, over2, over1


def _fen(rows):
    """GPU 第 1 步参照的份数（每行 gpu_vs_ref_each 的长度）：3 份照旧写「三份」，其余写数字。"""
    n = len(rows[0]["gpu_vs_ref_each"]) if rows and "gpu_vs_ref_each" in rows[0] else 3
    return "三份" if n == 3 else f"{n} 份"


def _names(rs):
    """未定责张量的四件事：超出 N 的倍数（N = 0 写明），两边对参照的 rel_L2，R。"""

    def one(r):
        n = r["gpu_noise_max"]
        over = f"{r['tpu_vs_gpu_min'] / n:.2f} N" if n > 0 else "N = 0"
        return (
            f"`{r['tensor']}`（{over}；rel_L2 对参照 GPU {r['gpu_vs_ref']:.2e}、TPU {r['tpu_vs_ref']:.2e}，"
            f"R {r['R_vs_ref']:.2f}）"
        )

    return "、".join(one(r) for r in rs)


# 另列的张量：{张量名: 出处}。不进未定责计数，消息里写明「另列」。条文见 mini 对齐细节「D11 豁免规则」：超 N 且 R > 1.2
# 的张量一律单列；登记过接受、且接受理由能在被判那一代的数上复核成立的才进这张表（不计入未定责），其余计入。
# 单列的张量挡不挡签字待拍板，所以消息不写「已接受」。
STEP1_ACCEPTED = {
    "model.mtp_layers.0.enorm.weight": "接受理由见对齐细节 D13、PROTOCOL 变更记录 2026-11-09",
}


def step1_vs_ref(fname, tpu, gpu, res=None, accepted=None):
    """第 1 步全部参数梯度：整网一条 + 按算子分组各一条（同一规则，见上）。res：结果目录（默认 mini 的）。
    accepted：接受的张量（见 STEP1_ACCEPTED），R 改记为已定责。"""
    allrows = json.loads(((res or RES) / fname).read_text())["rows"]
    acc = []
    if accepted:
        allrows = [dict(r) for r in allrows]
        for r in allrows:
            if r["tensor"] in accepted and r["R_vs_ref"] > R_LIMIT:
                acc.append(r)
                r["R_vs_ref_raw"], r["R_vs_ref"] = r["R_vs_ref"], 0.0
    # TPU 梯度文件取回损坏的张量（文件里的平方和对不上设备上算的，见 step1_vs_fp64.py）不可判：不进判定，整体至少待定
    bad = [r["tensor"] for r in allrows if r.get("tpu_file_corrupt")]
    rows = [r for r in allrows if not r.get("tpu_file_corrupt")]
    st, un2, un1, zbad, over2, over1 = _step1_judge(rows)
    if bad and st == "pass":
        st = "inconclusive"
    msg = (
        (
            f"TPU 梯度文件 {len(bad)} 个张量取回损坏、不可判（{', '.join(f'`{b}`' for b in bad)}），以下只算其余；"
            if bad
            else ""
        )
        + f"TPU `{tpu}` 对 GPU `{gpu}`：{len(rows)} 个张量，超 N 的 {len(over1)} 个、超 2N 的 {len(over2)} 个；"
        f"超 N 而对参照 R > 1.2（未定责）的 {len(un2) + len(un1)} 个"
        + (f"，其中超 2N 的 {len(un2)} 个：{_names(un2)}" if un2 else "")
        + (f"；1–2N 的 {len(un1)} 个：{_names(un1)}" if un1 else "")
        + (
            "；另 "
            + "、".join(f"`{r['tensor']}`（R {r['R_vs_ref_raw']:.2f}）" for r in acc)
            + f" 另列，不进未定责计数（按 D11 豁免规则：登记过接受、理由在被判那一代的数上复核成立；单列的挡不挡签字待拍板；{'；'.join(accepted[r['tensor']] for r in acc)}）"
            if acc
            else ""
        )
        + f"；GPU {_fen(rows)}逐位相同（N = 0）的 {sum(r['gpu_noise_max'] == 0 for r in rows)} 个张量里"
        + f" {sum(_below_floor(r) for r in rows)} 个低于分辨率（按下限过），其余对参照的 R "
        + ("有超过 1.2 的" if zbad else "都不超过 1.2")
    )
    out = [
        {
            "id": "L3.step1.grads.vs_gpu",
            "type": "tensor_tolerance",
            "level": "L3",
            "status": st,
            "threshold": {"kind": "numeric", "values": {"n_unattributed_over_n": {"max": 0}}, "source": STEP1_SOURCE},
            "metrics": {
                "n_unattributed_over_n": len(un2) + len(un1) + len(zbad),
                "n_unattributed_over_2n": len(un2),
                "n_below_resolution_floor": sum(_below_floor(r) for r in rows),
                "tensors": len(rows),
                "n_tpu_file_corrupt": len(bad),
                "n_accepted": len(acc),
                "n_over_n": len(over1),
            },
            "message": msg,
        }
    ]
    for g, pat in STEP1_OPS.items():
        hit = [r for r in rows if re.search(pat, r["tensor"])]
        gbad = [b for b in bad if re.search(pat, b)]
        if not hit:
            if res is not None:  # 截层模型（T2）没有的结构（如 B 没有 MTP）：这一组不出
                continue
            raise SystemExit(f"{fname} 里没有 {g} 的张量（{pat}）")
        gst, gun2, gun1, gz, gover2, gover1 = _step1_judge(hit)
        wr = max(hit, key=lambda r: r["R_vs_ref"])
        rmax = wr["R_vs_ref"]
        gacc = [r for r in acc if r in hit]
        m = (
            f"{len(hit)} 个张量：超 N 的 {len(gover1)} 个、超 2N 的 {len(gover2)} 个；对参照的 R 最大 {rmax:.2f}"
            f"（`{wr['tensor']}`：rel_L2 对参照 GPU {wr['gpu_vs_ref']:.2e}、TPU {wr['tpu_vs_ref']:.2e}）"
        )
        if gun2 or gun1 or gz:
            m += "；未定责：" + _names(gun2 + gun1 + gz)
        if gacc:
            m += "；另列（按 D11 豁免规则不计入未定责）：" + "、".join(
                f"`{r['tensor']}`（R {r['R_vs_ref_raw']:.2f}，{accepted[r['tensor']]}）" for r in gacc
            )
        if gbad:
            m += f"；另 {len(gbad)} 个 TPU 文件取回损坏、不可判"
            gst = "inconclusive" if gst == "pass" else gst
        out.append(
            {
                "id": f"L3.step1.ref.op.{g}",
                "type": "tensor_tolerance",
                "level": "L3",
                "status": gst,
                "threshold": {"kind": "numeric", "values": {"r_ratio": R_LIMIT}, "source": STEP1_SOURCE},
                "metrics": {
                    "r_ratio": rmax,
                    "rel_l2_gpu_vs_ref": wr["gpu_vs_ref"],
                    "rel_l2_tpu_vs_ref": wr["tpu_vs_ref"],
                    "tensors": len(hit),
                },
                "message": m,
            }
        )
    return out


def ihc_experiments():
    """t1t 第 1 步 iHC 梯度三种 TPU 写法的定责实验（诊断，只报）。"""
    d = _load("t1t-step1-ihc-experiments.json")
    zh = {"t1diag17": "t1t", "t1diag19": "+M28", "t1diag20": "关 excess"}
    parts = []
    for k, v in d["results"].items():
        run = re.search(r"\((t1diag\d+)\)", k).group(1)
        parts.append(f"{zh[run]} {v['n_gt_1.2']}/{v['n']}")
    return [_diag("L3.step1.ihc.experiments", "tensor_tolerance", "L3", "R>1.2：" + "，".join(parts), {"tensors": 45})]


def ihc_post_conditioning():
    """iHC FFN 一侧 post 门控梯度的条件数（GPU g2s-a 第 1 步一条序列；诊断，只报）。"""
    d = _load("t1-ihc-post-grad-conditioning.json")

    def rng(v):
        lo, hi = round(min(v["kappa"])), round(max(v["kappa"]))
        return f"{lo}" if hi - lo <= 1 else f"{lo}–{hi}"

    parts = [f"{k} 层 {rng(v)}" for k, v in d["layers"].items()]
    return [
        _diag(
            "L3.step1.ihc.post_conditioning",
            "tensor_tolerance",
            "L3",
            "κ：" + "，".join(parts),
            {"cases": len(d["layers"])},
        )
    ]


def step1():
    """T1 第 1 步（t1s 一代的五个开关）：全部参数梯度、索引器梯度对 CPU FP64 的 R、grad_norm。"""
    out = step1_vs_ref("t1-step1-grad-vs-ref.json", "t1diag15", "g1s-a/b/c")[:1]
    ig = [r for r in _load("t1s-step1-grad-vs-fp64.json")["rows"] if ".indexer." in r["tensor"]]
    rs = [r["R_vs_ref"] for r in ig]
    wr = max(ig, key=lambda r: r["R_vs_ref"])
    out.append(
        _rcheck(
            "L3.step1.grads.indexer.fp64",
            "tensor_tolerance",
            "L3",
            max(rs),
            f"索引器梯度（wq_b、wk、k_norm、weights_proj，各产出层与 MTP）对 CPU FP64 的 R {min(rs):.2f}–{max(rs):.2f}（t1s 一代 `t1diag15`）；"
            f"最差 `{wr['tensor']}`：rel_L2 对 FP64 GPU {wr['gpu_vs_ref']:.2e}、TPU {wr['tpu_vs_ref']:.2e}",
            {"tensors": len(rs)},
            ref="CPU FP64",
            gpu=wr["gpu_vs_ref"],
            tpu=wr["tpu_vs_ref"],
        )
    )
    dg = _load("t1-step1-diag-tpu.json")
    out.append(
        _diag(
            "L3.step1.grad_norm",
            "scalar_series",
            "L3",
            f"TPU {dg['grad_norm']:.5f}，GPU {dg['gpu_reference']['grad_norm_gpu_g1_runs'].split('（')[0]}（g1-a…h；TPU 开关前配置）",
        )
    )
    return out


def determinism():
    d = _load("hy4-mini-tpu-determinism.json")
    w = d["with_mcore_switches"]
    n = len(d["compared_fields"])
    bad = len(d["fields_differing"]) + len(w["fields_differing"])
    ok = d["bitwise_identical"] and w["bitwise_identical"]
    runs = list(d["runs"]) + list(w["runs"])
    return [
        _exact(
            "DET.tpu.rerun",
            "determinism",
            "L4",
            ok,
            "fields",
            n,
            bad,
            f"TPU 同设置两份 100 步、{n} 个逐步量逐位比（{' / '.join(runs)}，两组各两份）",
        )
    ]


def resume_bitwise():
    """t1-resume-bitwise.json：TPU 存档续跑对不断开逐位（x4-full / x4-s1 / x4-s2，C 口径之前的对齐配置 1000 步）。"""
    d = _load("t1-resume-bitwise.json")
    ps, fs = d["per_step"], d["final_state"]
    n = sum(w["compared_steps"] for w in (ps["steps_1_500"], ps["steps_501_1000"]))
    bad = ps["steps_1_500"]["differ_steps"] + ps["steps_501_1000"]["differ_steps"]
    ok = all(fs["state_sha"].values())
    return [
        _exact(
            "DET.tpu.resume",
            "determinism",
            "L4",
            ok,
            "steps",
            n,
            bad + (0 if ok else 1),
            f"第 500 步存档、新进程换切片续跑到 1000 步，对不断开的同一运行：{n} 步 × {ps['steps_1_500']['fields']} 个逐步量逐位比，"
            f"最终 {fs['state_sha_tensors']} 个状态张量 sha256 {'全同' if ok else '有不同'}（{' / '.join(d['runs'])}）",
        )
    ]


def l0_extras():
    smoke = json.loads((TINY / "results" / "hy4-cpu-smoke.json").read_text())
    neg = next(c for c in smoke["checks"] if c["id"] == "A.negative_control_unpatched_rope_fails")
    txt = (RES / "hy4-a-replay-check-ga-a.txt").read_text()
    m = re.search(r"核对 (\d+) 步、(\d+) 个单元", txt)
    diff = re.search(r"不同 (\d+) 处", txt)
    if not (m and diff):
        raise SystemExit("hy4-a-replay-check-ga-a.txt 的格式变了，读不出单元数与不同处数")
    return {
        "hy4-tiny-smoke": (
            TINY / "results" / "hy4-cpu-smoke.json",
            [
                _exact(
                    f"T0.smoke.{c['id']}",
                    "artifact_check",
                    "L0",
                    bool(c["ok"]),
                    "rules",
                    1,
                    0 if c["ok"] else 1,
                    f"{sum(bool(x['ok']) for x in smoke['checks'])} / {len(smoke['checks'])} 条 ok",
                )
                for c in smoke["checks"]
            ],
        ),
        "hy4-cpu-smoke": (
            TINY / "results" / "hy4-cpu-smoke.json",
            [
                _exact(
                    "L0.ckpt.rope.negative_control",
                    "artifact_check",
                    "L0",
                    bool(neg["ok"]),
                    "rules",
                    1,
                    0,
                    "不去交错的检查点按去交错的读法核，判不一致（期望如此）",
                )
            ],
        ),
        "hy4-a-replay-check-ga-a": (
            RES / "hy4-a-replay-check-ga-a.txt",
            [
                _exact(
                    "L0.data.fingerprint",
                    "data_fingerprint",
                    "L0",
                    txt.rstrip().endswith("OK"),
                    "units",
                    int(m.group(2)),
                    int(diff.group(1)),
                    f"{m.group(1)} 步 × 8 rank × 2 microbatch 的输入指纹对 golden ga-a",
                )
            ],
        ),
        **tiny_qkclip_scan(),
    }


def tiny_qkclip_scan():
    """hy4-tiny11 的 QK-Clip 扫描层检查（qk_clip_scan_check.py，台账 J1 / M1）：每层每头实测 √scale 与按该层最大 logit 算的期望逐位比。
    修前（hy4-tiny11-qk-clip-scan.json）扫描块各层共用一个缩放，修后（-postfix，MaxText b87a55aa）复跑。"""
    out = {}
    for name in ("hy4-tiny11-qk-clip-scan", "hy4-tiny11-qk-clip-scan-postfix"):
        f = ROOT / "results" / f"{name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        n = bad = 0
        for leaf in d["wq_b"]:
            got, exp = leaf["sqrt_scale_per_layer_head"], leaf["expected_sqrt_scale_per_layer_head"]
            if not isinstance(got[0], list):  # 余数块（不扫描）只有一层：[头]
                got, exp = [got], [exp]
            for g, e in zip(got, exp, strict=True):
                for x, y in zip(g, e, strict=True):
                    n += 1
                    bad += x != y
        tag = "修后（MaxText b87a55aa）" if name.endswith("postfix") else "修前"
        out[name] = (
            f,
            [
                _exact(
                    "T0.qkclip.scan_per_layer",
                    "artifact_check",
                    "L0",
                    True,
                    "objects",
                    n,
                    bad,
                    f"{tag}：{len(d['wq_b'])} 个 wq_b 叶子、{n} 个层 × 头的实测 √scale 对期望逐位比，不同 {bad} 个",
                )
            ],
        )
    return out


# ---- TPU 一侧的单算子、优化器、QK-Clip、逐层回放（文件不在时不出这几条）
L1_MINI_GROUPS = [  # (检查 id 后缀, 中文名, 算子名的正则)
    (
        "norm",
        "RMSNorm（输入、FFN 前、最后、MTP 的 enorm / hnorm / final）",
        r"(input_layernorm|post_attention_layernorm|final_norm|enorm|hnorm|final_layernorm)$",
    ),
    ("gate_proj", "输出门投影", r"self_attn\.gate_proj$"),
    ("attn", "注意力整块（自选 top-k）", r"\.self_attn$"),
    ("dense_mlp", "稠密 MLP（第 0 层）", r"^0\.mlp$"),
    ("moe", "MoE 整块（自选路由，含路由专家的 clamp 与共享专家）", r"mlp\(own routing\)$"),
    ("router_logits", "路由 logits（FP32）", r"mlp\.gate\.logits$"),
    ("shared_expert", "共享专家", r"mlp\.shared_experts$"),
    ("routed_experts", "路由专家（MoE 整块减共享专家）", r"mlp\.routed_experts$"),
    ("ihc_mix", "iHC 混合（4 路流 → 分支输入）", r"(attn_hc|ffn_hc)\.out$"),
    ("ihc_writeback", "iHC 写回 / MTP 残差（分支输出换成 GPU 的）", r"(attn_residual|ffn_residual)$"),
    ("hc_head", "hc_head", r"^hc_head$"),
    ("eh_proj", "MTP 的 eh_proj", r"eh_proj$"),
    ("lm_head", "FP32 输出头", r"^lm_head$"),
    ("cross_entropy", "交叉熵（GPU 的 logits 进 MaxText 的交叉熵）", r"^cross_entropy$"),
    ("embed", "embedding", r"^embed$"),
]


def tpu_elementwise():
    """tpu-elementwise-ulp.json 里已是 janus op_replay 检查的格式（判定按每条的 threshold），原样用。"""
    return _load("tpu-elementwise-ulp.json")["checks"]


def tpu_l1_mini(fname="t1-l1-op-tpu.json", prefix="L1.mini"):
    """t1-l1-op-tpu*.json：hy4-mini 逐算子 R（GPU 分母 Megatron-Core + TE），按算子类各取最差的一层。"""
    d = _load(fname)
    out = []
    for slug, zh, pat in L1_MINI_GROUPS:
        rows = [o for o in d["ops"] if re.search(pat, o["op"]) and "R" in o]
        if not rows:
            continue
        w = max(rows, key=lambda o: o["R"])
        out.append(
            _rcheck(
                f"{prefix}.{slug}",
                "op_replay",
                "L1",
                float(w["R"]),
                f"{zh}：最差 {w['op']}，rel_L2(TPU, FP64) {w['rel_l2_tpu_vs_fp64']:.3e} ÷ rel_L2(GPU, FP64) {w['rel_l2_gpu_vs_fp64']:.3e}",
                {"cases": len(rows)},
                gpu=w["rel_l2_gpu_vs_fp64"],
                tpu=w["rel_l2_tpu_vs_fp64"],
            )
        )
    return out


def tpu_optim(fname="t1-optim-replay-tpu.json", names=None, prefix="L1.optim"):
    """t1-optim-replay-tpu*.json：优化器连续多步回放，各变体取逐步逐张量最差的 R。"""
    d = _load(fname)
    out = []
    names = names or {"tpu_off": "配置原样", "tpu_ns_bf16": "再开 muon_mcore_ns_bf16"}
    for arm, zh in names.items():
        sm = d["summary"].get(arm)
        if not sm:
            continue
        n = sum(1 for r in d["rows"] if f"R_{arm}" in r)
        wr = next(r for r in d["rows"] if r["step"] == sm["worst"]["step"] and r["name"] == sm["worst"]["name"])
        out.append(
            _rcheck(
                f"{prefix}.{arm}",
                "optimizer_step",
                "L1",
                float(sm["R_max"]),
                f"{zh}：第 {sm['worst']['step']} 步 {sm['worst']['name']}；对 GPU 的 rel_L2 最大 Muon {sm['rel_l2_vs_gpu_muon_max']:.2e}、"
                f"Adam {sm['rel_l2_vs_gpu_adam_max']:.2e}；最差那一格对 FP64 的 rel_L2：GPU {wr['rel_l2_gpu_vs_fp64']:.2e}、TPU {wr[f'rel_l2_{arm}_vs_fp64']:.2e}",
                {"cases": n},
                gpu=wr["rel_l2_gpu_vs_fp64"],
                tpu=wr[f"rel_l2_{arm}_vs_fp64"],
            )
        )
    return out


def tpu_qk_clip():
    """t1-qk-clip-tpu.json：低阈值 QK-Clip。裁剪算法（残差 ÷ 裁剪改变量，过线 1e-3）、第 1 步逐头最大 logit 的 R、两条只报。"""
    d = _load("t1-qk-clip-tpu.json")
    c = d["checks"]
    f, h, b = c["formula"], c["clipped_heads"], c["gpu_bf16_separate_clip"]
    w = f["worst"]
    return [
        {
            "id": "L1.qkclip.formula",
            "type": "artifact_check",
            "level": "L1",
            "status": f["verdict"],
            "threshold": {
                "kind": "numeric",
                "values": {"resid_over_clip": 1e-3},
                "source": "analysis/qk_clip_compare.py 开头：推断（一次更新约 1e-5、裁剪改变量约 6e-4，比值约 1e-4，取 10 倍余量）",
            },
            "metrics": {"resid_over_clip": w["tpu_resid_over_clip"], "cases": len(d["formula_rows"])},
            "message": f"TPU 裁剪后 = GPU 裁剪前 × f(η_TPU)：最差第 {w['step']} 步第 {w['layer']} 层 {w['proj']}，残差 ÷ 裁剪改变量 "
            f"{w['tpu_resid_over_clip']:.2e}（GPU 自己 {w['gpu_resid_over_clip']:.1e}）",
        },
        _diag(
            "L1.qkclip.heads",
            "op_replay",
            "L1",
            f"30 步被裁的头：第 1 步 TPU / GPU-a / GPU-b {h['step1_clipped'][0]} / {h['step1_clipped'][1]} / {h['step1_clipped'][2]}；"
            f"逐步逐头集合差合计 TPU 对 a {h['total_set_diff_tpu_vs_a']}、b 对 a {h['total_set_diff_b_vs_a']}",
            {"cases": len(d["per_step"])},
        ),
        _diag(
            "L1.qkclip.gpu_bf16_copy",
            "op_replay",
            "L1",
            f"GPU 另裁 BF16 模型权重：与 bf16(裁剪后主权重) 最多差 {b['max_bf16_ulp']:.0f} 个 BF16 ulp，被裁行里至多 "
            f"{b['max_frac_differ_in_clipped_rows']:.1%} 的元素不同；30 步 lm_loss 的 MARE TPU 对 a {b['lm_loss_mare_tpu_vs_a']:.2e}、"
            f"b 对 a {b['lm_loss_mare_b_vs_a']:.2e}",
            {"cases": len(d["gpu_bf16_separate_clip_rows"])},
        ),
    ]


def tpu_l2():
    """逐层隔离回放（TPU tw-l2-g），输出与输入梯度各取最差一层的 R，参照是 CPU FP64 单层重算（t1-l2-layer-vs-fp64.json）。"""
    d = _load("t1-l2-layer-vs-fp64.json")
    out = []
    for q, zh in (("out", "前向输出"), ("in_grad", "反向：层输入的梯度")):
        rows = [r for r in d["rows"] if f"{q}.R" in r]
        if not rows:
            continue
        w = max(rows, key=lambda r: r[f"{q}.R"])
        out.append(
            _rcheck(
                f"L2.layer.{q}",
                "op_replay",
                "L2",
                float(w[f"{q}.R"]),
                f"{zh}：最差第 {w['layer']} 层，rel_L2(TPU, FP64) {w[f'{q}.tpu_vs_fp64']:.2e} ÷ rel_L2(GPU, FP64) "
                f"{w[f'{q}.gpu_vs_fp64']:.2e}",
                {"cases": len(rows)},
                ref="CPU FP64",
                gpu=w[f"{q}.gpu_vs_fp64"],
                tpu=w[f"{q}.tpu_vs_fp64"],
            )
        )
    return out


def tpu_transcendental():
    """t1-step1-transcendental-accuracy.json：开关 transcendental_accuracy_highest 关 / 开的第 1 步诊断（只报）。"""
    d = _load("t1-step1-transcendental-accuracy.json")
    kd, g = d["kd_row0_layer0"], d["step1_grads"]
    return [
        _diag(
            "L1.transcendental.kd_row0",
            "op_replay",
            "L1",
            f"第 0 层第 0 行 KD 对 FP64：开关关 {kd['off']['kd_row0_rel']:+.2e}、开 {kd['on']['kd_row0_rel']:+.2e}",
            {"cases": 2},
        ),
        _diag(
            "L3.step1.transcendental",
            "tensor_tolerance",
            "L3",
            f"第 1 步梯度 R > 1.2 的张量：关 {g['off']['n_R_gt_1.2']}、开 {g['on']['n_R_gt_1.2']}；第 3 层 FFN hc_base 的 R "
            f"{g['off']['focus_layer3_ffn_hc_base']['R']:.2f} → {g['on']['focus_layer3_ffn_hc_base']['R']:.2f}",
            {"cases": g["on"]["n_tensors"]},
        ),
    ]


def tpu_tasks():
    files = {
        "tpu-elementwise-ulp": tpu_elementwise,
        "t1-l1-op-tpu": tpu_l1_mini,
        "t1-l1-op-tpu-r2": lambda: tpu_l1_mini("t1-l1-op-tpu-r2.json", "L1.mini2"),
        "t1-l1-op-tpu-r3": lambda: tpu_l1_mini("t1-l1-op-tpu-r3.json", "L1.mini3"),
        "t1-optim-replay-tpu": tpu_optim,
        "t1-optim-replay-adam-acc-tpu": lambda: tpu_optim(
            "t1-optim-replay-adam-acc-tpu.json",
            {"tpu_off": "配置原样", "tpu_adam_acc": "再开 muon_adam_accurate"},
            "L1.optim_adam",
        ),
        "t1-optim-replay-adam-acc-r2-tpu": lambda: tpu_optim(
            "t1-optim-replay-adam-acc-r2-tpu.json",
            {"tpu_off": "配置原样", "tpu_adam_acc": "再开 muon_adam_accurate（最终版）"},
            "L1.optim_adam_r2",
        ),
        "t1-qk-clip-tpu": tpu_qk_clip,
        "t1-l2-layer-vs-fp64": tpu_l2,
        "t1-step1-transcendental-accuracy": tpu_transcendental,
    }
    return {n: (RES / f"{n}.json", fn()) for n, fn in files.items() if (RES / f"{n}.json").exists()}


def virtual():
    v = {
        "hy4-a-l1-op-tpu-fix": (RES / "hy4-a-l1-op-tpu-fix.json", l1_ops()),
        "t1-layer0-kd-attribution": (RES / "t1-layer0-kd-attribution.json", kd_attribution()),
        "t1-step1-grad-vs-ref": (RES / "t1-step1-grad-vs-ref.json", step1()[:1]),
        "t1s-step1-grad-vs-fp64": (
            RES / "t1s-step1-grad-vs-fp64.json",
            step1_vs_ref("t1s-step1-grad-vs-fp64.json", "t1diag15", "g1s-a/b/c")[:1] + step1()[1:2],
        ),
        "t1-step1-diag-tpu": (RES / "t1-step1-diag-tpu.json", step1()[2:]),
        "t1t-step1-grad-vs-ref": (
            RES / "t1t-step1-grad-vs-ref.json",
            step1_vs_ref("t1t-step1-grad-vs-ref.json", "t1diag17", "g2s-a/b/c"),
        ),
        "t1t-step1-ihc-experiments": (RES / "t1t-step1-ihc-experiments.json", ihc_experiments()),
        "t1-ihc-post-grad-conditioning": (RES / "t1-ihc-post-grad-conditioning.json", ihc_post_conditioning()),
        "t1v-step1-grad-vs-ref": (
            RES / "t1v-step1-grad-vs-ref.json",
            step1_vs_ref("t1v-step1-grad-vs-ref.json", "t1vdiag", "g2s-a/b/c"),
        ),
        "t1z-step1-grad-vs-ref": (
            RES / "t1z-step1-grad-vs-ref.json",
            step1_vs_ref("t1z-step1-grad-vs-ref.json", "t1zdiag2", "g2s-a/b/c"),
        ),
        "t1zg-step1-train-grad-vs-fp64": (
            RES / "t1zg-step1-train-grad-vs-fp64.json",
            step1_vs_ref(
                "t1zg-step1-train-grad-vs-fp64.json", "mz-opt2（训练步梯度）", "g2s-a/b/c", accepted=STEP1_ACCEPTED
            ),
        ),
        "t1zo-step1-train-grad-vs-fp64": (
            RES / "t1zo-step1-train-grad-vs-fp64.json",
            step1_vs_ref(
                "t1zo-step1-train-grad-vs-fp64.json",
                "tz-off-rep（训练步梯度）",
                "g2s-a/b/c",
                accepted=STEP1_ACCEPTED,
            ),
        ),
        "t1zo-step1-train-grad-vs-fp64-n6": (
            RES / "t1zo-step1-train-grad-vs-fp64-n6.json",
            step1_vs_ref(
                "t1zo-step1-train-grad-vs-fp64-n6.json",
                "tz-off-rep（训练步梯度）",
                "g2s-a … g2s-f（6 份）",
                accepted=STEP1_ACCEPTED,
            ),
        ),
        "t1zc-step1-train-grad-vs-fp64-n6": (
            RES / "t1zc-step1-train-grad-vs-fp64-n6.json",
            step1_vs_ref(
                "t1zc-step1-train-grad-vs-fp64-n6.json",
                "s1f-c（C 口径，训练步梯度）",
                "g2s-a … g2s-f（6 份）",
                accepted=STEP1_ACCEPTED,
            ),
        ),
        "t1zg-step1-grad-vs-fp64": (
            RES / "t1zg-step1-grad-vs-fp64.json",
            step1_vs_ref("t1zg-step1-grad-vs-fp64.json", "mz-diag1", "g2s-a/b/c", accepted=STEP1_ACCEPTED),
        ),
        "t1z-step1-grad-vs-fp64": (
            RES / "t1z-step1-grad-vs-fp64.json",
            step1_vs_ref("t1z-step1-grad-vs-fp64.json", "t1zdiag2", "g2s-a/b/c", accepted=STEP1_ACCEPTED),
        ),
        "t1v-step1-grad-vs-fp64": (
            RES / "t1v-step1-grad-vs-fp64.json",
            step1_vs_ref("t1v-step1-grad-vs-fp64.json", "t1vdiag", "g2s-a/b/c"),
        ),
        "t1t-step1-grad-vs-fp64": (
            RES / "t1t-step1-grad-vs-fp64.json",
            step1_vs_ref("t1t-step1-grad-vs-fp64.json", "t1diag17", "g2s-a/b/c"),
        ),
        "hy4-mini-tpu-determinism": (RES / "hy4-mini-tpu-determinism.json", determinism()),
        "t1-resume-bitwise": (RES / "t1-resume-bitwise.json", resume_bitwise()),
        **(
            {
                "mid-tm2-step1-grad-vs-fp64": (
                    ALIGN / "03-mid" / "results" / "mid-tm2-step1-grad-vs-fp64.json",
                    step1_vs_ref(
                        "mid-tm2-step1-grad-vs-fp64.json", "tm2diag", "gms-a/b/c", ALIGN / "03-mid" / "results"
                    ),
                )
            }
            if (ALIGN / "03-mid" / "results" / "mid-tm2-step1-grad-vs-fp64.json").exists()
            else {}
        ),
        **{
            n: (
                ALIGN / "02-real-weight" / "results" / f"{n}.json",
                step1_vs_ref(f"{n}.json", t, g, ALIGN / "02-real-weight" / "results"),
            )
            for n, t, g in (
                ("t2b-step1-grad-vs-fp64", "tb6diag", "gb2s-a/b/c"),
                ("t2d1-step1-grad-vs-fp64", "td6diag", "gd2s-a/b/c"),
            )
            if (ALIGN / "02-real-weight" / "results" / f"{n}.json").exists()
        },
    }
    v.update(l0_extras())
    v.update(tpu_tasks())
    v["t1-route-agree"] = (RES / "t1-route-agree-s1.json", route_agree())
    v["t1-l2-flip-stats-diag"] = (RES / "t1-l2-flip-stats.results.json", flip_stats_diag())
    for _, cs in v.values():
        for c in cs:
            c["message"] = _short(c["message"])
    return v


# 总览用的只统计写法 → 原结果名（检查清单核对时按原结果名核）
DIAG_ALIAS = {"t1-l2-flip-stats-diag": "t1-l2-flip-stats"}

# 不是现行配置的 TPU 抓数：2026-11-11 起对齐配置关 base2，第 1 步以关 base2 的 tw-ra1z 为准，开 base2 的 tw-ra1b 不计
ROUTE_AGREE_SKIP = {"tw-ra1b"}


def route_agree():
    """路由一致率（PROTOCOL 第 5 节，2026-11-09 起按对真值口径）：每层 R = TPU 对 CPU FP64 自选的翻转率（DSA 为不重合率）
    ÷ GPU 各份对 FP64 的最大值，≤ 1.2 过、≤ 2 待定、> 2 不过；每步一条，取路由与 DSA 各层、各 TPU 运行里最差的。"""
    out = []
    for step in (1, 8, 30, 100):
        d = _load(f"t1-route-agree-s{step}.json")
        rs = {"routing": [], "dsa": []}
        for kind in rs:
            for layer, v in d[kind].items():
                fr = v.get("vs_fp64_flip_rate") or v["vs_fp64_nonoverlap_rate"]
                gmax = max(x for n, x in fr.items() if n not in v["tpu"])
                for run, t in v["tpu"].items():
                    if run in ROUTE_AGREE_SKIP:
                        continue
                    rs[kind].append((t["truth"]["R"], layer, run, gmax, fr[run]))
        worst = max(rs["routing"] + rs["dsa"])
        r = worst[0]
        st = "pass" if r <= R_LIMIT else ("inconclusive" if r <= 2 else "fail")
        out.append(
            {
                "id": f"L3.route_agree.s{step}",
                "type": "discrete_agreement",
                "level": "L3",
                "status": st,
                "threshold": {
                    "kind": "numeric",
                    "values": {"r_ratio": R_LIMIT},
                    "source": "PROTOCOL.md 第 5 节：路由一致率对真值口径",
                },
                # 两边对 FP64 的翻转率（DSA 为不重合率）放在 rel_l2_* 的位置，格子照 R 比写「GPU a、TPU b → R」
                "metrics": {
                    "r_ratio": r,
                    "rel_l2_gpu_vs_ref": worst[3],
                    "rel_l2_tpu_vs_ref": worst[4],
                    "cases": len(rs["routing"]) + len(rs["dsa"]),
                },
                "message": f"第 {step} 步（TPU {'、'.join(sorted({x[2] for x in rs['routing']}))}）：路由各层 R {min(rs['routing'])[0]:.2f}–{max(rs['routing'])[0]:.2f}，"
                f"DSA 各层 R {min(rs['dsa'])[0]:.2f}–{max(rs['dsa'])[0]:.2f}（TPU 对 CPU FP64 的翻转率 ÷ GPU 三份对 FP64 的最大值）；"
                f"最差 {'DSA' if worst in rs['dsa'] else '路由'} 第 {worst[1]} 层（`{worst[2]}`）",
            }
        )
    return out


def flip_stats_diag():
    """L2 逐层翻转统计：PROTOCOL 第 5 节「只统计、不设过线」，总览里合成一条诊断（🟡），写两类里最大的 R；判定由 L3 路由一致率给（同一批数）。"""
    cs = json.loads((RES / "t1-l2-flip-stats.results.json").read_text())["checks"]
    zh = {"L2.flips.routing": "路由", "L2.flips.dsa_topk": "DSA top-k"}
    w = max(cs, key=lambda c: (c.get("metrics") or {}).get("r_ratio", 0))
    m = w.get("metrics") or {}
    return [
        _diag(
            "L2.flips.worst",
            w.get("type", "flip_stats"),
            "L2",
            f"只统计，判定见 L3；最差 {zh.get(w['id'], w['id'])} R {m.get('r_ratio', 0):.3f}",
            {k: m[k] for k in ("r_ratio", "rel_l2_gpu_vs_ref", "rel_l2_tpu_vs_ref") if k in m},
        )
    ]


# 同一运行的第二个窗口在总览里与主窗口合成一格：检查 id 加后缀，说明里「取最差」能看出是哪个窗口。
# 别名 → (原结果名, id 后缀)；检查清单核对时按原结果名核。
WINDOW_ALIAS = {
    "t1zc300-w": ("t1zc300-l4", "@前300步"),
    "t1zo300-w": ("t1zo300-l4", "@前300步"),
    "t1zo300qk-w": ("t1zo300-l4-qkclip", "@前300步"),
    "t1zga300-w": ("t1zga300-l4", "@前300步"),
    "t1mz300-w": ("t1mz300-l4-qkclip", "@前300步"),
}


def window_aliases():
    out = {}
    for alias, (name, suffix) in WINDOW_ALIAS.items():
        f = RES / f"{name}.results.json"
        cs = [{**c, "id": c["id"] + suffix} for c in json.loads(f.read_text())["checks"]]
        out[alias] = (f, cs)
    return out


# ---- 历代比较（history.md）：总览只放现行比较，各尺寸的历代与诊断代在这里，每格是判定与 r（数字从结果文件现算）
SYM = {"pass": "✅", "inconclusive": "🟡", "fail": "❌", "error": "❌"}
ORDER = ("error", "fail", "inconclusive", "pass")
L4_COLS = [  # (表头, 检查 id)
    ("主 loss", "L4.loss"),
    ("主 loss 后段", "L4.loss.tail"),
    ("MTP loss", "L4.mtp_loss"),
    ("MTP 后段", "L4.mtp_loss.tail"),
    ("索引器 KD", "L4.indexer_loss"),
    ("grad_norm", "L4.grad_norm"),
    ("param_norm", "L4.param_norm"),
    ("偏置均值", "L4.router.bias_mean"),
    ("偏置标准差", "L4.router.bias_std"),
]
# 每张表：(小标题, 说明, [(结果名, 被测, 设置), …])，按时间先后；结果名带 * 取最后一个匹配的文件
HISTORY = [
    (
        "mini · 1000 步（对 GPU g3-a；噪声 g3 两两：现行 tc-a、上一代 tz-off-rep 与 x5-prod 16 份 120 对，其余各代 8 份 28 对，两种 N 下的 r 不能直接比）",
        "M 编号是开关台账的行；「诊断」是只为拆开某个开关效果的运行，不当现行。",
        [
            ("t1u-l4", "t1u-a", "M28 之前"),
            ("t1w-l4*", "t1w", "+M28（t1v 的设置）"),
            ("t1zk-l4", "t1zk-a", "+M30、M32（t1z 的设置）"),
            ("t1zr-l4", "t1zr-a", "+M36"),
            ("t1zs-l4", "t1zs-a", "诊断：t1zr + M37"),
            ("t1zt-l4", "t1zt-a", "诊断：t1zs + M34"),
            ("t1zub-l4", "t1zu-b", "诊断：t1zt + M39"),
            ("t1zuc-l4", "t1zu-c", "诊断：t1zt + M39、M41"),
            ("t1zud-l4", "t1zu-d", "诊断：t1zt + M39、M40"),
            ("t1zu-l4", "t1zu-a", "t1zt + M39、M40、M41"),
            ("t1zv-l4", "t1zv-b", "诊断：t1zu + M42"),
            ("t1zz-l4", "t1zz-a", "验证：t1zw 再关 splash 的 base2 指数"),
            ("t1zw-l4", "t1zw-a", "上一代：t1zu + M42b"),
            ("t1zga-l4", "t1zg-a", "上一代：t1zw + M50（KD 广义 KL，只改记录值）"),
            ("t1zla-l4", "t1zl-a", "验证：t1zg + M51（精确 log），主 loss 变差，没采用"),
            (
                "t1zgb-l4",
                "t1zg-b",
                "t1zg-a 的开关、M50 修订版代码（MaxText fd30fbfb）：KD 记录值逐位相同，其余最多差 2 ulp",
            ),
            ("t1zo-l4", "tz-off-rep", "上一代：t1zg 再关 base2（噪声换 g3 16 份 120 对）"),
            (
                "t1zc-l4",
                "tc-a",
                "**现行**：C 口径，tz-off-rep 再开 M52、M58（2026-11-11 起；6 份取第一份；tc-b … tc-d、tc-f 只改执行方式，tc-e 改了重算策略）",
            ),
            (
                "t1zp-l4",
                "x5-prod",
                "验证：生产建议组合（tz-off-rep（C 口径之前）再开 M42、嵌入与 RMSNorm scale 梯度 FP32，关 M8、M10、M27、M30、M36、M42b）",
            ),
        ],
    ),
    (
        "mini · 300 步（t1n、t1s 对 GPU g1-a，其余对 g2-a；噪声各自 8 份两两 28 对）",
        "t1n、t1s 两代按当时的角色（GPU 当被测）记差；g1 → g2 是 GPU 参照的一处修正（台账 G17）。",
        [
            ("t1-l4-nozm", "t1n-a", "第一代（路由偏置零均值关）"),
            ("t1s-l4", "t1s-a", "第二代（再开四个舍入开关与索引器真 FP32）"),
            ("t1t-l4", "t1t-c", "M28 之前"),
            ("t1v-l4*", "t1v", "+M28"),
            ("t1z-l4", "t1z-a", "+M30、M32"),
            ("t1zr300-l4", "t1zr-a", "+M36（前 300 步）"),
            ("t1zs300-l4", "t1zs-a", "诊断：+M37（前 300 步）"),
            ("t1zt300-l4", "t1zt-a", "诊断：+M34（前 300 步）"),
            ("t1zub300-l4", "t1zu-b", "诊断：t1zt + M39（前 300 步）"),
            ("t1zuc300-l4", "t1zu-c", "诊断：t1zt + M39、M41（前 300 步）"),
            ("t1zud300-l4", "t1zu-d", "诊断：t1zt + M39、M40（前 300 步）"),
            ("t1zu300-l4", "t1zu-a", "t1zt + M39、M40、M41（前 300 步）"),
            ("t1zv300-l4", "t1zv-b", "诊断：t1zu + M42（前 300 步）"),
            ("t1zz300-l4", "t1zz-a", "验证：再关 base2（前 300 步）"),
            ("t1zw300-l4", "t1zw-a", "上一代：t1zu + M42b（前 300 步）"),
            ("t1zga300-l4", "t1zg-a", "上一代（前 300 步）"),
            ("t1zla300-l4", "t1zl-a", "验证：+M51（前 300 步）"),
            ("t1zo300-l4", "tz-off-rep", "上一代：再关 base2（前 300 步）"),
            ("t1zc300-l4", "tc-a", "**现行**：C 口径，再开 M52、M58（前 300 步）"),
            ("t1zp300-l4", "x5-prod", "验证：生产建议组合（前 300 步）"),
        ],
    ),
    (
        "real-weight · D1（300 步对 gd2-a、1000 步对 gd3-a；噪声各自 28 对）",
        "",
        [
            ("t2d1-l4", "td2-a", "第一代，300 步（t1v 等价设置）"),
            ("t2d1z-l4", "td3-a", "第二代，300 步（t1z 的开关、M25 暂关）"),
            ("t2d14-l4*", "td4-a", "第三代，300 步（t1zr 再加 M34、M37，M25 重新打开）"),
            ("t2d15-l4*", "td5-a", "1000 步（当时 T2 配置的全部开关，含 M39–M41，不含 M42b）"),
            ("t2d15b-l4", "td5b-b", "最近一代，1000 步（含 M42b；base2 开、早于 M50，现行对齐配置在 D1 上尚无运行）"),
        ],
    ),
    (
        "real-weight · B（300 步对 gb2-a、1000 步对 gb3-a；噪声各自 28 对；第一代 tb2-a 第 1 步更新后出 NaN，没有 L4）",
        "",
        [
            ("t2bz-l4", "tb3-b", "第二代，300 步（同 td3-a 的设置）"),
            ("t2b4-l4*", "tb4-a", "第三代，300 步（同 td4-a 的设置）"),
            ("t2b5-l4*", "tb5-b", "1000 步（同 td5-a 的设置）"),
            ("t2b5b-l4", "tb5b-a", "最近一代，1000 步（含 M42b；base2 开、早于 M50，现行对齐配置在 B 上尚无运行）"),
        ],
    ),
    (
        "mid · 300 步（对 GPU gm-a，噪声 gm 两两 28 对）",
        "tm-a 与 tm-b 差不止一个开关，不能当 A/B。",
        [
            ("tm-l4", "tm-a", "fd796f6 时的配置（没有 M39–M41、M42b）"),
            ("tmb-l4", "tm-b", "最近一代（含 M39–M41 与 M42b；base2 开、早于 M50，现行对齐配置在 mid 上尚无运行）"),
        ],
    ),
]
# 第 1 步整网（L3）的历代：(结果名, 被测, 设置)，判定同总览 L3 一行
HISTORY_L3 = [
    ("t1s-step1-grad-vs-fp64", "t1diag15", "t1s 设置，对 g1s 三份"),
    ("t1t-step1-grad-vs-fp64", "t1diag17", "t1t 设置，对 g2s 三份"),
    ("t1v-step1-grad-vs-fp64", "t1vdiag", "t1v 设置（+M28），对 g2s 三份"),
    ("t1z-step1-grad-vs-fp64", "t1zdiag2", "上一代：t1z 设置（+M30、M32），对 g2s 三份"),
    ("t1zg-step1-grad-vs-fp64", "mz-diag1", "旁证：base2 开那一代配置的诊断程序梯度，对 g2s 三份"),
    ("t1zg-step1-train-grad-vs-fp64", "mz-opt2", "上一代：base2 开的配置训练步实际用的梯度，对 g2s 三份"),
    (
        "t1zo-step1-train-grad-vs-fp64",
        "tz-off-rep",
        "C 口径之前的配置（base2 关）训练步实际用的梯度，对 g2s 三份（2026-11-11 前的尺子）",
    ),
    (
        "t1zo-step1-train-grad-vs-fp64-n6",
        "tz-off-rep",
        "上一代（C 口径之前）：同上，尺子换成 g2s 6 份 15 对（2026-11-11 补 g2s-d … f）",
    ),
    (
        "t1zc-step1-train-grad-vs-fp64-n6",
        "s1f-c",
        "**现行**：C 口径（M36、M52、M58 都开）训练步实际用的梯度，对 g2s 6 份 15 对",
    ),
]


def _r_of(c):
    """一条 L4 检查 → (r, 判定量)：三个判定量里实测 ÷ N 最大的。multi_golden 只存了绝对值（*_max）。"""
    up = c["threshold"]["noise"]["upper"]
    best = (0.0, "")
    for k, n in up.items():
        raw = c["metrics"].get(k)
        m = raw if raw is not None else c["metrics"].get(k + "_max")
        if m is not None and n:
            best = max(best, (abs(m) / n, k), key=lambda t: t[0])
    return best


def _resolve(name):
    hits = sorted(ALIGN.glob(f"*/results/{name}.results.json"))
    if not hits:
        raise SystemExit(f"历代比较引用的结果文件不存在：{name}")
    return hits[-1]


def history_md(v):
    rel = Path(__file__).resolve().relative_to(ROOT.parents[1]).as_posix()
    md = [
        "[janus](../../../README.md) › [混元 4](../README.md) › [总览](dashboard.md) ｜ 历代比较",
        "",
        "# 混元 4 · 历代比较",
        "",
        f"由 `{rel}` 从结果文件生成，数字和判定都不手写。[总览](dashboard.md)只放各尺寸的现行比较；"
        "这里放历代与诊断代，便于看每一代开关的效果。",
        "",
        "- L4 每格：判定与 r = TPU 对 GPU 基线的差 ÷ GPU 自己重跑的噪声上限 N，带符号均值、MARE、最大单步三个判定量里取 r 最大的；"
        "r ≤ 1 过（✅），1–2 待定（🟡），> 2 不过（❌）；— 是这次没有这一项。每格的 N 与 TPU 实测见各结果文件的 report.md。",
        "- 300 步与 1000 步的 r 不能直接比大小：GPU 自己重跑的散布随步数变宽。",
        "- 每一代改了什么：开关台账 [`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md)；判法的改动：[`PROTOCOL.md`](../PROTOCOL.md) 变更记录。",
        "",
    ]
    for title, lead, rows in HISTORY:
        md += [f"## {title}", ""] + ([lead, ""] if lead else [])
        md += [
            "| 被测 | 设置 | " + " | ".join(h for h, _ in L4_COLS) + " |",
            "|---|---|" + "---|" * len(L4_COLS),
        ]
        for name, run, what in rows:
            f = _resolve(name)
            cs = {c["id"]: c for c in json.loads(f.read_text())["checks"]}
            cells = []
            for _, cid in L4_COLS:
                c = cs.get(cid)
                cells.append(f"{SYM[c['status']]} {_r_of(c)[0]:.2f}" if c else "—")
            link = (
                f"[`{run}`](../alignments/{f.parent.parent.name}/results/{f.name[: -len('.results.json')]}.report.md)"
            )
            md.append(f"| {link} | {what} | " + " | ".join(cells) + " |")
        md.append("")
    md += [
        "## mini · 第 1 步整网（L3）",
        "",
        "全部参数梯度对 GPU 第 1 步参照的噪声（现行 6 份 15 对，之前各代 3 份 3 对），超出的对 CPU FP64 整网重算定责（规则见总览 L3 一组）。",
        "",
        "| 被测 | 设置 | 判定 | 张量 | 超出 N 而未定责 | 其中超出 2N |",
        "|---|---|---|---|---|---|",
    ]
    for name, run, what in HISTORY_L3:
        c = next(c for c in v[name][1] if c["id"] == "L3.step1.grads.vs_gpu")
        m = c["metrics"]
        md.append(
            f"| `{run}` | {what} | {SYM[c['status']]} | {m['tensors']} | {m['n_unattributed_over_n']} | "
            f"{m['n_unattributed_over_2n']} |"
        )
    (ROOT / "dashboard" / "history.md").write_text("\n".join(md) + "\n")


# 总览头条的数字一律从结果文件读（plain_legend），这里只放文字骨架
L4_NAME = {
    "L4.loss": "主 loss 全程",
    "L4.loss.tail": "主 loss 第 101 步起",
    "L4.mtp_loss": "MTP loss",
    "L4.mtp_loss.tail": "MTP loss 第 101 步起",
    "L4.indexer_loss": "索引器 KD",
    "L4.grad_norm": "梯度范数",
    "L4.param_norm": "参数范数",
    "L4.router.bias_mean": "路由偏置均值",
    "L4.router.bias_std": "路由偏置标准差",
}
METRIC_NAME = {"signed_mean": "带符号均值", "mare": "平均绝对相对误差", "max_step": "单步最大"}
HEAD_L4, HEAD_L4_300, HEAD_QK, HEAD_STEP1, HEAD_STEP1_N3, HEAD_QK_L1 = (
    "t1zc-l4",
    "t1zc300-l4",
    "t1zo-l4-qkclip",
    "t1zc-step1-train-grad-vs-fp64-n6",
    "t1zo-step1-train-grad-vs-fp64",
    "t1zo-qkclip-maxlogit",
)
FACTORIAL = "mini-factorial-tpu-6"  # 主 loss 第 551–1000 步的单因素交叉（C 口径 6 份）
REPEATS_C = "t1zc-repeats-6"  # C 口径 6 份各自对 g3-a 的 L4 r（16 份尺子）
REPEATS_C_RUNS = (
    "tc-a",
    "tc-b",
    "tc-c",
    "tc-d",
    "tc-e",
    "tc-f",
)  # by_check.*.c.values 的顺序（与结果文件 gpu_loo_rank_of_tpu 里 tc-* 键同序）


def _e(x, sign=False):
    """3.5e-4 写法（指数不补零）。"""
    t = f"{x:+.2e}" if sign else f"{x:.2e}"
    return re.sub(r"e([+-])0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), t)


def _e6(x):
    """+7.0e-6 写法：以 1e-6 为单位留一位小数。"""
    return f"{x * 1e6:+.1f}e-6"


def _pairs_to_runs(k):
    """两两配对数 → 份数（n(n−1)/2 = k）。"""
    n = round((1 + (1 + 8 * k) ** 0.5) / 2)
    if n * (n - 1) // 2 != k:
        raise SystemExit(f"噪声对数 {k} 不是 n(n−1)/2")
    return n


def plain_legend(v):
    """总览头条：L4 读 t1zc-l4 / t1zc300-l4（C 口径）与 t1zo-l4-qkclip（C 口径之前），L3 读 C 口径 6 份尺子的 t1zc-step1-train-grad-vs-fp64-n6（3 份尺子的对照仍是 C 口径之前的 t1zo-step1-train-grad-vs-fp64），
    低阈值 QK-Clip 读 t1zo-qkclip-maxlogit，主 loss 后段的残余读 mini-factorial-tpu-6。改了结果文件，make dashboard 后头条跟着变。"""

    def checks(name):
        return {c["id"]: c for c in json.loads(_resolve(name).read_text())["checks"]}

    l4, l4s, qk = checks(HEAD_L4), checks(HEAD_L4_300), checks(HEAD_QK)
    cnt = {k: sum(c["status"] == k for c in l4.values()) for k in ("pass", "inconclusive", "fail")}
    inco = [cid for cid, c in l4.items() if c["status"] == "inconclusive"]
    lo = l4["L4.loss"]
    sm, up = lo["metrics"]["signed_mean_max"], lo["threshold"]["noise"]["upper"]["signed_mean"]

    def rr(c):
        r, k = _r_of(c)
        return f"{METRIC_NAME[k]} {r:.2f} 倍"

    s1 = next(c for c in v[HEAD_STEP1][1] if c["id"] == "L3.step1.grads.vs_gpu")
    s3 = next(c for c in v[HEAD_STEP1_N3][1] if c["id"] == "L3.step1.grads.vs_gpu")
    rows = [r for r in json.loads((RES / f"{HEAD_STEP1}.json").read_text())["rows"] if not r.get("tpu_file_corrupt")]
    over = [r for r in rows if r["gpu_noise_max"] > 0 and r["tpu_vs_gpu_min"] / r["gpu_noise_max"] > 1]
    far = [r for r in over if r["R_vs_ref"] > R_LIMIT]
    un = [r for r in far if r["tensor"] not in STEP1_ACCEPTED]
    acc = [r for r in far if r["tensor"] in STEP1_ACCEPTED]
    if s1["metrics"]["n_unattributed_over_n"] != len(un):
        raise SystemExit("头条的 L3 未定责计数和结果对不上")

    def one(r):
        return (
            f"`{r['tensor'].removeprefix('model.')}`（超出波动 {r['tpu_vs_gpu_min'] / r['gpu_noise_max']:.2f} 倍，"
            f"离精确值 GPU {_e(r['gpu_vs_ref'], False)}、TPU {_e(r['tpu_vs_ref'], False)}，R {r['R_vs_ref']:.2f}）"
        )

    fac = json.loads((RES / f"{FACTORIAL}.json").read_text())["cells"]
    fb, fm, fc = (fac[k]["lm_551_1000"]["mean"] for k in ("base", "M58", "C"))
    ftail = json.loads((RES / f"{FACTORIAL}.json").read_text())["tail_threshold_751_1000"]
    rep = json.loads((RES / f"{REPEATS_C}.json").read_text())["by_check"]
    rep_inco = []
    for cid in ("L4.loss", "L4.loss.tail", "L4.mtp_loss", "L4.mtp_loss.tail", "L4.grad_norm"):
        vals = rep[cid]["c"]["values"]
        if len(vals) != len(REPEATS_C_RUNS):
            raise SystemExit(f"{REPEATS_C}.json 的 {cid} 份数和 REPEATS_C_RUNS 对不上")
        rep_inco += [
            f"{run} {L4_NAME[cid]} {x:.2f} 倍" for run, x in zip(REPEATS_C_RUNS, vals) if x > 1 and run != "tc-a"
        ]
    ql = next(c for c in json.loads(_resolve(HEAD_QK_L1).read_text())["checks"] if c["id"] == "L1.qkclip.max_logit")
    qm = ql["metrics"]
    n_fen = _fen(rows)
    return [
        "**现状：还不能签字**（签字＝正式宣布两边对齐了）。以现行比较 tc-a 为准（2026-11-11 起对齐配置改 C 口径：TPU 在超连接与 hc_head 一带"
        f"照 GPU 的写法舍 16 位，GPU 不改；C 口径 6 份（tc-b … tc-d、tc-f 只改执行方式，tc-e 改了重算策略、反向会重选路由）取第一份；1000 步的波动上限是 GPU {_pairs_to_runs(l4['L4.loss']['threshold']['noise']['refs'])} 份）："
        "hy4-mini（混元 4 全部结构、缩小尺寸的随机初始化模型）在 TPU 和 GPU 上用同一份权重、同一份数据训练 1000 步，"
        f"9 项曲线 {cnt['pass']} 项对上、{cnt['inconclusive']} 项略超（"
        + "、".join(f"{L4_NAME[i]}：{rr(l4[i])}" for i in inco)
        + "），"
        + ("没有不过；" if not cnt["fail"] else f"{cnt['fail']} 项不过；")
        + f"主 loss {'对上' if lo['status'] == 'pass' else '略超'}（带符号均值，即逐步相对差的平均：TPU {_e(sm, True)}、GPU 自己重跑的波动上限 {_e(up, False)}，{sm / up:.2f} 倍；"
        f"三个量里最差的 {rr(lo)}）。C 口径之前（tz-off-rep）主 loss 略超，后段偏高的主因是超连接里流的梯度怎么舍，TPU、GPU 两头各改一处都验证了："
        f"第 551–1000 步主 loss 对 GPU 16 份均值，只开 M36 的对照 {fac['base']['n']} 份平均 {_e6(fb)}，在它上面只加开流梯度舍法（M58）{fac['M58']['n']} 份 {_e6(fm)}"
        f"（p {fac['M58']['vs_control']['lm_551_1000']['perm_p']:.3f}），其中 {fac['M58']['n_tail']} 份仍超尾部阈值（第 751–1000 步均值超 {ftail * 1e6:.0f}e-6）；"
        f"但 C 口径（M36、M52、M58 都开）{fac['C']['n']} 份平均仍是 {_e6(fc)}，{fac['C']['n_tail']} 份尾部还在，"
        f"和只开 M36 的对照分不开（p {fac['C']['vs_control']['lm_551_1000']['perm_p']:.2f}）；M52 开着时补 M58 全套或任一半都测不出作用"
        "（mini-factorial-c-parts-vs-m52.json），所以 C 口径下主 loss 判档刚好过线不等于后段修好了，机制待查（"
        f"{FACTORIAL}.json，各份读数见对齐报告；"
        f"对照格是 {'、'.join(fac['base']['runs'])}，不含 tz-off-rep；M58 格、C 格各含一份 remat_policy=full 的运行）。"
        + (
            f"C 口径 6 份都判（{REPEATS_C}.json）时没有不过，但另几份有待定：{'；'.join(rep_inco)}。"
            if rep_inco
            else ""
        )
        + f"同一运行前 300 步梯度范数{'略超' if l4s['L4.grad_norm']['status'] == 'inconclusive' else '对上'}（{rr(l4s['L4.grad_norm'])}）。"
        f"C 口径之前的另一份运行 tz-off-qkc（派生每步最大 logit 后记作 tz-off-qkq）的每步最大注意力打分 1000 步{'略超' if qk['L4.qkclip.max_logit']['status'] == 'inconclusive' else '见下表'}（{rr(qk['L4.qkclip.max_logit'])}，"
        "base2 开的上一代已接受，关 base2 后没有重做同一状态的证据，是否沿用尚未决定；C 口径上没测）。挡签字的还有："
        f"第一步整网（C 口径）{len(rows)} 个梯度里 {len(over)} 个超出 GPU {n_fen}之间的波动"
        f"（{len(over) - len(far)} 个离精确值不超过 GPU 的 1.2 倍、算已查明来源；另 {len(far)} 个比 GPU 远："
        + "、".join(one(r) for r in un)
        + f"，计 {len(un)} 个未定责"
        + (
            "（超出波动都不到 2 倍，所以这一项是略超不是没对上）；"
            if not s1["metrics"]["n_unattributed_over_2n"]
            else "；"
        )
        + "、".join(one(r) for r in acc)
        + "，另列（按 D11 豁免规则：比 GPU 远的张量都单列，enorm 的接受理由「两边离精确值都不到一次 BF16 舍入」复核成立、不计入，超连接小参数没有登记过接受、计入；单列的挡不挡签字待拍板）。"
        f"GPU 第一步参照 2026-11-11 从 3 份补到 6 份（看过 3 份的判定之后补跑，6 份能否作正式判定待拍板），C 口径之前那一代 3 份时是 {s3['metrics']['n_over_n']} 个超出、{s3['metrics']['n_unattributed_over_n']} 个未定责），"
        f"低阈值下（C 口径之前的配置）第一步第 {qm['worst_layer']} 层每头最大注意力打分离精确值 GPU {_e(qm['rel_l2_gpu_vs_ref'], False)}、TPU {_e(qm['rel_l2_tpu_vs_ref'], False)}，"
        f"{qm['r_ratio']:.2f} 倍（未定责），"
        "参照答案（GPU 上同设置的标准训练）没和 GPU 生产程序比过，三条事后改的判法要在真实权重上做前瞻验证（判法事先固定，再看新结果）。",
        "",
        "为什么、生产怎么配、还差什么，见[对齐报告](../alignments/01-mini/ALIGNMENT-REPORT.md)。",
        "",
        "TPU 自己只改一样东西（并行方式、编译开关、代码写法、版本、硬件）结果变不变、变多少，见 [TPU 自比](../alignments/04-tpu-self/README.md)；真有影响的几件逐件讲在 [`impacts/`](../alignments/04-tpu-self/impacts/README.md)。",
        "",
        "| 符号 | 意思 |",
        "|---|---|",
        "| ✅ / 🟡 / ❌ | 对上 / 略超或只看不判（GPU 列是做过、没写成结果文件）/ 没对上 |",
        "| ⬜ / — | 还没做 / 不适用 |",
        "| 对上 | 差别不超过 GPU 自己重跑的波动；量不出波动的地方，TPU 离精确值（CPU 上 64 位计算）不超过 GPU 的 1.2 倍 |",
        "| 细节 | 每一格的数字、门槛与推导在[总览细节](detail.md)，术语见[知识库的术语对照表](../knowledge.md#术语对照表) |",
    ]


HOW_TO_READ = [
    "知识库页：[总览](dashboard.md)每一格的推导。术语见[知识库的术语对照表](../knowledge.md#术语对照表)。",
    "",
    "## 怎么读这张总览",
    "",
    "### 读法",
    "",
    "混元 4 各文档（总览、对齐报告、根源一览、差异清单）的 N、r、R、定责依据、类别都按这一张表读。",
    "",
    "| 术语 | 含义 | 判定线 |",
    "|---|---|---|",
    "| N | 噪声上限：GPU 同设置重跑之间的差。L4 取两两配对里这个量的最大值（mini 前 300 步 8 份 28 对、1000 步 16 份 120 对；其他尺寸 8 份 28 对），L3 取第 1 步参照两两的最大 rel_L2（mini 现行 6 份 15 对） | — |",
    "| r | TPU 对 GPU 的差 ÷ N；只量两边差多大，不定责。L4 每项取带符号均值、MARE、最大单步里最差的一个 | ≤ 1 过，> 2 不过，之间待定 |",
    "| 带符号均值 | L4 判定量：窗口内逐步相对差的平均（带符号，看系统偏差） | 对 N 判 |",
    "| MARE | L4 判定量：逐步相对差绝对值的平均（看整体离多远） | 对 N 判 |",
    "| 最大单步 | L4 判定量：窗口内单步最大的相对差（看尖峰） | 对 N 判 |",
    "| R | rel_L2(TPU, CPU FP64) ÷ rel_L2(GPU, CPU FP64)；真值是 CPU 上的 FP64 重算，与被测不同源，能定责 | ≤ 1.2 过 |",
    "| 分辨率下限 | GPU 噪声 N = 0、两边对真值都小于一次 BF16 舍入（u = 2^-8 ≈ 3.9e-3）的张量，R 只是两次舍入之比 | 直接判过（`PROTOCOL.md` 第 5 节） |",
    "| 定责依据 | 只认三种：对 CPU FP64 的 R；读代码对照写明的语义标准；只改一边后两边的差收拢 | — |",
    "| 类别 | A 对齐专用（只为照 GPU 的舍入位置）；B 修 bug、生产也开（更准且代价 < 1%）；C 取舍（更准但代价 ≥ 1%）；设置（按训练配方）。生产建议看对真值的误差加代价，对齐配置看是否照 GPU 写法，两者可以不同 | — |",
    "| 代价 | hy4-mini 300 步单开关 A/B（`tq-*`）：全开步时 ÷ 只关这一个开关的步时 − 1（各取第 11–300 步中位数），全开 0.4767 s | — |",
    "| 「T1 开」 | 在 mini 的对齐配置 `hy4-mini-t1.yml` 里打开（T2 = 真实权重截层，mid 同理） | — |",
    "| 严格档 / 参考档 | 严格档是 janus 的验收判定（噪声、逐位、R）；参考档是同类迁移项目的经验门槛，只并列报告（`docs/reference/judging.md`「两档门槛」） | 只看严格档 |",
    "",
    "比值都和两个绝对值一起写：「GPU a、TPU b → R」或「TPU x，N y」。",
    "",
    "### 怎么读一格",
    "",
    "| 项 | 读法 |",
    "|---|---|",
    "| 格子 | 实测值，括号里是离严格门槛多远（×N 或 实测 / 门槛），后面是严格档符号，「｜参」后是参考档符号；列的指标或 N 与门槛栏不同时，括号里另写 |",
    "| 多条检查 | 写最差的一条；mini 的 L4 格子合了 1000 步与前 300 步两个窗口，写较差的那个 |",
    "| N=…（k 对） | k 是算 N 用的噪声参照对数，不是份数 |",
    "| 说明 | 表下逐条写非 ✅ 格子的推导（实测｜严格档｜参考档）与出处：根源一览的行、各尺寸报告的 D 条目 |",
    "| 诊断 | 只报不判，记成 🟡 |",
    "| 不在这里的 | 历代比较见[历代比较](history.md)；mid、真实权重截层、tiny 见[支线页](other-sizes.md) |",
    "",
    "### 符号",
    "",
    "- 严格档：✅ 过　🟡 待定、诊断或进行中　❌ 不过或出错　⬜ 未开始　— 不适用",
    "- 参考档：参✅ 过　参☑️ 记录了的都过、有门槛项没记录　参❌ 不过　参❔ 所需指标未记录",
    "",
]

GOLDEN_NOTE = [
    "### L4 比的是谁",
    "",
    "| 项 | 事实 |",
    "|---|---|",
    "| golden | GPU 一侧（B200 × 8，Megatron-Core）的结果；TPU（MaxText，v5p 8 个设备）是被测。语义以 MaxText 的混元 4 实现为准，优化器与 loss 的细节 MaxText 加默认关的开关对上原版 Megatron-Core（`PROTOCOL.md` 第 1.2 节） |",
    "| 基线与噪声 | GPU 同设置重跑（1000 步 `g3-a…p` 16 份 120 对，前 300 步 `g2-a…h` 8 份 28 对）；TPU 只和 `g3-a` / `g2-a` 逐步比，其余各份只用来量 N；TPU 对各份各比一次只作诊断 |",
    "| 第 1 步（L3） | 逐张量对 GPU 第 1 步参照（mini 现行 6 份）的噪声，超出的对 CPU FP64 整网重算看 R |",
    "| 学习率 | 登记的常数 1e-5，归 L0 设置核对，不计入 L4 |",
    "| 单次运行 r 的分辨率 | 只开一个只把 FP32 log 的误差从 3 ULP 收到 ≤ 1 ULP 的开关（M51；它在 TPU 上本身改数值，CPU 上梯度写法逐位不变），第 1 步 10 个索引器梯度张量不逐位（差 ≤ 2.3e-4），1000 步主 loss 带符号均值 r 0.36 → 1.44；诊断程序与训练步分开编译，第 1 步梯度相对差中位数 2.9e-4。r 在 1 附近的差别不用来比较各代好坏，判定只看过 / 待定 / 不过与定责证据 |",
    "| 其余 | 开关、GPU 参照的修正、测量口径见 [mini 对齐报告](../alignments/01-mini/ALIGNMENT-REPORT.md)；根源与修法见 [`ROOT-CAUSES.md`](../ROOT-CAUSES.md)；代号见 mini 对齐细节第 1 节 |",
]

OTHER_HOW_TO_READ = [
    "## 怎么读这张表",
    "",
    "- 读法同[总览](dashboard.md)；只列有非 ✅ 格子的行（全过的行照样计入汇总）。跨尺寸的对照见 mini 对齐细节「跨尺寸看」。",
    "",
]
OTHER_NOTE = [
    "### 这几个尺寸各测什么",
    "- **mid**（hy4-mid，5.19B，随机初始化）：只换尺寸，看 mini 的结论放大宽度后还在不在（[报告](../alignments/03-mid/ALIGNMENT-REPORT.md)）。",
    "- **B、D1**（真实权重截层）：B 测 MoE（第 0 层稠密 + 第 1 层 MoE），D1 测 MTP（第 0 层稠密 + 1 层 MTP）"
    "（[报告](../alignments/02-real-weight/ALIGNMENT-REPORT.md)）。",
    "- **tiny**（hy4-tiny，5.4M）：只做 TPU 一侧的 CPU 冒烟（[报告](../alignments/00-tiny/ALIGNMENT-REPORT.md)）。",
    "- 判法同 mini：每个尺寸用 GPU 同设置多份两两之差量噪声上限 N（这里各尺寸都是 8 份 28 对；mini 现行 1000 步是 16 份 120 对），TPU 只和指定的一份比；L3 对 CPU FP64 看 R。",
]

SIZES = ["00-tiny", "01-mini", "02-real-weight", "03-mid"]


def check_claimed(v):
    """总览里每一格引用的检查，都要在某个尺寸的检查清单（checks.yaml）里有一行，否则失败。"""
    import yaml

    from janus.report import inventory as INV

    sources = INV.Sources(ROOT, virtual=v)
    got = set()
    for name in SIZES:
        for s in INV.claimed(
            yaml.safe_load((ALIGN / name / "dashboard" / "checks.yaml").read_text()), sources
        ).values():
            got |= s
    missing = []
    for fname in ("items.yaml", "other-sizes.items.yaml"):
        missing += _missing(yaml.safe_load((ROOT / "dashboard" / fname).read_text()), sources, got)
    if missing:
        raise SystemExit("总览引用的这些检查在各尺寸的 checks.yaml 里都没有对应的行：\n" + "\n".join(missing))


def _missing(spec, sources, got):
    missing = []
    for g in spec["groups"]:
        for it in g["items"]:
            for col in spec["columns"]:
                srcs = it.get(col)
                for src in srcs if isinstance(srcs, list) else [srcs]:
                    if not src or ":" not in src or src.startswith(("wip:", "todo:")):
                        continue
                    name = src.partition(":")[0]
                    if name in WINDOW_ALIAS:  # 别名按原结果名核
                        orig, suf = WINDOW_ALIAS[name]
                        src = f"{orig}:{src.partition(':')[2].replace(suf, '')}"
                    elif name in DIAG_ALIAS:  # 只统计的写法按原结果核
                        src = f"{DIAG_ALIAS[name]}:{src.partition(':')[2]}"
                    _, cs = sources.checks(src)
                    miss = {(sources.resolve(src.partition(":")[0]), c["id"]) for c in cs} - got
                    missing += [f"{col} · {it['name']} · {n}:{i}" for n, i in sorted(miss)]
    return missing


def main():
    from janus.report import gates

    gates.SCI_BELOW, gates.SHORT_EXP = 0.1, True  # 混元 4 的数值一律写 a e-n（3.5e-4）
    gates.UNICODE_MINUS, gates.R_DIGITS, gates.SIGNED_VALUE = True, 3, True
    v = virtual()
    for name in SIZES:  # 各尺寸的检查清单（CHECKS.md）：骨架在各自的 dashboard/checks.yaml
        adir = ALIGN / name
        Overview(
            ROOT,
            name,
            [],
            [""],
            "",
            checks_dirs=[name],
            out_dir=adir / "dashboard",
            checks_md=adir / "CHECKS.md",
            builder=Path(__file__),
            virtual=v,
        ).build_checks({"groups": []})
    v = {**v, **window_aliases()}  # 别名只给总览用，不进各尺寸的检查清单
    check_claimed(v)
    Overview(
        ROOT,
        "混元 4",
        HOW_TO_READ,
        GOLDEN_NOTE,
        "**本页结构**：怎么读 → L4 比的是谁 → 各列是什么 → L0–L4 各组检查（每组表下有「说明」）→ 汇总。"
        "历代与诊断代见[历代比较](history.md)，其他尺寸见[支线页](other-sizes.md)。",
        checks_md=False,
        builder=Path(__file__),
        virtual=v,
        prefer=("mini", "gpu"),
        detail_stem="detail",
        plain_legend=plain_legend(v),
    ).main()
    Overview(
        ROOT,
        "混元 4 · 其他尺寸（支线）",
        OTHER_HOW_TO_READ,
        OTHER_NOTE,
        "**本页结构**：怎么读 → 这几个尺寸各测什么 → 各列是什么 → L0–L4（只列有非 ✅ 格子的行）→ 汇总。主线（hy4-mini）在[总览](dashboard.md)。",
        checks_md=False,
        builder=Path(__file__),
        virtual=v,
        prefer=("mid", "B", "D1", "tiny"),
        stem="other-sizes",
        skip_all_pass=True,
    ).main()
    history_md(v)


if __name__ == "__main__":
    main()
