"""对齐总览：读 items.yaml 与各对齐实践 alignments/*/results/ 下的结果文件，生成 dashboard.md（仓库里看）与 dashboard.html（带颜色）。

用法：python models/hunyuan3/dashboard/build.py   （在仓库根目录跑，要 matplotlib）；生成逻辑在 janus/report/overview.py，本文件只放混元 3 的说明文字与配置。
产出：models/hunyuan3/dashboard/dashboard.{md,html}（跨实践总览：每行一项检查，每列一个实践）；
      models/hunyuan3/CHECKS.md（检查清单：每级比什么、对象是哪些、怎么判、v5p / v7x 结果；骨架在 checks.yaml，janus/report/inventory.py 渲染）；
      每个对齐实践 alignments/<实践>/README.md 里的逐级大表（janus/report/tables.py）与 figures/ 下的配图。
状态符号（严格档）：✅ 通过 / 对齐；🟡 证据不足或进行中；❌ 不对齐或有错误；⬜ 未开始；— 不适用。
参考档（同类 GPU→TPU 迁移项目的实践门槛，janus/report/reference_gates.yaml）：格子里「参」+ 符号，只并列报告，不决定验收。
故障演练列：✅ = 这一项至少抓到一种埋的错，且期望它抓的都抓到了；· = 没有哪种故障要求它报警。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))      # 仓库根：import janus

from janus.report.overview import Overview  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAULTS = ROOT / "alignments" / "00-gpu-noise-and-drills" / "faults" / "phase1-faults.md"


# 00 的 L4 逐步图是第一阶段 9 次 1000 步运行互比（analysis/curves.py 的默认用法），不是 s-a 对 golden 的那 100 步；
# s-a..s-d 的逐步原值没有进仓库，所以 00 的 L4 用这组图当证据。
LEGACY_FIGS = {
    "00-gpu-noise-and-drills": {
        "L4.loss": ["figures/lm_loss-overlay.png", "figures/lm_loss-pairdiff.png", "figures/lm_loss-mean.png"],
        "L4.grad_norm": ["figures/grad_norm-overlay.png", "figures/grad_norm-pairdiff.png", "figures/grad_norm-mean.png"],
        "L4.param_norm": ["figures/param_norm-overlay.png", "figures/param_norm-pairdiff.png", "figures/param_norm-mean.png"],
    },
}
# 被测对 golden 的 L4 逐步图：实践 → (结果文件, 被测运行, golden 运行, 窗口)。逐步原值在 results/series/<运行>.metrics.jsonl，
# golden 的逐步原值放在 00 的 results/series/（同一份 GPU golden，不重复存）。
SERIES_COMPARE = {
    # 01 的 L4 正式验收第一层（t-rmap-on-1000 对精确轨迹 rce-a）不配曲线图：图不进仓库，曲线用 curves.py 本地画
    "03-gpu-remat-golden": [("remat-noise", "rc-a", ["rc-b", "rc-c", "rc-d", "rc-e"], (1, 1001))],
}
# 故障演练表上方的代号释义（00 README 与总览共用）；定义见 faults/phase1-faults.md「每种故障埋了什么」。
FAULT_CODES = "代号：`f-weight`（第 2 层 linear_proj 的 FP32 主权重 ×(1+1e-3)）、`f-weight2`（同一权重 ×(1+1e-2)）、`f-data`（rank 0 输入 token 错位一格）、`f-router`（路由改在 BF16 算）、`f-lr`（学习率差 1 个 FP32 ULP）、`f-drop`（开 dropout 0.1）、`f-grad`（第 1 层 qkv 梯度在抓数之后 ×1/8）、`f-gscale`（同一梯度在抓数之前 ×1/8）、`f-bias`（路由偏置不更新）。"

# 总览开头讲清 v5p、v7x 两列比的是谁：golden 是什么、rcc 改了哪一处、为什么只对 rcc-a、噪声怎么取、其余对齐环节在哪。
# 以「  - 」开头的是上一条的子条目（HTML 里同样缩一级）。
GOLDEN_NOTE = [
    "### v5p、v7x 两列的 L4 比的是谁",
    "- **golden**：对齐的参照答案，是 GPU 生产训练程序（B200 × 8，Megatron-Bridge）跑出的结果。",
    "- **最终 golden `rcc`**：生产程序其余设置不变，只改一处。",
    "  - 改了什么：输出头 dgrad 沿 12 万词表的长求和改成分段 FP32 累加。",
    "  - 为什么改：B200 上这次长求和有系统偏差，不修的话参照本身就偏。",
    "- **基线与噪声**：",
    "  - `rcc` 跑了 10 份（`rcc-a..j`）。",
    "  - TPU 只和事先指定的一份 `rcc-a`（最终 golden 的第 1 份，验收的比较基线）比。",
    "  - 10 份两两配对的 45 对只用来量噪声上限 N。",
    "- **对未修的生产 golden `rc`**（整层重算加路由回放，即原版生产程序，5 份）：差另写成已解释差异。",
    "- **其余对齐环节**：`rcc` 只是对参照的一处修正。",
    "  - TPU 一侧的精度开关、数据回放、测量口径等见 `models/hunyuan3/ALIGNMENT-REPORT.md` 第 5、6、8 节。",
    "- **代号**：每个代号与运行名前缀的定义见 `docs/glossary.md`。",
]

# 总览开头的读法说明（Markdown）。按小节写，一条一事。
HOW_TO_READ = [
    "## 怎么读这张总览", "",
    "### 两档门槛", "",
    "- **严格档**：janus 自己的验收判定。",
    "  - 对噪声参照 N：≤ N 过、> 2N 不过、中间待定；",
    "  - 逐位类必须逐位相同；",
    "  - R 比照目录里的线。",
    "- **参考档**：同类 GPU→TPU 迁移项目的实践门槛，只并列报告、不决定验收。",
    "- 两档的含义与参考值见 `docs/reference/judging.md`「两档门槛」。", "",
    "### 怎么读一行", "",
    "- 「指标」「门槛」两栏：这一项比的是什么、严格档和参考档的线（取多数列相同的那个）。",
    "- 平台格子写**实测值**，括号里是离严格门槛多远（×N 或 实测 / 门槛的倍数）。",
    "  - 这一列的指标或门槛和门槛栏不同时（比如 gpu 列的噪声参照 N 是另一组运行标定的），括号里另写这一列自己的指标与 N，以格子里的为准。",
    "- 后面是严格档符号，「｜参」后面是参考档符号；没有「｜参」就是参考档无对应门槛。",
    "- 选中多条检查的格子写最差的那条。",
    "- 每组表下面的「说明」逐条写非 ✅ 格子的完整推导（实测｜严格档｜参考档）。",
    "- 说明里「N=…（k 对）」的 k 是算 N 用的噪声参照对数：",
    "  - 一对是两份运行的一次比较，k 对不是 k 份运行；",
    "  - 本仓库各目录的噪声对都是若干份运行两两配对的全部组合"
    "（3 对 = 3 份、6 对 = 4 份、10 对 = 5 份、15 对 = 6 份、28 对 = 8 份、45 对 = 10 份）；",
    "  - 第 1 步整步现行只用 GPU 第 1 步重跑 3 对（2026-11-08 起）；"
    "旧结果里的 24 对是已撤回的两侧执行噪声并集（GPU 3 份两两 3 对 + TPU 7 份两两 21 对），只留作判法变更记录；",
    "  - 用了哪几份见各目录的 `runs.noise`。", "",
    "### 符号", "",
    "- 严格档：✅ 过　🟡 待定或进行中　❌ 不过或出错　⬜ 未开始　— 不适用",
    "- 参考档：参✅ 过　参☑️ 记录了的都过、有门槛项没记录　参❌ 不过　参❔ 所需指标未记录", "",
    "### GPU 列里的 ❌", "",
    "- 不是工具出错：它们是只换执行方式（专家并行度、梯度累积切分、激活重算）时实测到的真实差异。",
    "- 说明这些设置本身就是数值的一部分，原因与处理见 `alignments/00-gpu-noise-and-drills/README.md`。",
    "- 故障演练单独成表，在最后。", "",
]



def main():
    Overview(
        ROOT,
        "混元 3",
        HOW_TO_READ,
        GOLDEN_NOTE,
        "**本页结构**：怎么读这张总览 → 各列是什么 → L0–L4 各组检查（每组表下有「说明」）→ 故障演练 → 汇总。",
        faults_file=FAULTS,
        fault_codes=FAULT_CODES,
        legacy_figs=LEGACY_FIGS,
        series_compare=SERIES_COMPARE,
        checks_dirs=["01-v5p-acceptance", "02-v7x-acceptance"],
    ).main()


if __name__ == "__main__":
    main()
