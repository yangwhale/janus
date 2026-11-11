[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 13

# 案例 13：从 BF16 检查点起训，权重侧的精度开关第 1 步是空操作，要用训练后权重或多步 A/B 判

> **状态：机理已查明；训练后权重的重放与主权重离网比例已实测。**

## 现象

以 T1 yml 全开（同 `t1zu-a`，MaxText `3741e818`）为底，只关输出头精度开关 M13：

- 第 1 步前向输出头 logits 开关前后逐位相同（`tw-sw-fwd2`）；M27 注意力整块、M8 路由 logits 与 MoE 整块也逐位相同。
- 第 1 步反向 `lm_head.weight` 梯度 TPU 对 GPU 的 rel_L2 开 1.81e-3、关 1.81e-3（GPU 三份之间 1.01e-3）；开关前后两次 TPU 梯度之间只差 5.7e-5。第 1 步 loss 都是 12.889265（`tw-sw-base` / `tw-sw-m13`）。
- 300 步（对 `g2-a`）：主 loss 最大单步 N 4.53e-5，开 TPU 2.75e-5、r 0.61，关 TPU 2.73e-4、**r 6.03**；param_norm 带符号均值 N 1.90e-7，开 TPU −1.09e-7、r 0.57，关 TPU −4.71e-7、**r 2.48**（`tq-base` / `tq-m13`）。

## 机制

- hy4-mini 的随机初始化检查点照发布检查点存 dtype：输出头、索引器 `weights_proj`、路由权重等存 BF16，iHC、hc_head、主干路由偏置与 sink 存 FP32（[`model/make_tiny_hf.py`](../model/make_tiny_hf.py)）。载入后主权重是 FP32，但值全在 BF16 格点上；输出头的输入是最后那个 norm 的 BF16 输出，也在格点上。
- HIGHEST 把 FP32 操作数拆成高、低两部分 BF16 分几遍乘。操作数在格点上时低位全是 0，多出的几遍贡献为 0，剩下那一遍 BF16 × BF16 在 FP32 累加里是精确的，所以 M13 第 1 步逐位不变。「先舍 BF16 再乘」（M27、M8）对格点上的权重是恒等变换，同理。
- 第 1 步更新后主权重离开格点，default 精度开始截掉权重低位，两条轨迹分开；到 300 步放大成 r 6.03 是推断。

## 证据

- **离网比例**（`tgrid-b`，T1 全开 300 步，lm_loss 与 `t1zu-a` 逐位相同；[`t1-bf16-offgrid-frac.json`](../alignments/01-mini/results/t1-bf16-offgrid-frac.json)）：存 BF16 的矩阵第 0 步离网比例 0；第 1 步更新后输出头 1、路由权重 0.9994、索引器 0.9994、注意力 0.993、路由专家 0.9995。嵌入只更新出现过的行，第 1 步 0.090、300 步 0.520。iHC、hc_head 第 0 步就是 1。盲区只有第 1 步那一次前向。
- **训练后权重的重放**（`tw-sw-ovr`，换上 GPU `g2o-a` 第 8 步的 FP32 主权重，输入仍是第 1 步算子输入；`analysis/weight_override_refs.py`）。表中是对两种 FP64 参照的 rel_L2：FP32 权重 / 同一份权重先舍 BF16：

| 开关 | 开 | 关 | 开时等于 |
|---|---|---|---|
| M13 输出头 logits | 8.3e-8 / 2.73e-3 | 2.73e-3 / 8.2e-8 | FP32 权重 |
| M27 weights_proj（层 0、1、5、MTP） | 9.6e-4–1.25e-3 / 8e-8 | 8e-8 / 9.6e-4–1.24e-3 | BF16 权重 × FP32 输入 |
| M8 路由 logits（层 1–6、MTP） | 1.41e-3–2.46e-3 / 8e-8–1.2e-7 | 8e-8–1.2e-7 / 约 1e-3 | BF16 权重 |

- **度量要看位**：`x != f32(bf16(x))` 在 TPU 上判不出离网（按 FP32 存的 iHC 也只判出 6.1e-5）；看低 16 位是否全 0，iHC 第 0 步就是 1；CPU 上两种判法结果相同。v5p 优化后 HLO 里那对 convert 仍在，推断往返是在 TPU 融合的代码生成里被 excess precision 省掉的。
- **模拟舍入用 reduce_precision**：FP32 → BF16 → FP32 的 astype 往返会被省掉：M42b 用它舍 dp 时 CPU 单测 dsink 对 FP64 只差 2.6e-7（按 GPU 写法应是 9.8e-4），用 `jax.lax.reduce_precision` 后 9.80e-4。省在 HLO 优化还是代码生成没有查。

## 结论

- **方法**：从 BF16 检查点起训时，第 0 / 1 步逐位相同不能当作「开关没用」的证据；L3 第 1 步比较只为激活与梯度一侧的舍入定责。权重侧开关用训练后权重的单算子重放或多步单开关 A/B 判（PROTOCOL 变更记录 2026-11-08）。判一个 FP32 值在不在 BF16 格点上看位；在 JAX 里模拟一次舍入用 reduce_precision。
- **适用范围**：受影响的是只作用在权重一侧舍入或精度的开关，本仓库是 M13、M27、M8（[r5](../ROOT-CAUSES.md#r5)、[r2](../ROOT-CAUSES.md#r2)）；作用在激活或梯度一侧的（M10、M12、M23）第 1 步照样量得到。主权重从 FP32 随机初始化直接起训，或检查点按 FP32 存的参数（iHC、hc_head、路由偏置、sink），不受影响。
- **定责**：只改 TPU 一边的 A/B——两次 300 步运行只差 M13，关掉后主 loss 最大单步 r 0.61 → 6.03（N 4.53e-5，TPU 2.75e-5 → 2.73e-4）。读代码对照语义——重放里开关把 TPU 放到 GPU 那种 FP64 上（1e-7 量级）：输出头 GPU 是真 FP32，weights_proj 与路由用 BF16 模型权重（台账 G26、M13、M27、M8，[CHANGES-LEDGER](../CHANGES-LEDGER.md)），所以三个开关都要开，T1 都开着。
