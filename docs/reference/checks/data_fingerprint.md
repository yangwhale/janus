[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `data_fingerprint`

# data_fingerprint

**用途**：数据错位、续训接不上。

**输入**：

- 两份 {单元: sha256}，单元是「步.r<rank>.m<microbatch>.<字段>」；
- 另有每步全局 `nonpad` 总和；
- 可选首步 loss（被测, 参照）；
- `params.fields` 是期望字段，默认 `tokens`、`labels`、`loss_mask`、`position_ids`。

**判据**：

- 任一侧任一条记录缺期望字段 → `error`（缺字段等于少比了，不能报一致；不跳过）。
- 窗口内某一步一侧有记录、另一侧一条都没有（缺整步，如被测步数比窗口短）→ `error`，消息列出缺的步。
  - 同一步里只缺部分 rank / microbatch 不算缺整步，按单元不同报 `fail`。
- 否则每个单元逐位相同且首步 loss 逐位相同 → `pass`。
  - 首步 loss 取被测 `meta.steps.first` 与参照上换算后的同一步（见 `loss_series` 的步号约定）。
  - 跨平台比较（GPU 对 TPU）首步 loss 从来不逐位，要设 `first_loss: false`，首步差交给 L3 / L4；不关的话这一项必然 `fail`，看起来像数据没对上。
- 步号：只有首步 loss 换算；各单元按两侧各自的步号配对，`steps` 窗口两边用同一个步号，不换算。两侧步号约定不是恒等（例如一侧从 0 起，或 GPU 带 `skip_steps`）时，会拿错开一步的数据比，报出来像数据错位；janus 没有给参照一侧单写步号的字段，这种运行先用同一约定重跑再比。

**输出**：

- `units_per_field`（每个字段比了多少单元）、`units`、`differ`、`first_loss`。
- 明细每个单元一行，给 sha256 前 16 位。
- 缺字段时 `missing_fields` 与前几处缺失位置。
- 缺整步时 `missing_steps`（`{test: [步…], ref: [步…]}`，只列有缺的一侧）。
