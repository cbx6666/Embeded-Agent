# Agent

`agent/` 是嵌入式 Agent 的主领域包，主链路为：

`Event -> Reducer -> EventPriorityRouter -> Decision Scheduling -> Rule/LLM -> Validator -> Guard -> ActionRealizer -> Execution`

核心边界：

- `state/`：Agent 当前运行状态（含 `RuntimeHistory` 短期历史），不保存稳定偏好。
- `memory/`：系统从长期交互中学习到的可证据化长期记忆。
- `user/`：用户认知层，包含静态显式 `UserProfile` 和动态决策快照 `PersonalContext`。
- `decision/`：只消费 `PersonalContext`，不直接读取任何 store。
- `scheduling/`：只按系统时间产生低频 P1 检查，不直接调用 LLM。
- `execution/`：执行 `Action`、记录 `ActionResult` 和轻量 `RuntimeTrace`。

事件原则：

- 事实进入 State，并形成有界 rolling summary。
- 请求进入 Decision；P0B 走规则，P0A 才走语义规划。
- P1 由 Scheduler 低频触发，再经过 AutonomousCheckPolicy gate。
- 反馈进入 RuntimeHistory / 异步 Memory，不自动生成动作。
- 动作统一经过 Validator、Guard、ActionRealizer 和 Execution。

Authoritative Source：

- 显式资料和显式偏好只能来自 `UserProfile`。
- 行为偏好和行为模式只能来自 `LongTermMemory`。
- 最近对话、最近动作和传感器滚动采样只能来自 `RuntimeHistory`。
- 决策上下文只能来自 `PersonalContextBuilder`。
