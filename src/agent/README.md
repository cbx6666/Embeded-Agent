# Agent

`agent/` 是嵌入式 Agent 的主领域包，主链路为：

`Event -> RuntimeHistory -> LongTermMemoryPipeline -> PersonalContextBuilder -> DecisionPipeline -> Action`

核心边界：

- `history/`：当前运行期短期历史，不保存稳定偏好。
- `memory/`：系统从长期交互中学习到的可证据化长期记忆。
- `user/`：用户认知层，包含静态显式 `UserProfile` 和动态决策快照 `PersonalContext`。
- `decision/`：只消费 `PersonalContext`，不直接读取任何 store。

Authoritative Source：

- 显式资料和显式偏好只能来自 `UserProfile`。
- 行为偏好和行为模式只能来自 `LongTermMemory`。
- 最近对话、最近动作和传感器滚动采样只能来自 `RuntimeHistory`。
- 决策上下文只能来自 `PersonalContextBuilder`。
