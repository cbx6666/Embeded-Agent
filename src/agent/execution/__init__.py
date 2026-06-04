"""Agent 执行层 (execution layer) 公开入口。

本包的职责是 **执行层面** 的闭环和控制，不涉及 LLM 认知：

- `loop.py`：多步内部事件闭环 (AgentLoop)，防止动作结果丢失。
- `device_adapter.py`：动作的确定性执行边界 (DeviceAdapter)，把 Action 变成真实调用。
- `action_result.py`：动作执行结果模型 (ActionResult)，记录成功/失败/原因。
- `internal_events.py`：把动作结果映射为内部 Event，回流到 AgentCore 再处理。
- `autonomous_tick.py`：构造自主检查事件 (periodic_check / focus_health_check 等)，作为
  自主 tick 的事件源。
- `trace.py`：调试 trace 数据结构 (LoopTrace)，记录每轮闭环的 intents/actions/results。

本包不做 LLM 规划、不生成 IntentPlan、不写长期记忆，也不包含设备语义策略。
上游是 AgentCore 和设备执行结果；下游是内部 Event 队列、CLI trace 和测试。

名字说明：
取名 `execution/` 而非 `runtime/` 或 `loop/`，是因为本包专注于动作执行、结果回流、
自主检查和闭环循环，是 Agent 的执行层。`execution/` 精确表达了"把决策变成实际动作
并回收结果"的职责边界。"""
