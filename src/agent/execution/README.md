# Execution

## 职责

`execution/` 负责 Agent 的动作执行、闭环、结果回流和 trace。

## 不负责什么

本目录不做 LLM 认知、不生成 IntentPlan、不写长期记忆，也不包含设备语义策略。

## 核心文件

- `loop.py`：多步内部事件闭环，避免动作结果丢失。
- `device_adapter.py`：确定性执行边界。
- `action_result.py`：动作执行结果模型。
- `internal_events.py`：把动作结果转成内部 Event。
- `trace.py`：调试 trace 数据结构。
- `autonomous_tick.py`：构造自主检查事件。

## 上游和下游

上游是 AgentCore 和设备执行结果。下游是内部 Event 队列、CLI trace 和测试。

## 扩展方式

新增动作结果反馈：在 `internal_events.py` 中添加明确的结果到 Event 映射，并补闭环测试。

## 示例

```python
actions = loop.run_once(event)
```
