# State

## 职责

`state/` 保存 Agent 的运行时状态 dataclass。状态由 reducer 确定性更新，并由 JsonStore 持久化。

## 不负责什么

本目录不做 LLM 推理，不生成长期记忆，不写用户画像策略，也不执行动作。

## 核心文件

- `agent_state.py`：组合所有子状态。
- `user_state.py`：用户 presence、attention、emotion、fatigue。
- `focus_state.py`：专注计时状态。
- `interaction_state.py`：对话/交互状态。
- `environment_state.py`：环境传感器状态。
- `memory_state.py`：短期工作集。
- `user_profile_state.py`：长期 profile 数据结构。

## 上游和下游

上游是 reducer 和 UserProfileService。下游是 AgentContextBuilder、MemoryContextBuilder、DeviceAdapter trace。

## 扩展方式

新增状态字段：只在对应 dataclass 中添加字段，并在 reducer 中明确哪类 Event 可以更新它。

## 示例

```python
state = reduce_state(state, event)
```
