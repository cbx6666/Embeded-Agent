# State

`state/` 保存 Agent 当前运行状态 dataclass。状态由 reducer 确定性更新，并由
`JsonStore` 持久化。

本目录不做 LLM 推理，不生成长期记忆，不写用户画像，也不执行动作。

核心文件：

- `agent_state.py`：组合运行状态、当前用户和 `RuntimeHistory`。
- `user_state.py`：用户 presence、attention、emotion、fatigue。
- `focus_state.py`：专注计时状态。
- `interaction_state.py`：对话与交互状态。
- `environment_state.py`：环境传感器状态。
- `cooldown_state.py`：提醒和通知类动作的冷却时间记录。
- `runtime_history.py`：短期运行窗口。

短期历史模型 `RuntimeHistory` 在本目录 `agent/state/runtime_history.py`；显式用户画像模型在
`agent/user/user_profile.py`。
