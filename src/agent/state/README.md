# State

`state/` 保存 Agent 当前运行状态。状态由 `reducer.py` 确定性更新，并由 `JsonStore` 持久化。

本目录不做 LLM 推理，不生成长期记忆，不写用户画像，也不执行动作。

## 核心文件

| 文件 | 职责 |
|------|------|
| `agent_state.py` | 组合运行状态、当前用户和 `RuntimeHistory` |
| `user_state.py` | presence、attention、emotion、fatigue、posture、activity |
| `focus_state.py` | 专注计时状态 |
| `interaction_state.py` | 对话与交互状态 |
| `environment_state.py` | 环境传感器状态 |
| `cooldown_state.py` | 提醒和周期检查的冷却时间 |
| `runtime_history.py` | 短期运行窗口 + `RuntimeHistoryService` |
| `reducer.py` | `reduce_state(state, event)` 事件归约 |
| `summary_builder.py` | `build_behavior_distraction_summary`、`build_wellness_care_summary`、`build_environment_care_summary`、`build_sensor_status_summary` |

## wellness_care_summary

`wellness_care_check` 进入 LLM 前，`summary_builder` 在窗口内合并 fatigue / emotion / posture：

- `fatigue_level`、`fatigue_confidence`
- `emotion`、`emotion_confidence`
- `posture`（持续坏姿态 / 占比触发，且当前仍为坏姿态）
- `should_care` 与 selected intent（强触发由 Python 决定，LLM 不能改成 no_op）
- `dominant_signal`：`fatigue | emotion | posture | none`
- `reason`

> 旧的 `build_wellness_summary` / `build_periodic_state_summary` 随 `periodic_state_check` 一并废弃删除。

## 上游和下游

- 上游：adapters 产生的 `Event`
- 下游：`EventRouter`、决策 handler、`Guard`、`summary_builder`、LLM prompt

显式用户画像模型在 `services/user_profile_model.py`，不在本目录。
