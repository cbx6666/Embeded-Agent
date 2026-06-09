# Event

## 职责

`event/` 定义进入 AgentCore 的标准事件模型与分流器。

## 核心文件

- `event_model.py`：统一 Event dataclass
- `types.py`：EventType 闭集
- `event_builders.py`：标准 Event 构造函数
- `router.py`：`EventRouter` 分流

## 分流类型

- `speech_llm`：`speech_recognized`
- `behavior_distraction`：`system_triggered` + `trigger=behavior_distraction_check` + `source=agent_autonomy`
- `wellness_care`：`system_triggered` + `trigger=wellness_care_check` + `source=agent_autonomy`
- `environment_care`：`system_triggered` + `trigger=environment_care_check` + `source=agent_autonomy`
- `sensor_status`：`system_triggered` + `trigger=sensor_status_report` + `source=agent_autonomy`
- `rule`：`focus_start_requested` / `focus_stop_requested` / `timer_finished`
- `state_only`：其余事件（含已废弃删除的 `periodic_state_check`），只更新 State / RuntimeHistory
