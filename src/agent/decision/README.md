# Decision

## 职责

`decision/` 包含多个决策处理器，按 `EventRouter` 分流结果分别处理：

| 处理器 | 分流 | LLM | 输出 |
|--------|------|-----|------|
| `SpeechLLMHandler` | `speech_llm` | `speech_recognized` | Intent + Action |
| `BehaviorDistractionHandler` | `behavior_distraction` | `behavior_distraction_check` | 提醒文案（经 Guard） |
| `WellnessCareHandler` | `wellness_care` | `wellness_care_check` | 疲劳/情绪/姿态关怀（经 Guard） |
| `EnvironmentCareHandler` | `environment_care` | `environment_care_check` | 环境关怀（经 Guard） |
| `SensorStatusHandler` | `sensor_status` | 无（确定性播报） | speak/display 或 no_op |
| `RuleHandler` | `rule` | 无 | Intent + Action |

链路：`Handler.decide() -> Intent -> ActionRealizer.realize() -> Action[]`

## 核心文件

- `speech_llm_handler.py`：`answer_user / start_focus / stop_focus / set_tts_volume / no_op`
- `behavior_distraction_handler.py`：玩手机分心 Python 预检 + LLM 措辞 + Guard
- `wellness_care_handler.py`：`build_wellness_care_summary` + LLM + Guard
- `environment_care_handler.py`：`build_environment_care_summary` + LLM + Guard
- `sensor_status_handler.py`：传感器数值播报，确定性边界（away/speaking/刚说过话），不调 LLM
- `rule_handler.py`：专注与计时器规则

## 上游和下游

- 上游：`AgentCore` 传入 `Event`、`AgentState`、`LLMClient`、`user_context`（各 LLM 入口）
- 下游：`action/realizer.py` → `device/adapter.py`

## 示例

```python
result = speech_handler.decide(
    state=state,
    event=event,
    llm_client=llm_client,
    user_context={"memories": {...}, "profile": {...}},
)
```
