# 环境感知模块

## Event

| Event.type | 语义 | 生产者 |
|------------|------|--------|
| `light_level_updated` | 光照等级 | ESP32/STM32、`adapters/environment/` |
| `temperature_humidity_updated` | 温湿度 | 同上 |
| `noise_level_updated` | 噪声等级 | 同上 |

## 分流

三类环境事件在 `EventRouter` 中归类为 **`state_only`**：只更新 `AgentState` 和 `RuntimeHistory`，**跳过 LLM**。

## 决策链路

每 60 秒的 `environment_care_check` 触发 `EnvironmentCareHandler`：

- Python：`summary_builder.build_environment_care_summary()` 汇总异常环境（含 `abnormal` 列表）
- LLM：判断是否输出 `adjust_environment_feedback`
- 落地：经 `Guard` 后 `speak` + `display`

旧 trigger（`environment_check`、`periodic_state_check` 等）已删除；环境判断不再并入单一周期 LLM 入口。
