# 行为与姿势感知模块

## Event

| Event.type | 语义 | 生产者 |
|------------|------|--------|
| `user_presence_updated` | 用户是否在场 | `behavior/` YOLO26、摄像头逻辑 |
| `user_attention_updated` | 注意力与行为线索 | 同上 |
| `user_posture_updated` | 坐姿/姿势 | 同上 |
| `user_activity_updated` | 活动类型（如使用手机） | 同上 |

## 分流

以上事件在 `EventRouter` 中归类为 **`state_only`**：只更新 `AgentState` 和 `RuntimeHistory`，**永不直接走 LLM**。

## 决策链路

每 20 秒的 `behavior_distraction_check`（`system_triggered`）触发 `BehaviorDistractionHandler`：

- Python：`summary_builder.build_behavior_distraction_summary()` 汇总注意力/姿势/在场等窗口信号
- LLM：结合 `RuntimeHistory` 与 `Memory` 偏好判断是否提醒
- 落地：`speak` + `display`（`ActionRealizer` → `DeviceAdapter`）

旧入口 `periodic_state_check` / `build_periodic_state_summary` 已删除。
