# 情绪与疲劳感知模块

## Event

| Event.type | 语义 | 生产者 |
|------------|------|--------|
| `user_fatigue_updated` | 疲劳档位与置信度 | `vision_affect/`（EAR/MAR 等） |
| `user_emotion_updated` | 情绪与置信度 | `vision_affect/`（WuJie-OM、RAF、DeepFace 等） |

## 分流

两类事件在 `EventRouter` 中归类为 **`state_only`**：只更新 State/History，**不走 LLM**。

## wellness_care_check（周期汇总）

每 30 秒的 `wellness_care_check` 触发 `WellnessCareHandler`：

- Python：`summary_builder.build_wellness_care_summary()` 合并 fatigue、emotion、姿态等窗口信号，算出 `should_care` 与关怀方向
- LLM：一次 `build_wellness_prompt` 决策；无异常时输出 `no_op`
- 落地：`suggest_rest` / `offer_emotion_care` 等 → `speak` + `display`

旧入口 `periodic_state_check` / `build_wellness_summary` 已删除。
