# 情绪与疲劳感知模块

## Event

| Event.type | 语义 | 边界 |
|------------|------|------|
| `user_emotion_updated` | 当前情绪状态。 | 只上报识别事实，不直接输出关怀动作。 |
| `user_fatigue_updated` | 当前疲劳等级。 | 只上报识别事实，不直接输出休息提醒。 |

建议的闭集：

- `emotion`: `neutral` / `tired` / `stressed` / `happy`
- `fatigue_level`: `none` / `mild` / `moderate` / `high`

当前阶段范围：

- 支持视觉输入和 mock 输入
- 不覆盖语音情绪特征
- 不覆盖多模态融合

## Action

本模块当前无专用 Action。

说明：

- 情绪和疲劳模块是感知模块，只负责发 Event。
- 关怀提醒、休息建议、异常提示由内核转换成显示、语音、灯光模块的能力 Action。

## Adapter

| Adapter | 责任 |
|---------|------|
| `VisionAffectInputAdapter` | 从摄像头或视觉模型输出中识别情绪和疲劳，并发出标准事件。 |
| `mock_input.py` | 提供情绪和疲劳的 mock 输入。 |
