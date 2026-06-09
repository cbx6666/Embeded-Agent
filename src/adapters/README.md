# adapters

`adapters/` 是输入输出适配层，负责把外部世界转换成系统可处理的事件，或把系统动作转换成具体输出。

## 职责

- 接收用户输入（CLI、mock、摄像头、麦克风、环境传感器等）
- 解析 mock 命令
- 执行控制台输出、显示输出、语音输出
- 为硬件联调提供边界

## 与 Agent 的协议

**Event**：adapters 只产生 `src/agent/event/types.py` 中注册的 20 种事件。已删除 `user_text_input`、`display_sensor_updated` 等，CLI 不再把任意文本当事件注入。

**Action**：adapters 只执行 5 种动作：`speak`、`display`、`start_timer`、`stop_timer`、`set_tts_volume`。已删除 `render_pet_expression`、`set_light_state`、`start_voice_capture`、`stop_voice_capture` 等。

## 文件说明

| 路径 | 责任 |
|------|------|
| `cli_input.py` | 解析 `/focus start`、`/focus stop` 等结构化命令；普通文本不再产生事件 |
| `mock_input.py` | 将 `/mock ...` 转换为状态更新事件 |
| `console_output.py` | 执行 `speak`、`display`、`set_tts_volume` |
| `screen/screen_adapter.py` | 桌宠显示；只消费 `display` |
| `screen/screen_adapter.py` | 屏幕输出；只消费 `display` |
| `voice/` | 板级语音（百度 ASR/TTS、唤醒词、麦克风仲裁）；上报 `speech_recognized` 等语音事件，消费 `speak` / `set_tts_volume` |
| `behavior/` | YOLO26 行为/姿势；上报 `user_presence_updated` / `user_attention_updated` / `user_posture_updated` / `user_activity_updated` |
| `vision_affect/` | 疲劳（EAR/MAR）与情绪；上报 `user_fatigue_updated` / `user_emotion_updated` |
| `environment/` | ESP32/STM32 环境传感器；上报 `light_level_updated` 等 |

## 视觉依赖

见根目录 `requirements.txt`。启动示例：

```bash
python -m src.main
python -m src.main --llm
python -m src.main --llm --vision --emotion-backend raf --raf-ckpt path/to/raf_resnet18.pth
```

环境变量 `EMBED_EMOTION_BACKEND`、`EMBED_PERCEPTION_HZ` 可覆盖部分运行时参数。

## 原则

外部接口可以变化，但进入 Agent 的 Event 和 Action 必须落在注册闭集内。适配层不修改 `AgentState`，不直接调用决策逻辑。
