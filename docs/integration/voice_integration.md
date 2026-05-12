# 语音交互模块

## Event

| Event.type | 语义 | 边界 |
|------------|------|------|
| `voice_wake_detected` | 检测到唤醒词。 | 只表示语音链路状态变化。 |
| `voice_input_started` | 开始录音。 | 只表示采集开始。 |
| `voice_input_stopped` | 结束录音。 | 只表示采集结束。 |
| `speech_recognized` | ASR 识别完成，产出文本。 | 是推荐主事件；只上报识别结果，不做意图判断。 |
| `tts_started` | TTS 开始播报。 | 只表示执行状态。 |
| `tts_finished` | TTS 播报结束。 | 只表示执行状态。 |
| `voice_volume_changed` | 音量偏好变化。 | 只表示参数变化。 |
| `voice_timbre_changed` | 音色偏好变化。 | 只表示参数变化。 |
| `voice_speed_changed` | 语速偏好变化。 | 只表示参数变化。 |

## Action

| Action.type | 语义 | 边界 |
|-------------|------|------|
| `start_voice_capture` | 开始语音采集。 | 只负责打开语音输入链路。 |
| `stop_voice_capture` | 停止语音采集。 | 只负责关闭语音输入链路。 |
| `set_tts_voice` | 设置音色。 | 只改 TTS 参数。 |
| `set_tts_volume` | 设置音量。 | 只改 TTS 参数。 |
| `set_tts_speed` | 设置语速。 | 只改 TTS 参数。 |
| `speak` | 播报文本。 | 是推荐主 Action；提醒/关怀/异常语义放在 payload 的 `kind` / `level` / `reason` 中。 |

无动作场景不属于语音 Action。当前系统统一用空 `Action` 列表表示“不执行”，不再定义 `none`。

建议的 `speak.payload` 字段：

- `text`
- `interrupt`
- `voice`
- `volume`
- `speed`
- `emotion`
- `kind`
- `level`
- `reason`

## Adapter

| Adapter | 责任 |
|---------|------|
| `mic_input.py` | 处理唤醒、录音、VAD、ASR，并发出语音输入事件。 |
| `tts_output.py` | 消费 `speak` 和 TTS 设置动作，执行播报。 |
| `voice_settings_adapter.py` | 处理音量、音色、语速设置。 |
