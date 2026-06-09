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

## Action（当前闭集）

语音采集由 `VoiceRuntime` 内部状态机驱动，**不再**通过 `start_voice_capture` / `stop_voice_capture` Action 暴露。

Agent 对外可执行的语音相关 Action：

| Action.type | 语义 | 边界 |
|-------------|------|------|
| `speak` | 播报文本。 | 提醒/关怀/异常语义放在 payload 的 `kind` / `level` / `reason` 中。 |
| `set_tts_volume` | 设置音量。 | 只改 TTS 参数。 |

已删除且不再生成/执行：`start_voice_capture`、`stop_voice_capture`、`set_tts_voice`、`set_tts_speed`。

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

## 唤醒应答文案

默认应答文案的权威来源为 `src/adapters/voice/wake/local_wake_ack.py` 中的 `DEFAULT_WAKE_ACK_TEXT` 与 `DEFAULT_WAKE_ACK_PHRASES`。`VoiceRuntime`、`BoardVoiceAdapter` 与 `main.py --wake-ack` 均引用该常量，不在多处硬编码 fallback。

## Adapter（当前实现）

| 模块 | 责任 |
|------|------|
| `board_voice_adapter.py` | 对外 facade，委托 `VoiceRuntime`。 |
| `runtime/voice_runtime.py` | 编排唤醒、VAD、ASR、TTS 与 Agent 事件桥。 |
| `wake/local_wake_ack.py` | 本地唤醒应答 WAV 预加载与默认文案。 |
| `input/audio_input_manager.py` | 唯一 capture 流（单 arecord + ring buffer）。 |
| `tts/playback_manager.py` | 统一 TTS 队列（唤醒应答 / Agent / 自主提醒）。 |
| `wake/detector.py` | 唤醒词检测（仅 `feed_audio`，不独立 arecord）。 |
