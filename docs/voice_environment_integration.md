# 语音与环境感知：我们要负责的 Event、Action 和 Adapter

本文只讲**和「语音输入输出、光照/温湿度/噪声环境感知」**相关的内容。  
更细的字段闭集、上报时机和动作 payload 约定见下文表格与示例。

---

## 一、我们这条线要负责什么

| 类别 | 我们关心的部分 |
|------|----------------|
| **Event（事件）** | 语音唤醒、语音采集开始/结束、语音识别结果、TTS 开始/结束、音量调整、音色调整、语速调整、光照更新、温湿度更新、噪声更新 |
| **Action（动作）** | 开始语音采集、结束语音采集、设置 TTS 音色、设置 TTS 音量、设置 TTS 语速、环境提醒输出 |
| **Adapter（适配器）** | 麦克风输入适配器、语音播报输出适配器、环境传感器输入适配器 |

**边界**：唤醒词检测、VAD、ASR、TTS 引擎、音频驱动、传感器采样、滤波、防抖、阈值判定都属于 `src/adapters/` 及设备侧实现；本阶段只定义标准 `Event` / `Action` 契约，不修改 `reducer.py`、`policy.py`、`core.py`、`state/`、`memory_service.py`。

---

## 二、白话：我们在做什么、逻辑是什么

1. 麦克风或语音前端检测到唤醒词后，上报 `voice_wake_detected`，让内核知道用户希望开始语音交互。
2. 语音采集链路开始录音时，上报 `voice_input_started`；结束录音时，上报 `voice_input_stopped`，便于内核维护对话阶段。
3. ASR 把用户语音转成文字后，上报 `speech_recognized`，其中携带识别文本、置信度、来源和可选会话信息。
4. 语音配置变化（音量、音色、语速）用独立事件上报，方便内核决定是否记忆用户偏好，或给出确认反馈。
5. 光照、温湿度、噪声传感器各自独立上报标准事件，让内核按业务规则触发“光线不足提醒”“空气干燥提醒”“环境噪声提醒”“温度过低提醒”等动作。
6. 内核需要设备执行时，输出标准 `Action`，再由语音/传感器相关 adapter 消费并落到真实设备。

整体逻辑一句话：**Adapter → Event → 内核 → Action → Adapter**

---

## 三、Event 有哪些（与本业务线直接相关）

| Event（`type`） | 作用（白话） | 典型谁产生 |
|-----------------|-------------|------------|
| `voice_wake_detected` | 检测到唤醒词，开始一轮语音交互 | 麦克风唤醒词适配器 |
| `voice_input_started` | 开始采集用户语音 | 麦克风 / VAD 适配器 |
| `voice_input_stopped` | 结束采集用户语音 | 麦克风 / VAD 适配器 |
| `speech_recognized` | 语音识别得到文本 | ASR 适配器 |
| `tts_started` | TTS 开始播报 | TTS 输出适配器 |
| `tts_finished` | TTS 播报结束 | TTS 输出适配器 |
| `voice_volume_changed` | 用户音量偏好发生变化 | 语音设置适配器 / 语音指令适配器 |
| `voice_timbre_changed` | 用户切换音色 | 语音设置适配器 / 语音指令适配器 |
| `voice_speed_changed` | 用户调整语速 | 语音设置适配器 / 语音指令适配器 |
| `light_level_updated` | 光照传感器上报最新光照 | 光照传感器适配器 |
| `temperature_humidity_updated` | 温湿度传感器上报最新数值 | 温湿度传感器适配器 |
| `noise_level_updated` | 噪声传感器上报最新噪声值 | 噪声传感器适配器 |

（仓库里已有类型以 `src/agent/event/types.py` 为准。）

### Event payload 建议

- `voice_wake_detected`
  - 必填：`keyword`、`source`
  - 可选：`confidence`、`session_id`
- `voice_input_started`
  - 必填：`source`
  - 可选：`session_id`、`trigger`
- `voice_input_stopped`
  - 必填：`source`
  - 可选：`session_id`、`reason`、`duration_ms`
- `speech_recognized`
  - 必填：`text`、`source`
  - 可选：`confidence`、`session_id`、`language`、`is_final`
- `tts_started`
  - 必填：`source`
  - 可选：`text`、`voice_id`、`session_id`
- `tts_finished`
  - 必填：`source`
  - 可选：`text`、`voice_id`、`session_id`、`status`
- `voice_volume_changed`
  - 必填：`volume`、`source`
  - 说明：`volume` 建议为 `0~100`
- `voice_timbre_changed`
  - 必填：`voice_id`、`source`
  - 可选：`style`
- `voice_speed_changed`
  - 必填：`speed`、`source`
  - 说明：`speed` 建议为 `0.5~2.0`
- `light_level_updated`
  - 必填：`light_lux`、`source`
  - 可选：`level`、`is_low_light`
- `temperature_humidity_updated`
  - 必填：`temperature_c`、`humidity_pct`、`source`
  - 可选：`temperature_level`、`humidity_level`
- `noise_level_updated`
  - 必填：`noise_db`、`source`
  - 可选：`level`、`is_noisy`

---

## 四、Action 有哪些（本业务线场景下会怎么用）

| Action（`type`） | 作用（白话） | 在本业务线中 |
|------------------|-------------|--------------|
| `start_voice_capture` | 让设备开始采集用户语音 | 唤醒后启动录音，或引导用户继续说话 |
| `stop_voice_capture` | 让设备结束语音采集 | 识别完成、超时、取消交互时停止录音 |
| `set_tts_voice` | 设置 TTS 音色 | 切换男声/女声/儿童音等具体音色 |
| `set_tts_volume` | 设置 TTS 音量 | 用户说“声音调大一点”后应用到播报链路 |
| `set_tts_speed` | 设置 TTS 语速 | 用户说“说慢一点/快一点”后更新播报语速 |
| `environment_alert` | 输出环境提醒 | 温度过低、空气干燥、光线不足、噪声过大时由相关 adapter 落地 |
| `speak` | 语音播报文本 | 与 TTS 适配器配合输出自然语音 |
| `display` | 屏幕或控制台显示文本 | 无法播报或需要同时展示时使用 |

（仓库里已有类型以 `src/agent/action/types.py` 为准。）

### Action payload 建议

- `start_voice_capture`
  - 必填：`source`
  - 可选：`trigger`
- `stop_voice_capture`
  - 必填：`source`
  - 可选：`reason`
- `set_tts_voice`
  - 必填：`voice_id`
- `set_tts_volume`
  - 必填：`volume`
- `set_tts_speed`
  - 必填：`speed`
- `environment_alert`
  - 必填：`sensor`、`level`、`message`
  - 说明：`sensor` 建议闭集为 `light` / `temperature` / `humidity` / `noise`

---

## 五、Adapter 有哪些（作用是什么）

| Adapter | 作用（白话） | 与本业务线的关系 |
|---------|-------------|------------------|
| `mic_input.py`（建议新增） | 接收唤醒词、录音状态、ASR 结果并转成标准事件 | 负责发出语音输入相关 Event |
| `tts_output.py`（建议新增） | 消费 `speak`、`set_tts_*` 等动作并调用真实 TTS 引擎 | 负责落地语音输出与播报参数设置 |
| `environment_sensor_input.py`（建议新增） | 读取光照、温湿度、噪声传感器并转成标准事件 | 负责发出环境感知相关 Event |
| 现有 `console_output.py` | 在 MVP 阶段可作为环境提醒和语音提醒的兜底输出 | 便于在无硬件时联调 Action |

---

## 六、与项目组 / 内核的衔接

- 本 PR 是否**新增或修改** `Event` / `Action` 类型：**是**；新增类型如下。  
  - `Event.type`
    - `voice_wake_detected`：`keyword`、`source`，可选 `confidence`、`session_id`
    - `voice_input_started`：`source`，可选 `session_id`、`trigger`
    - `voice_input_stopped`：`source`，可选 `session_id`、`reason`、`duration_ms`
    - `speech_recognized`：`text`、`source`，可选 `confidence`、`session_id`、`language`、`is_final`
    - `tts_started`：`source`，可选 `text`、`voice_id`、`session_id`
    - `tts_finished`：`source`，可选 `text`、`voice_id`、`session_id`、`status`
    - `voice_volume_changed`：`volume`、`source`
    - `voice_timbre_changed`：`voice_id`、`source`
    - `voice_speed_changed`：`speed`、`source`
    - `light_level_updated`：`light_lux`、`source`，可选 `level`、`is_low_light`
    - `temperature_humidity_updated`：`temperature_c`、`humidity_pct`、`source`，可选 `temperature_level`、`humidity_level`
    - `noise_level_updated`：`noise_db`、`source`，可选 `level`、`is_noisy`
  - `Action.type`
    - `start_voice_capture`：`source`，可选 `trigger`
    - `stop_voice_capture`：`source`，可选 `reason`
    - `set_tts_voice`：`voice_id`
    - `set_tts_volume`：`volume`
    - `set_tts_speed`：`speed`
    - `environment_alert`：`sensor`、`level`、`message`
- **交付给内核/联调方**：关键 Event payload 示例与推荐频率如下。

### 示例事件

```python
from src.agent.event import Event

voice_event = Event(
    type="speech_recognized",
    timestamp=1713700000,
    payload={
        "text": "请帮我查询今天的天气",
        "source": "mic_asr",
        "confidence": 0.93,
        "session_id": "voice-session-001",
        "language": "zh-CN",
        "is_final": True,
    },
)

env_event = Event(
    type="temperature_humidity_updated",
    timestamp=1713700005,
    payload={
        "temperature_c": 17.5,
        "humidity_pct": 28.0,
        "source": "sht30",
        "temperature_level": "low",
        "humidity_level": "dry",
    },
)
```

- 推荐上报频率：
  - `voice_wake_detected`：每次唤醒成功上报一次。
  - `voice_input_started` / `voice_input_stopped`：每轮录音各一次。
  - `speech_recognized`：若支持流式识别，可中间结果低频上报，最终结果必须 `is_final=True` 再上报一次。
  - `light_level_updated` / `temperature_humidity_updated` / `noise_level_updated`：建议 1~2 秒一次，并由 adapter 做滤波、防抖与阈值判断，避免过高频率冲击内核。


