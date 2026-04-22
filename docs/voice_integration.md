# 语音功能：我们要负责的 Event、Action 和 Adapter

本文只讲**和「语音输入采集与语音播报输出」**相关的内容。  
更细的 ASR/TTS 引擎选型、串口音频模块协议、缓存策略可继续补在本文后续小节或 PR 说明中。

---

## 一、我们这条线要负责什么

| 类别 | 我们关心的部分 |
|------|----------------|
| **Event（事件）** | `voice_input_captured` |
| **Action（动作）** | `speak`、`play_voice` |
| **Adapter（适配器）** | `src/adapters/voice_adapter.py` |

**边界**：麦克风采样、VAD、ASR、TTS、音量控制、是否打断当前播报等，放在 `src/adapters/`；内核只看到标准 `Event` / `Action`，不依赖具体语音芯片、ASR 服务、TTS 厂商。

---

## 二、白话：我们在做什么、逻辑是什么

1. 用户说话后，语音侧先在适配器里做采集和识别，把结果整理成标准事件。
2. 识别完成后，适配器只向内核上报 `voice_input_captured`，不直接去改状态。
3. 内核后续可以根据识别文本决定是否当作对话输入、命令输入，或者忽略。
4. 当内核需要播报时，发 `speak` 或 `play_voice`，语音适配器再去调具体 TTS 或语音模块。
5. 执行失败、重试、是否可打断等都由语音适配器和底层设备处理，对内核透明。

整体逻辑一句话：**Adapter → Event → 内核 → Action → Adapter**

---

## 三、Event 有哪些（与本业务线直接相关）

| Event（`type`） | 作用（白话） | 典型谁产生 |
|-----------------|-------------|------------|
| `voice_input_captured` | 上报一段语音识别后的文本，以及置信度、语言、是否最终结果等信息 | `VoiceAdapter` |

（仓库里已有类型以 `src/agent/event/types.py` 为准。）

---

## 四、Action 有哪些（本业务线场景下会怎么用）

| Action（`type`） | 作用（白话） | 在本业务线中 |
|------------------|-------------|--------------|
| `speak` | 通用播报动作 | 可兼容现有简单播报链路 |
| `play_voice` | 明确要求语音播报，可带 voice / emotion / interrupt / volume 等参数 | 作为语音输出主动作，供内核以后细化控制播报风格 |

（仓库里已有类型以 `src/agent/action/types.py` 为准。）

---

## 五、Adapter 有哪些（作用是什么）

| Adapter | 作用（白话） | 与本业务线的关系 |
|---------|-------------|------------------|
| `VoiceAdapter` | 统一封装语音输入输出：把语音识别结果变成 `voice_input_captured`，把 `speak` / `play_voice` 落到 TTS 或语音模块 | 本业务线主适配器 |

---

## 六、与项目组 / 内核的衔接

- 本 PR 是否**新增或修改** `Event` / `Action` 类型：**是**；新增 `voice_input_captured`、`play_voice`。  
  - `voice_input_captured.payload` 要点：`text` 必填；`source` 必填；`is_final` 必填；`confidence` / `language` / `audio_id` 选填。  
  - `play_voice.payload` 要点：`text` 必填；`interrupt` 必填；`voice` / `emotion` / `volume` 选填。
- **交付给内核/联调方**：建议 ASR 最终结果（`is_final=true`）再正式上报给内核；中间结果可按产品需要选择不上报，或低频上报给调试链路。推荐最终结果事件在一句话结束时上报一次。

```json
{
  "type": "voice_input_captured",
  "timestamp": 1713772800,
  "payload": {
    "text": "开始专注二十五分钟",
    "source": "microphone",
    "is_final": true,
    "confidence": 0.93,
    "language": "zh-CN",
    "audio_id": "utt-0001"
  }
}
```
