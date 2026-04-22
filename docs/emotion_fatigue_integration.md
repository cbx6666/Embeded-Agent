# 情绪与疲劳：接入说明

本文档路径：**`docs/emotion_fatigue_integration.md`**，命名遵循 **`docs/team_integration_guide.md`** 中的约定：`docs/<主题>_integration.md`。

本文是**本团队情绪识别 + 疲劳/困倦检测**的模块说明，包含：**白话逻辑与 Event / Action / Adapter 分工**、**payload 与闭集**、**记忆与实施计划**。  
其他同学请另建 **`docs/<你们的主题>_integration.md`**，并遵循 **`docs/team_integration_guide.md`** 中的模板与流程。

> **现阶段分工**：以 **`Event` / `Action` 契约**、`factories`、**`vision_affect` / mock 适配器**与本文档为主，便于内核同学**单独**在 `reducer` / `policy` / `core` 里写业务逻辑；非内核同学提交时请**不要改** `reducer`、`policy`、`core`、`state`、`memory_service`（见 `team_integration_guide.md`「现阶段边界」）。下文「职责边界」描述的是**目标架构**，实现进度以仓库代码为准。

---

## 1. 设计目标与职责边界

**设计目标**

- 支持连续情绪检测（人脸裁剪 + RAF-DB / ResNet18 分类）。
- 支持连续疲劳几何特征（MediaPipe 关键点、闭眼比例、哈欠等时间窗统计）。
- 情绪侧只保留短期样本与时间窗摘要，避免内存与持久化膨胀。
- 保持 `event -> state/memory -> policy -> action` 边界清晰。

**职责边界**

- `Event`：描述事实（`user_emotion_updated`、`user_fatigue_updated`）。
- **检测实现（MediaPipe、EAR、PERCLOS、ResNet 等）**：只存在于 **`adapters/vision_affect/`**；**对内核不可见**——内核不得实现「如何从图像算疲劳」，只消费已打包好的事件 payload（见 **`docs/team_integration_guide.md`**「分层原则」）。
- `State/Memory`：情绪流样本与摘要；**疲劳写入 `UserState` 由内核同学后续在 `reducer` 中接入**；当前事件仍会进入 `recent_events`。
- `Policy`：读状态后决定是否产生 `Action`（如专注中休息提醒可同时看情绪与疲劳）。
- `Action`：播报、显示、计时器等，不承载模型细节。

---

## 2. 我们这条线负责什么（Event / Action / Adapter）

| 类别 | 我们关心的部分 |
|------|----------------|
| **Event** | 把摄像头侧结果变成 **`user_emotion_updated`**（情绪）与 **`user_fatigue_updated`**（疲劳层级）。 |
| **Action** | 一般为全局已有类型（`speak` / `display` / `none` 等）；情绪与疲劳只影响 **policy 何时、用什么文案** 输出，不强制新增 Action 类型。 |
| **Adapter** | **输入**：摄像头 + MediaPipe + ResNet 管线 → `handle_event`；**输出**：`console_output`（及未来 `tts_output`）消费 `speak` / `display`。 |

**边界**：模型、阈值、降频在 **输入 Adapter**；内核只做 **更新状态 + 选 Action**。

---

## 3. 白话：在做什么、逻辑是什么

1. 人在摄像头前 → 一路 **表情分类**，一路 **闭眼/哈欠等疲劳信号**。  
2. **两路分开发事件**：`user_emotion_updated` 与 `user_fatigue_updated`，避免「表情正常但很困」说不清。  
3. **Adapter** 在合适频率下组 `Event`，调用 **`AgentCore.handle_event`**。  
4. **内核**先改状态，再按规则产生 **Action**。  
5. **输出 Adapter** 把 `speak` / `display` 落到控制台或 TTS。

整体：**Adapter → Event → 内核 → Action → Adapter**。

---

## 4. Event / Action / Adapter 一览（本业务线）

### 4.1 Event（完整列表见 `src/agent/event/types.py`）

| `type` | 作用（白话） | 典型谁产生 |
|--------|-------------|------------|
| `user_emotion_updated` | 上报情绪档位与可选置信度、RAF 字段等；更新 `user.emotion`，参与情绪样本与窗口摘要。 | 未来视觉输入 Adapter；`mock_input`；`user_emotion_updated_from_rafdb()`。 |
| `user_fatigue_updated` | 上报疲劳层级与可选 PERCLOS、哈欠窗口等；**内核接入后**写入 `user.fatigue_*`；此前仅记录于 `recent_events`。 | `vision_affect` / `/mock fatigue`。 |

与策略相关但不必新造类型：如 **`timer_ticked`** 与情绪/疲劳一起在 policy 里决定是否提醒休息。

### 4.2 Action（`src/agent/action/types.py`）

| `type` | 在情绪/疲劳链路中 |
|--------|-------------------|
| `speak` / `display` | 休息提醒、状态反馈等文案。 |
| `start_timer` / `stop_timer` | 与专注模式配合，不直接表达情绪。 |
| `none` | 本轮不输出。 |

### 4.3 Adapter

| 文件 / 规划 | 作用 |
|-------------|------|
| `mock_input.py` | `/mock emotion …` → `user_emotion_updated`。 |
| `cli_input.py` | 文本输入；「现在状态如何」间接依赖已写入的情绪/疲劳状态。 |
| `console_output.py` | 执行 `speak` / `display`。 |
| `vision_affect/`（`VisionAffectInputAdapter` + `VisionAffectConfig`） | 摄像头 → MediaPipe Face Mesh / 可选 ResNet18 → `user_fatigue_updated` + 可选 `user_emotion_updated`；**阈值、帧率、防抖等均在配置与适配器内**，内核只收 Event。 |
| 可选 `tts_output.py` | 将 `speak` 变为语音。 |

---

## 5. 闭合集合与 payload（契约）

### 5.1 RAF-DB 原始输出（7 类）

| `label_id` | `raf_emotion` |
|-----------|----------------|
| 1 | `surprise` |
| 2 | `fear` |
| 3 | `disgust` |
| 4 | `happiness` |
| 5 | `sadness` |
| 6 | `anger` |
| 7 | `neutral` |

无法解析时工厂侧回退 `neutral`（与 `src/agent/event/factories.py` 一致）。

### 5.2 `UserEmotion`（4 类）

`neutral`，`tired`，`stressed`，`happy` — 定义见 `src/agent/state/types.py`。

### 5.3 RAF → `UserEmotion`（`RAF_TO_AGENT_EMOTION`）

| `raf_emotion` | → `emotion` |
|---------------|-------------|
| `happiness` | `happy` |
| `neutral` | `neutral` |
| `sadness` / `anger` / `disgust` / `fear` | `stressed` |
| `surprise` | `neutral` |

### 5.4 `user_emotion_updated` payload

| 字段 | 必需 | 说明 |
|------|------|------|
| `emotion` | 是 | 须为 `UserEmotion` |
| `confidence` | 否 | `[0, 1]` |
| `source` | 否 | 如 `mock`、`camera` |
| `person_id` | 否 | |
| `model` | 否 | 如 `raf-db` |
| `raf_emotion` / `raf_label_id` | 否 | RAF 侧原始信息 |

### 5.5 `FatigueLevel`（4 类）

`none`，`mild`，`moderate`，`high` — 阈值由适配器配置；类型别名当前在 **`src/adapters/vision_affect/pipeline.py`**，与 `user_fatigue_updated` 的 payload 一致（待内核接入状态后可迁入 `agent.state`）。

### 5.6 `user_fatigue_updated` payload

| 字段 | 必需 | 说明 |
|------|------|------|
| `fatigue_level` | 是 | `FatigueLevel` |
| `confidence` | 否 | `[0, 1]` |
| `perclos` | 否 | `[0, 1]` |
| `yawn_in_window` | 否 | bool |
| `window_sec` | 否 | 统计窗口秒数 |
| `source` | 否 | 建议 `mediapipe_pipeline` |
| `person_id` | 否 | |

与 `user_emotion_updated` 的关系：可只发疲劳事件；若产品需要也可再发 `user_emotion_updated`（`emotion: tired`）做融合。

---

## 6. 当前实现与记忆

- **情绪**：`MemoryService` 对 `user_emotion_updated` 维护 `emotion_samples`、按默认 **60s** 窗口 rollup `emotion_summaries`，并 trim 容量。  
- **疲劳**：`user_fatigue_updated` 会进入 **`recent_events`**；**`UserState` 疲劳字段与 `policy` 中休息提醒等逻辑待内核同学实现**；疲劳专用样本队列与摘要见 **§9**。

---

## 7. 对外可见与策略

- `/history`：`recent_events`，以及情绪的 `emotion_samples` / `emotion_summaries`。  
- 「现在状态如何」：最近情绪窗口摘要 + 当前疲劳层级（及 PERCLOS 若存在）。  
- 专注中 **`timer_ticked`** 的休息提醒：**当前实现**为 `attention == focused` 且 `emotion == tired`（见 `policy`）；接入疲劳状态后，内核可再合并 `fatigue_level`。

---

## 8. 推荐接入与实施阶段

1. 帧 → MediaPipe → 对齐裁剪 → ResNet18 → 情绪事件（可用 `user_emotion_updated_from_rafdb`）。  
2. 同管线计算 PERCLOS / 哈欠 → 降频后发 `user_fatigue_updated`。  
3. 可选融合：高疲劳时额外发 `user_emotion_updated`（`tired`）。

**阶段建议**：环境冻结 → 单帧 EAR/MAR + ResNet 离线验证 → 时间窗与 `FatigueLevel` → 适配器 + `handle_event` → 实机调参。

---

## 9. 后续可扩展点

- 疲劳侧 `fatigue_samples` / `fatigue_summaries`。  
- `/mock fatigue`、按 `person_id` 分人摘要、归档与质量阈值等。

---

## 10. 与代码对齐说明

**`EventType`** 已含 `user_fatigue_updated`；**`src/adapters/vision_affect/`** 与 **`/mock fatigue`** 可发出该事件。**内核侧**对疲劳的 `reducer` / `policy` / `UserState` 字段由负责人后续接入。运行见根目录 `README.md`（`--vision` / `requirements-vision.txt`）。

**模板对照**：本文 **§2～§4** 对应 `team_integration_guide.md` 第三节模板的前五节；**§5 起**为 payload 与闭集（各模块可按需增减）。他组请复制 **`docs/team_integration_guide.md`** 内模板另存为 **`docs/<主题>_integration.md`**。
