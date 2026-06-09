# Agent 功能深入测试流程

本文档面向**验收与联调**：覆盖 Agent 全部分流、用户画像、多维记忆、个性化多元回答、自主关怀、媒体、Guard/调度，以及可选的真实 LLM / 硬件链路。

> 架构与模块说明见 `src/agent/README.md`；数据落盘见 `docs/design/storage_layout.md`。

---

## 0. 测试前准备

### 0.1 环境

```bash
cd /root/Embeded-Agent
python -m pytest -q                    # 基线：应全绿（当前 202+ passed）
python -m src.main --help            # 确认 CLI 参数帮助正常
```

### 0.2 模式选择

| 模式 | 命令 | 适用场景 |
|------|------|----------|
| **CLI Agent** | `python -m src.main --llm` | 无摄像头/麦克风，用 `/mock` + 事件注入测 Agent 逻辑 |
| **全栈** | `python -m src.main` | 桌宠 + 视觉 + 语音 + 环境传感器 |
| **仅语音 LLM** | `python scripts/voice_llm_only.py` | 专注语音链路 |

真实 LLM 测试需在 `.env` 配置 `DEEPSEEK_API_KEY`（及可选 `BAIDU_TTS_*`）。无 API 时用 `tests/fakes/fake_llm_service.py` 做确定性回归。

### 0.3 观测工具

| 工具 | 用途 |
|------|------|
| `agent> /state` | 完整 `AgentState` 快照 |
| `agent> /history` | 最近事件、消息、专注记录 |
| `agent> /profile` | 当前用户画像（权威 Profile） |
| `python scripts/debug/inspect_memory.py` | 查看 `MemoryService` 抽取结果 |
| `python scripts/debug/inspect_trace.py` | 查看 trace 日志（若已开启） |
| 终端 `[记忆检索]` 行 | 每轮 LLM 前检索/使用了哪些记忆 |

### 0.4 重要约定

- **自然语言对话**不走 CLI 文本框，应通过 `speech_recognized`（`/voice_once` 或代码注入）进入。
- CLI 只识别：专注开始/结束、`/mock`、`/profile` 系列、系统命令。
- 自主检查周期：`behavior_distraction_check` 20s、`wellness_care_check` 30s、`environment_care_check` 60s、`sensor_status_report` 300s。

---

## 1. 自动化回归矩阵（必须先过）

按功能域分批运行，失败时先修再继续手动测试。

```bash
# 核心分流 / 动作闭集 / 专注规则
pytest tests/test_agent_refactor.py -q

# 调度器 + 周期 trigger + 传感器确定性播报
pytest tests/test_periodic_and_scheduler.py -q

# 自主关怀：wellness / environment / behavior + 个性化 prompt
pytest tests/test_care_checks.py -q

# 记忆使用策略（轮换、候选、speech 边界）
pytest tests/test_memory_usage_hints.py -q

# 记忆抽取与检索
pytest tests/test_memory_llm.py -q

# 媒体选曲 / pending suggestion / 语音媒体控制
pytest tests/test_media_feature.py -q

# TTS 文案校验（完整句、半句话拦截）
pytest tests/test_reply_validator.py -q

# 语音管线 / 唤醒应答
pytest tests/test_voice_pipeline.py tests/test_voice_adapters.py tests/test_local_wake_ack.py -q

# 全量
pytest -q
```

| 测试文件 | 覆盖的 Agent 能力 |
|----------|-------------------|
| `test_agent_refactor` | EventRouter 分流、5 种 Action、RuleHandler 专注、`focus_complete_care` 轮换兴趣 |
| `test_periodic_and_scheduler` | 多任务调度、旧 trigger 不得复活、`sensor_status_report` 数值播报 |
| `test_care_checks` | wellness/environment/behavior Handler、Guard、invalid reply 不播 |
| `test_memory_usage_hints` | `memory_usage_hints` 构造、profile 爱好入候选、speech 不因视觉疲劳抢话 |
| `test_media_feature` | `MediaController`、曲库、pending 批准/拒绝 |
| `test_reply_validator` | 播报文案质量门槛 |

---

## 2. CLI 交互测试（`--llm`）

```bash
python -m src.main --llm
```

### 2.1 基础系统命令

| 步骤 | 输入 | 预期 |
|------|------|------|
| 帮助 | `/help` | 列出 mock / profile / 专注命令 |
| 状态 | `/state` | 含 `user`、`focus`、`interaction`、`environment` |
| 历史 | `/history` | 随操作累积事件与消息 |

### 2.2 专注计时（RuleHandler，不走 LLM）

| 步骤 | 输入 | 预期 |
|------|------|------|
| 开始 | `开始专注 25 分钟` | `start_timer` + `speak` + `display`；`focus.active=true` |
| 结束 | `结束专注` | `stop_timer`；`focus.active=false` |
| 自然完成 | 等待计时结束或 mock `timer_finished` | `complete_focus`；若配 LLM 则 `focus_complete_care` 个性化文案 |

**验收点**：`test_agent_refactor` 中 `focus_start/stop` 的 `used_llm=false`；`timer_finished` 会调 `focus_complete_care` role。

### 2.3 Mock 状态注入（模拟感知）

用 `/mock` 驱动 `state_only` 事件，为后续自主检查积累窗口数据：

```text
/mock presence present
/mock fatigue high
/mock emotion tired
/mock posture leaning
/mock behavior phone_use
/mock attention distracted
/mock activity working
```

**验收点**：`/state` 中对应字段更新；`runtime_history.signal_trends` 有近期采样。

> 环境数值（温湿度/光照）CLI 无 mock，见 §3.2 用事件注入。

---

## 3. 程序化事件注入（深入测 Agent）

CLI 无法直接输入「用户说的话」或环境传感器读数时，用短脚本注入（推荐单独终端或 `python -c`）。

```python
from src.agent.core import build_default_core
from src.agent.event import Event
from src.adapters.console_output import ConsoleOutput
from tests.fakes.fake_llm_service import FakeLLMService

fake = FakeLLMService()
fake.set_response("speech_recognized", {"intent": "answer_user", "reply": "好的，我在。"})
core = build_default_core(output=ConsoleOutput(), llm_service=fake)
core.start_autonomous_scheduler()

# 用户语音
core.handle_event(Event(type="speech_recognized", timestamp=1000, payload={"text": "今天有点累"}))

# 环境读数（供 environment_care / sensor_status）
core.handle_event(Event(
    type="temperature_humidity_updated", timestamp=1001,
    payload={"temperature_c": 31.0, "humidity_pct": 22, "temperature_level": "high", "humidity_level": "dry"},
))
core.handle_event(Event(
    type="light_level_updated", timestamp=1002,
    payload={"light_lux": 80, "light_level": "low"},
))

# 手动触发自主检查（也可等调度器）
core.handle_event(Event(
    type="system_triggered", timestamp=1100,
    payload={"trigger": "wellness_care_check", "source": "agent_autonomy"},
))
print(core.last_decision_result)
core.shutdown()
```

### 3.1 各 Handler 注入清单

| 能力 | trigger / event | 前置状态建议 | 观察字段 |
|------|-----------------|--------------|----------|
| 语音理解 | `speech_recognized` | 任意 | `last_decision_result.source=speech_llm`，`actions` 含 `speak` |
| 分心提醒 | `behavior_distraction_check` | `phone_use` + YOLO 窗口（见 pytest mock summary） | `remind_distraction`，冷却 60s |
| 疲劳/情绪/姿态关怀 | `wellness_care_check` | `fatigue high` 连续窗口或 `emotion`/`posture` 异常 | `suggest_rest` / `offer_emotion_care` 等 |
| 环境关怀 | `environment_care_check` | 低光照/干燥/高温等 + `presence present` | `adjust_environment_feedback` 或 `no_op` |
| 传感器播报 | `sensor_status_report` | 有温湿度等读数 | 确定性文案含具体数值，**不调 LLM** |
| 专注结束关怀 | `timer_finished` | 先 `focus_start_requested` | `focus_complete_care` role 被调用 |

---

## 4. 用户画像（UserProfile）测试

画像为**权威显式数据**，与 LLM 抽取记忆分离。

### 4.1 CLI 画像操作

```text
/users
/profile
/switch_user xiaoli 小李
/set_info age 12
/set_info hobbies 画画,足球,听相声
/set_pref reminder_style 温和
/set_pref favorite_music_styles 轻音乐,古风
/set_pref favorite_content_types 音乐,相声,脱口秀
```

| 验收项 | 操作 | 预期 |
|--------|------|------|
| 多用户隔离 | 切换用户后 `/profile` | `current_user_id` 变化，资料独立 |
| 爱好入 prompt | 设置 hobbies 后触发 wellness LLM | `[记忆检索]` 或 prompt 中出现 profile 字段 |
| 偏好影响语气 | `reminder_style=严格` vs `温和` | 自主关怀 prompt 中 `memory_usage_hints` 的 `style_hints` / `avoid_patterns` 不同 |

持久化文件：`data/user/user_profiles.json`（见 `storage_layout.md`）。

---

## 5. 多维记忆（MemoryService）测试

### 5.1 写入路径（异步抽取）

1. 启动带真实 LLM 的 Agent：`python -m src.main --llm`
2. 注入或语音说含偏好的句子，例如：
   - 「我平时写代码累了喜欢听相声」
   - 「别太频繁打断我」
   - 「我更喜欢严格一点的提醒」
3. 等待异步 `memory_extract` 完成（通常数秒内）
4. 检查：

```bash
python scripts/debug/inspect_memory.py --memory data/memory/user_memory.json
python scripts/debug/inspect_memory.py --user default --json
```

| 验收项 | 预期 |
|--------|------|
| 新记忆入库 | `type` 为 hobby / preference / work_style 等 |
| 不阻塞主链路 | 语音回复先于记忆写入完成 |
| 矛盾处理 | 说「其实我不喜欢相声了」后，旧条目标 contradicted（见 `test_memory_llm`） |

### 5.2 读取路径（LLM 前检索）

每次 LLM 调用前，`AgentCore._user_context()` 合并：

`profile + preferences + memories.by_type + recent_messages + memory_usage_hints`

**验收**：终端出现类似日志：

```text
[记忆检索] context=wellness_care recommended_content=... personalization_candidates_count=6 ...
```

### 5.3 `memory_usage_hints` 行为（必测）

| 场景 | 验证方式 | 预期 |
|------|----------|------|
| 语音不因摄像头疲劳抢话 | 用户说「你好」，state 里 `fatigue=high` | `speech` hints 的 `focus=general`，reply 不夹带休息提醒 |
| 用户亲口说累 | `speech_recognized` text 含「累」 | `focus=fatigue`，可结合 `recommended_content` |
| wellness 多元候选 | profile 有 3+ 爱好 + memory 有多类 | `personalization_candidates` 5~8 条、多 category |
| 兴趣轮换 | 连续两次 `wellness_care_check` 或两次 `timer_finished` | `care_rotation_index` 递增；`recommended_content.label` 不同 |
| 避免重复唠叨 | 同一关怀方向连续触发 | prompt 含 `recent_reminder_texts`，LLM 被要求换说法 |

自动化参考：`tests/test_memory_usage_hints.py`、`test_focus_complete_personalizes_and_rotates_interest`。

---

## 6. 多元回答与 TTS 文案质量

### 6.1 Prompt 层

所有会播报的 LLM 入口均注入 `tts_reply_quality.md`（经 `prompt_builder._with_tts_quality`）：

- `speech_recognized`
- `behavior_distraction_check`
- `wellness_care_check`
- `environment_care_check`
- `focus_complete_care`

**验收**：对任意 `build_*_prompt` 输出 grep `完整句子` 与 `语音合成`。

### 6.2 Runtime 校验层

`reply_validator.py` 在 Handler 落地前拦截：

- 空字符串 → `no_op`，`invalid_llm_reply: true`
- 半句话（如「有点累，要不要。」「注意坐姿，放松。」）→ 不送 TTS
- 正常短句（「歇会儿」「光线有点暗」）→ 允许播报

```bash
pytest tests/test_reply_validator.py tests/test_care_checks.py::FallbackSuppressionTest -q
```

### 6.3 真实 LLM 多元性抽检（需 API）

同一状态 **连续触发 3~5 次** `wellness_care_check`（可调低 `WellnessCareCheckPolicy` 冷却仅用于实验环境）：

| 验收项 | 说明 |
|--------|------|
| 措辞不重复 | 不应每轮都是「要不要听抒情歌」 |
| 自然口语 | 无「系统检测到」「根据你的记忆」 |
| 第二人称 | 无「用户喜欢…」照抄记忆原文 |
| 兴趣轮换 | 轮次间点缀不同 `recommended_content` / 候选 |
| 姿态不限模板 | `posture` 触发时句式多样（非每次「坐直」） |

记录每轮 `speak` payload 的 `text` 与 `[记忆检索]` 日志，人工比对。

---

## 7. 分功能深入测试用例

### 7.1 语音 LLM（`speech_recognized`）

| # | 注入文本 / 语音 | 预期 intent | 备注 |
|---|----------------|-------------|------|
| S1 | 「你好」 | `answer_user` | 纯对话 |
| S2 | 「开始专注 25 分钟」 | `start_focus` | 含 `duration_sec` |
| S3 | 「结束专注」 | `stop_focus` | |
| S4 | 「音量调到 60」 | `set_tts_volume` | |
| S5 | 「放一首外语歌」 | `media_control` + `play_media` | 需曲库；track_id 合法 |
| S6 | pending 建议后「好的」 | `play_media` | 先 wellness `suggest_media` 产生 pending |
| S7 | pending 后「不要」 | `answer_user` | 拒绝播放 |
| S8 | 视觉 fatigue=high 时说「你好」 | 回复**不含**疲劳关怀 | 边界：speech vs wellness |

### 7.2 分心检查（`behavior_distraction_check`）

| # | 前置 | 预期 |
|---|------|------|
| B1 | `phone_use` 窗口达标 + present | `remind_distraction` + speak |
| B2 | `dialogue_state=listening` | 延后，不消耗提醒（defer） |
| B3 | LLM 返回空/病句 | `no_op`，`invalid_llm_reply` 或 `fallback_suppressed` |
| B4 | 60s 内重复 trigger | Guard 冷却拦截 |

### 7.3 疲劳/情绪/姿态（`wellness_care_check`）

| # | 前置 | 预期 intent |
|---|------|-------------|
| W1 | fatigue high 持续 ≥20s | `suggest_rest` |
| W2 | emotion 负面持续 | `offer_emotion_care` |
| W3 | posture leaning 持续 | 姿态相关关怀 |
| W4 | fatigue + 低光照同时 | **wellness 优先**，environment 不抢答 |
| W5 | `focus.active=false` | 回复不出现番茄钟/剩余分钟 |
| W6 | intent=suggest_media | 只**询问**是否听，不直接 `play_media` |

### 7.4 环境关怀（`environment_care_check`）

| # | 前置 | 预期 |
|---|------|------|
| E1 | 低光照 + present | `adjust_environment_feedback` 或 LLM no_op |
| E2 | 用户 away | 强制 no_op |
| E3 | LLM 返回 `suggest_rest` | 夹断为 no_op（不能产出休息提醒） |
| E4 | `speaking` 中 | defer 到下一轮 |

### 7.5 传感器播报（`sensor_status_report`）

| # | 前置 | 预期 |
|---|------|------|
| R1 | 有温湿度读数 + present | speak 含具体数字，如 `31.2`、`68` |
| R2 | away / listening | no_op 或 defer，周期回退 |
| R3 | FakeLLM 不应被调用 | `sensor_status_report` ∉ `fake.calls` |

### 7.6 专注结束关怀（`focus_complete_care`）

| # | 步骤 | 预期 |
|---|------|------|
| F1 | 完成一轮专注 | 默认 Rule 文案被 LLM 覆盖（成功时） |
| F2 | 连续完成两轮 | `care_rotation_index` 增加；两轮口播兴趣不同 |
| F3 | LLM 失败/病句 | 保留 Rule 默认完成文案，停表仍成功 |

### 7.7 媒体（`MediaController`）

```bash
pytest tests/test_media_feature.py -q
```

| # | 场景 | 预期 |
|---|------|------|
| M1 | 用户明确点播 | 先 TTS 再 `play_media`（`defer_after_speak`） |
| M2 | 曲库选外语 folder | 不选中文流行 |
| M3 | next_media | 不重复 `current_track_id` |
| M4 | 唤醒词打断播放 | VoiceRuntime 与 MediaController 协作 |

---

## 8. 调度器与 Guard

| # | 测试 | 命令/方式 | 预期 |
|---|------|-----------|------|
| G1 | 四任务周期 | 启动后 `/state` 或日志 | 见 §0.4 周期表 |
| G2 | 优先级 | behavior 20s 与 wellness 30s 同到期 | behavior 先 emit |
| G3 | 旧 trigger 不复活 | 注入 `periodic_state_check` | `state_only`，不走 LLM |
| G4 | TTS 中环境检查 | `dialogue_state=speaking` | environment defer |
| G5 | 冷却防刷屏 | 短间隔重复关怀 | 第二次 Guard blocked |

```bash
pytest tests/test_periodic_and_scheduler.py tests/test_autonomous_defer_fix.py -q
```

---

## 9. 语音与全栈（可选硬件）

### 9.1 唤醒 + ASR + LLM + TTS

```bash
python -m src.main --llm --voice --voice-debug --wake-word 小助
# 或全栈
export DISPLAY=:1
python -m src.main --voice-debug
```

| 环节 | 验收 |
|------|------|
| 唤醒 | 说「小助」→ 本地应答 WAV |
| 录音 | VAD 停录；`/voice_replay` 可回放 |
| LLM | 回复为完整句，无半句话播报 |
| 队列 | 自主提醒与用户回复经 `TTSPlaybackManager` 串行 |

### 9.2 感知联调（无桌宠/语音）

```bash
python scripts/test_full_stack_integration.py --duration 45
python scripts/test_full_stack_integration.py --duration 60 --emotion-backend none
```

检查输出「Event 功能覆盖」段：疲劳/情绪/行为/姿势 Event 是否收到。

### 9.3 ESP32 环境传感器

```bash
python -m src.main --llm --esp32-sensor --esp32-port /dev/ttyUSB0
pytest tests/test_esp32_sensor.py -q
```

---

## 10. 推荐测试顺序（一站式）

```text
Day 1  自动化全绿 → CLI 专注/mock/profile → 事件注入脚本跑通 S1/W1/E1/R1
Day 2  记忆写入/检索 → hints 轮换 → 专注结束关怀 F1-F3
Day 3  媒体 M1-M4 → Guard/调度 G1-G5 → invalid reply 回归
Day 4  （可选）真实 LLM 多元抽检 + 语音全链路 + 全栈 integration
```

---

## 11. 通过标准 checklist

- [ ] `pytest` 全量通过
- [ ] `python -m src.main --help` 正常
- [ ] EventRouter 七类分流与 README 一致，旧入口不复活
- [ ] UserProfile 多用户、爱好、偏好可编辑且进入 prompt
- [ ] Memory 异步抽取可查、`memory_usage_hints` 日志可见
- [ ] wellness 关怀有多候选、有轮换、不模板化复读
- [ ] speech 不因视觉状态抢话；用户亲口状态可触发对应 focus
- [ ] 空/病句 LLM reply 不播报（`invalid_llm_reply`）
- [ ] sensor 播报含准确数值且不调 LLM
- [ ] 媒体点播与 pending 批准/拒绝路径正确
- [ ] （可选）真实语音唤醒→ASR→LLM→TTS 端到端

---

## 12. 相关文档

| 文档 | 内容 |
|------|------|
| `src/agent/README.md` | 内核架构、Handler、策略 |
| `docs/integration/*.md` | 语音/显示/行为/情绪/环境适配 |
| `docs/design/storage_layout.md` | `data/` 各域说明 |
| `docs/commands.txt` | CLI / mock / profile 命令速查 |
| `README.md` | 运行方式与环境变量 |
