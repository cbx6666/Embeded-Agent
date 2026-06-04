# Embeded-Agent

Embeded-Agent 是一个面向嵌入式交互场景的 LLM-centered Agent Runtime Prototype。当前实现把 LLM 放在语义认知层，把 Python 放在 deterministic runtime boundary：LLM 负责理解、规划、审查、表达和长期记忆候选；runtime 负责 state、validator、guard、store、action、trace 和 replay。

## 核心链路

```text
Event
-> Reducer
-> RuntimeHistoryService
-> LongTermMemoryPipeline
-> PersonalContextBuilder
-> AgentContextBuilder
-> LLMAgentOrchestrator
-> IntentPlanValidator
-> DeterministicGuard
-> ActionRealizer
-> DeviceAdapter
-> ActionResult
-> RuntimeTrace
```

## 核心能力

- 统一建模 `Event / State / RuntimeHistory / LongTermMemory / PersonalContext / Intent / Action`。
- 四角色 LLM Agent：`SituationAnalyst`、`IntentPlanner`、`SafetyCritic`、`ResponseWriter`。
- 长期记忆管线：observe、extract、critic、consolidate、validate、store。
- Deterministic boundary：schema validation、registered intent、presence/cooldown guard、action realization、device adapter。
- 个性化上下文：显式 `UserProfile`、证据化 `LongTermMemory` 和短期 `RuntimeHistory` 收口到 `PersonalContext`。
- 可观测性：轻量 `RuntimeTrace` 支持 debug print、JSON dump 和测试断言。
- 长时间运行验证：`tests/scenarios/`、`tests/replay/` 和 `scripts/runtime_experiments/`。

## 项目结构

```text
src/
  main.py
  adapters/
  agent/
  services/
  storage/

tests/
  scenarios/
  replay/

scripts/
  runtime_experiments/
  debug/

docs/
  design/
  integration/
  requirements/
  shared/
```

## 文档入口

- `docs/design/src_architecture_design.md`：当前 `src/` 架构设计文档。
- `docs/design/storage_layout.md`：本地 `data/` 目录的数据域布局说明。
- `docs/requirements/agent_requirements.md`：Agent 需求与目标说明。
- `docs/integration/`：行为、显示、环境、语音、情绪等适配说明。
- `src/agent/README.md`：Agent 内核目录说明。

## LLM 配置

生产 `LLMService` 使用 DeepSeek **Chat Completions**（对话型 API，不是 reasoner）。模型由 `.env` 决定：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

- **`deepseek-chat`**（默认）：回复快，适合语音交互。
- **`deepseek-reasoner`**：推理链更长，单轮常慢数倍，不建议用于唤醒对话。

默认 **`--llm-mode fast`**：对语音识别/文本输入只调 **1 次** LLM（约 2～5s），把四角色的关键输入（当前状态、最近对话、用户偏好、结构化 JSON）压缩进一次 `fast_dialogue` 请求；需要完整规划与安全审查时用 `--llm-mode full`。

未配置 API 时，生产链路会抛出配置错误。测试和实验通过显式 fake/double 提供 deterministic LLM 行为，不由 `LLMService` 内置 mock。

## 固定唤醒词（Sherpa-ONNX，无需 Picovoice 注册）

首次在一台机器上准备模型与「小助」关键词：

```bash
source /opt/ai-envs/shared/bin/activate
pip install sherpa-onnx pypinyin
python scripts/setup_sherpa_kws.py --keyword 小助
```

默认 `--wake-backend sherpa-onnx`，说「**小助**」→ **本地预加载短句**（最快）→ 录音 → ASR → LLM 详细回复。

首次生成本地应答音频（只需一次，需百度 TTS 或 espeak）：

```bash
python scripts/generate_wake_ack_audio.py
# 离线备选：python scripts/generate_wake_ack_audio.py --backend espeak
```

```bash
export DISPLAY=:1
python -m src.main --voice-debug --wake-word 小助
```

调灵敏度：

```bash
python -m src.main --wake-keywords-threshold 0.2   # 越小越容易唤醒
python -m src.main --wake-backend energy           # 回退到音量触发（非固定词）
```

## 语音链路：哪些在本地、哪些在云端

| 环节 | 默认 | 说明 |
|------|------|------|
| 唤醒「小助」 | **本地** Sherpa KWS | 无网络 |
| 即时应答「我在，请说。」 | **本地** 预加载 WAV | 与录音**并行**播放，不阻塞开麦 |
| 用户说话 → 文本 | **云端** 百度 ASR | **默认唤醒即录**（`sync`），可连着唤醒词说话；VAD 说完停录 |
| 理解 + 生成回复 | **云端** DeepSeek（`deepseek-chat` + `--llm-mode fast`） | 默认 **LLM SSE 流式** + **逐句百度 TTS**（首句更快出声） |
| Agent 语音播报 | **云端** 百度 TTS（默认） | `--tts-backend sherpa-onnx` 可改板端离线 |

仍想用 Sherpa 离线合成 Agent 回复时：

```bash
python scripts/setup_sherpa_tts.py
python -m src.main --tts-backend sherpa-onnx
```

百度 TTS 为默认，需在 `.env` 配置 `BAIDU_TTS_*`（与 ASR 可共用密钥）。

语音采集与流式（默认已开）：

```bash
# 默认：唤醒词命中立即开录，应答并行；VAD 说完停录
python -m src.main --capture-mode vad --silence-duration 0.8

# 旧行为：等应答播完再录
python -m src.main --wake-record-timing after_ack --post-ack-delay 0.5

# 固定时长录音（旧行为）
python -m src.main --capture-mode fixed --post-wake-duration 6

# 关闭 LLM 流式 + 逐句 TTS
python -m src.main --no-cloud-streaming
```

说明：百度短 ASR 仍是「录完后整段上传」，VAD 缩短等待；TTS 按句合成，在 LLM 还在生成后续内容时可先播第一句。

**录放分离（默认）**：麦克风走摄像头/USB（`--voice-alsa-device auto`），TTS/应答走独立扬声器（默认 `--tts-alsa-device plughw:1,0`）。查看本机设备：

```bash
python -m src.main --list-audio-devices
```

若自动识别不对，可在 `.env` 或命令行显式指定：

```bash
export EMBED_VOICE_CAPTURE_ALSA_DEVICE=plughw:0,0   # 摄像头麦
export EMBED_TTS_ALSA_DEVICE=plughw:1,0             # 板载扬声器，不参与录音
python -m src.main
```

**听录音排查 ASR**（录到的到底是什么）：

```bash
# 录完自动播放一遍 + 保存 latest.wav
python -m src.main --playback-recording

# 同时保留每次唤醒的独立文件
python -m src.main --playback-recording --keep-voice-recordings

# 运行中在 agent> 再听一次
/voice_replay

# 或手动（路径见终端 [BoardVoiceAdapter] 录音已保存）
aplay -D plughw:1,0 data/voice_recordings/latest.wav
```

## 运行

默认**全栈**（桌宠全屏 + 视觉 + 语音 + LLM Agent，需 `.env` 中 `DEEPSEEK_API_KEY`）：

```bash
python -m src.main
```

板子/VNC 示例：

```bash
export DISPLAY=:1
python -m src.main
# 等价于原先：python -m src.main --screen --screen-fullscreen --vision --voice --camera 1 --emotion-backend deepface
```

仅 CLI Agent + LLM（原先裸 `python -m src.main` 的行为）：

```bash
python -m src.main --llm
```

全栈下关闭部分模块：

```bash
python -m src.main --no-voice          # 不要语音
python -m src.main --no-screen-fullscreen   # 桌宠窗口化
```

`--llm` 模式下按需加适配器：

```bash
python -m src.main --llm --vision --camera 0
python -m src.main --llm --screen
python -m src.main --llm --voice --voice-loop
python -m src.main --llm --voice --no-wake   # 手动 /voice_once
```

可选姿势检测（YOLO，当前为占位实现，Atlas 部署见 `docs/integration/external/`）：

```bash
python -m src.main --pose
python -m src.main --llm --pose --pose-device npu --pose-model yolov8n-pose.pt
```

## 测试

```bash
python -m pytest -q
```

## 运行实验

```bash
python scripts/runtime_experiments/study_session_experiment.py
python scripts/runtime_experiments/long_term_memory_experiment.py
python scripts/runtime_experiments/multi_user_isolation_experiment.py
python scripts/runtime_experiments/hallucination_resistance_experiment.py
python scripts/runtime_experiments/retrieval_quality_experiment.py
```

实验输出默认写入：

- runtime experiments：`data/experiments/runtime/`
- retrieval experiment：`data/experiments/retrieval/`

## Debug CLI

```bash
python scripts/debug/inspect_memory.py --memory data/memory/long_term_memory.json
python scripts/debug/inspect_personal_context.py --user default --text "gentle reminder"
python scripts/debug/inspect_trace.py data/experiments/runtime/study_session_experiment/trace_logs.json
python scripts/debug/replay_events.py data/experiments/runtime/study_session_experiment/events.json
```
