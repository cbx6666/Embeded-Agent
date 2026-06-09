# Embeded-Agent

Embeded-Agent 是一个面向嵌入式交互场景的 Agent Runtime。LLM 在固定入口参与语义决策（`speech_recognized`、自主检查 `behavior_distraction_check` / `wellness_care_check` / `environment_care_check` 等）；Python runtime 负责事件归约、状态维护、分流、Guard、动作落地与设备执行。

## 核心链路

```text
Event
-> reduce_state
-> RuntimeHistoryService
-> EventRouter
-> speech_llm | behavior_distraction | wellness_care | environment_care | sensor_status | rule | state_only
-> ActionRealizer
-> DeviceAdapter
-> ActionResult
```

## 核心能力

- 统一建模 `Event`（20 种）/ `State` / `RuntimeHistory` / `Intent` / `Action`（5 种）。
- 多路 LLM 入口：用户语音单次理解、分心/疲劳/环境等自主检查（20s/30s/60s 周期）。
- 异步偏好记忆：`MemoryService` 从语音提取偏好，供 LLM prompt 检索。
- 确定性边界：Guard（防刷屏/不在场/TTS 中）、动作闭集、DeviceAdapter 不支持则明确失败。
- 显式 `UserProfile` + 短期 `RuntimeHistory` + 偏好记忆合并进 prompt。
- 硬件联调：adapters 只产生注册 Event、只执行注册 Action。

## 项目结构

```text
src/
  main.py
  adapters/
  agent/
  services/
  storage/

tests/
  test_agent_refactor.py
  fakes/

scripts/
  debug/

docs/
  design/
  integration/
  requirements/
  testing/
  shared/
```

## 文档入口

- `src/agent/README.md`：当前 Agent 内核架构（权威）。
- `docs/testing/agent_functional_test_guide.md`：**Agent 功能深入测试流程**（画像、记忆、多元回答、全 Handler）。
- `docs/archive/src_architecture_design.md`：历史架构文档（已归档，非当前运行架构）。
- `docs/design/storage_layout.md`：本地 `data/` 目录的数据域布局说明。
- `docs/requirements/agent_requirements.md`：Agent 需求与目标说明。
- `docs/integration/`：行为、显示、环境、语音、情绪等适配说明。
- `docs/commands.txt`：CLI / mock / profile 命令速查。

## LLM 配置

生产 `LLMService` 使用 DeepSeek **Chat Completions**（对话型 API，不是 reasoner）。模型由 `.env` 决定：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

- **`deepseek-chat`**（默认）：回复快，适合语音交互。
- **`deepseek-reasoner`**：推理链更长，单轮常慢数倍，不建议用于唤醒对话。

语音与各自主检查各走 **1 次** LLM（独立 prompt），无四角色串行。`sensor_status_report` 为确定性数值播报，不调 LLM。未配置 API 时生产链路抛出配置错误；测试通过 `tests/fakes/fake_llm_service.py` 注入确定性行为。

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
| 理解 + 生成回复 | **云端** DeepSeek（`deepseek-chat`，单次 `speech_recognized` LLM） | 默认 **LLM SSE 流式** + **逐句百度 TTS**（首句更快出声） |
| Agent 语音播报 | **云端** 百度 TTS（默认） | `--tts-backend sherpa-onnx` 可改板端离线 |

仍想用 Sherpa 离线合成 Agent 回复时：

```bash
python scripts/setup_sherpa_tts.py
python -m src.main --tts-backend sherpa-onnx
```

百度 TTS 为默认，需在 `.env` 配置 `BAIDU_TTS_*`（与 ASR 可共用密钥）。

语音采集（`AudioInputManager` 单路 capture + VAD tap）：

```bash
# VAD 说完停录（默认静音 0.8s）
python -m src.main --silence-duration 0.8 --max-capture-duration 15
```

说明：唤醒应答、Agent 回复、自主提醒均经 `TTSPlaybackManager` 统一排队播放；百度短 ASR 为录完后整段上传。

**音频路由（默认按设备名称，不依赖 card 编号）**：盒子(UAC)听唤醒词并播报，摄像头(C920)录用户话。USB 插拔或重启后 card 编号可能变化，系统按名称自动匹配。查看本机解析结果：

```bash
python -m src.main --list-audio-devices
```

若自动识别不对，可用别名或显式指定（推荐别名，避免 card 编号变化）：

```bash
export EMBED_VOICE_CAPTURE_ALSA_DEVICE=camera   # 用户说话：摄像头麦
export EMBED_WAKE_CAPTURE_ALSA_DEVICE=box       # 唤醒词：盒子麦
export EMBED_TTS_ALSA_DEVICE=speaker            # 播报：盒子扬声器
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

深入功能验收（用户画像、记忆检索、个性化多元回答、自主关怀、媒体等）见
[`docs/testing/agent_functional_test_guide.md`](docs/testing/agent_functional_test_guide.md)。

## Debug CLI

```bash
python scripts/debug/inspect_memory.py --memory data/memory/user_memory.json
```
