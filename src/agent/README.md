# Agent

`agent/` 是嵌入式硬件 Agent 的主领域包。主链路：

```text
Event
  -> reduce_state（更新 AgentState）
  -> RuntimeHistoryService（更新短期历史）
  -> EventRouter（分流）
  -> speech_llm | behavior_distraction | wellness_care | environment_care | sensor_status | rule | state_only
  -> ActionRealizer（Intent -> Action）
  -> DeviceAdapter（执行）
  -> 异步记忆抽取 + JsonStore 持久化
```

## 目录结构

```text
src/agent/
  core/           AgentCore、build_default_core、Intent/DecisionResult 模型
  context/        memory_usage_hints（每轮临时记忆使用策略，不落盘）
  event/          EventType、Event、event_builders、EventRouter
  state/          AgentState、reducer、runtime_history、summary_builder
  decision/       speech_llm / behavior_distraction / wellness_care / environment_care / sensor_status / rule handler
  llm/            LLMClient、prompt_builder、reply_validator（TTS 文案轻量校验）
  memory/         MemoryService（异步 LLM 记忆抽取与检索）
  guard/          Guard（防刷屏、不在场、TTS 中、冷却）
  action/         ActionType、Action、action_builders、ActionRealizer
  device/         DeviceAdapter
  scheduler/      AutonomousScheduler（多任务周期调度）
  prompts/        speech_recognized.md、behavior_distraction_check.md、wellness_care_check.md、
                  environment_care_check.md、focus_complete_care.md、tts_reply_quality.md（公共播报质量约束）
  policy_config.py
```

## 事件分流（EventRouter）

| 分流 | 事件 | 处理方式 |
|------|------|----------|
| `speech_llm` | `speech_recognized` | 单次 LLM，理解命令/对话/专注/音量 |
| `behavior_distraction` | `system_triggered` + `trigger=behavior_distraction_check` | 玩手机分心专项检查 LLM（20s） |
| `wellness_care` | `system_triggered` + `trigger=wellness_care_check` | 疲劳/情绪/姿态关怀 LLM（30s） |
| `environment_care` | `system_triggered` + `trigger=environment_care_check` | 环境关怀 LLM（60s，可 no_op） |
| `sensor_status` | `system_triggered` + `trigger=sensor_status_report` | 传感器数值规则播报（300s，不调 LLM、不检索 Memory） |
| `rule` | `focus_start_requested`、`focus_stop_requested`、`timer_finished` | RuleHandler 定计时语义；`timer_finished` 的专注结束关怀文案另由 LLM 个性化（带轮换兴趣，失败回退默认文案） |
| `state_only` | 其余全部（含已废弃删除的 `periodic_state_check`） | 只更新 State / RuntimeHistory |

## 动作闭集（5 种）

`speak`、`display`、`start_timer`、`stop_timer`、`set_tts_volume`

## LLM 决策入口

每个 LLM 入口在调用前由 `AgentCore._user_context()` 统一构造 `user_context`：
`profile + preferences + memories + recent_interaction + memory_usage_hints`，
并输出一行 `[记忆检索]` 追溯日志（本轮检索/使用了哪些记忆）。

- **speech_recognized**：用户语音 → `build_speech_prompt` → LLM → Intent → ActionRealizer；同时异步 `submit_speech_memory`。
- **behavior_distraction_check**：Python 严格预检（窗口占比 + 仍在玩 + **硬性要求 YOLO 检出手机**）→ 确定性提醒，LLM 只写措辞。
- **wellness_care_check**：Python 算 `should_care` 与 focus（fatigue/emotion/posture）→ LLM 只写一句关怀文案，不能否决强触发、不能改成环境提醒。
- **environment_care_check**：仅环境（光照/温度/湿度/噪声）→ LLM 判断是否 no_op + 文案。
- **sensor_status_report**：每 300s 确定性数值播报，不调用 LLM、不检索 Memory、不生成 memory_usage_hints。

另有后台 `memory_extract` LLM（异步，不阻塞主链路，只从用户语音抽取）。

播报质量：`prompt_builder` 为所有会进 TTS 的 LLM 入口注入 `tts_reply_quality.md`；
`reply_validator` 在 Handler 落地前拦截空串与明显半句话（`invalid_llm_reply`），不恢复硬编码 fallback。

> 注：`periodic_state_check` 为旧入口，已废弃删除；不再是 LLM 入口，落入 `state_only`。

## 测试

- 自动化：`python -m pytest -q`
- 深入验收流程：`docs/testing/agent_functional_test_guide.md`（画像、记忆、`memory_usage_hints` 轮换、各 Handler、媒体、Guard）

## 策略配置（policy_config.py）

- `LLMRoutingPolicy`：分流与 prompt role
- `SchedulePolicy` / `ScheduledTaskPolicy`：多任务周期与优先级
- `BehaviorDistractionCheckPolicy` / `WellnessCareCheckPolicy` / `EnvironmentCareCheckPolicy`：各自检阈值
- `SensorReportPolicy`：传感器播报边界
- `GuardPolicy`：提醒冷却与打断保护
- `ActionPolicy`：动作边界与默认文案
- `MemoryPolicy`：记忆抽取与检索（含各 context_type 类型权重）

## 示例

```python
from src.agent.core import build_default_core
from src.agent.event.event_model import Event

core = build_default_core(llm_service=fake_llm)
actions, results = core.handle_event(
    Event(type="speech_recognized", timestamp=1000, payload={"text": "开始专注"})
)
```
