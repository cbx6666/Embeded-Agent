> **历史文档，非当前运行架构。** 请以 `README.md`、`src/agent/README.md` 与 `docs/integration/` 为准。

# src 架构设计文档

> 本文档描述 2026-06 中间态实现，其中 `periodic_state_check` 等描述已过时。旧四角色 pipeline、`PersonalContext`、`LongTermMemoryPipeline` 等已删除。

## 1. 项目定位

面向嵌入式交互场景的 **Agent Runtime**：Python runtime 负责事件、状态、分流、执行边界；LLM 只在两个固定入口参与语义决策。

```text
语义归 prompt，边界归 runtime。
```

| 维度 | 当前建模 |
| --- | --- |
| 外部输入 | `Event`（20 种闭集） |
| 运行状态 | `AgentState` |
| 短期上下文 | `RuntimeHistory`（内嵌于 AgentState） |
| 偏好记忆 | `MemoryService`（JSON，`preferences.json`） |
| 显式用户资料 | `UserProfile`（`UserProfileStore`） |
| 决策输出 | `Intent` → `Action`（5 种闭集） |
| 执行反馈 | `ActionResult` |

## 2. src 目录结构

```text
src/
  main.py                 CLI / 全栈启动入口
  adapters/               外部世界 ↔ Event/Action 协议
  agent/                  Agent 内核（见 src/agent/README.md）
  services/               LLM、Timer、UserProfile 等业务服务
  storage/                JsonStore、UserProfileStore

src/agent/
  core/                   AgentCore、models
  event/                  types、router、event_builders
  state/                  agent_state、reducer、runtime_history、summary_builder
  decision/               speech_llm / periodic_state / rule handlers
  llm/                    LLMClient、prompt_builder
  memory/                 MemoryService
  guard/                  Guard
  action/                 types、realizer、action_builders
  device/                 DeviceAdapter
  scheduler/              AutonomousScheduler
  prompts/                speech_recognized.md、periodic_state_check.md
  policy_config.py        五个策略 dataclass
```

## 3. 主链路

```text
Event
  -> reduce_state
  -> RuntimeHistoryService.record_event
  -> EventRouter.classify
       ├─ speech_llm      -> SpeechLLMHandler
       ├─ periodic_state  -> PeriodicStateHandler (+ Guard)
       ├─ rule            -> RuleHandler
       └─ state_only      -> no_op
  -> ActionRealizer
  -> DeviceAdapter.execute
  -> 异步 MemoryService（speech 偏好）+ JsonStore.save_state
```

### AgentCore（`core/agent_core.py`）

单事件调度中枢。`build_default_core()` 组装默认依赖：

| 组件 | 默认实现 |
| --- | --- |
| Runtime state | `JsonStore("data/runtime/runtime_store.json")` |
| User profile | `UserProfileStore("data/user/user_profiles.json")` |
| Preference memory | `MemoryService("data/memory/preferences.json")` |
| LLM | `LLMService()` → 包装为 `LLMClient` |
| Timer | `TimerService(background=True)` |
| Output | `ConsoleOutput()` |

## 4. Event 层

### EventType（20 种）

见 `src/agent/event/types.py`。已删除 `user_text_input`、`display_sensor_updated` 等 10 种。

### EventRouter 四类分流

| kind | 条件 |
| --- | --- |
| `speech_llm` | `type == speech_recognized` |
| `periodic_state` | `system_triggered` 且 `trigger=periodic_state_check`、`source=agent_autonomy` |
| `rule` | `focus_start_requested`、`focus_stop_requested`、`timer_finished` |
| `state_only` | 其余全部 |

旧 system trigger（`focus_health_check`、`wellness_check` 等）落入 `state_only`，不映射、不兼容。

### Reducer（`state/reducer.py`）

`reduce_state(state, event)` 确定性更新 `AgentState`。不调用 LLM，不生成 Action。

## 5. State 层

`AgentState` 子结构：`user`、`interaction`、`focus`、`environment`、`cooldown`、`runtime_history`、`current_user_id`。

`RuntimeHistoryService`（`state/runtime_history.py`）维护短期窗口：最近消息、动作、提醒记录、信号趋势等。

`summary_builder.py` 为 `periodic_state_check` 构造：

- `wellness_summary`（合并 fatigue + emotion）
- `attention_summary`、`posture_summary`、`environment_summary`
- `focus_summary`、`recent_reminders`、`user_presence`、`memory_preferences`

## 6. Decision 层

### SpeechLLMHandler

- 入口：`speech_recognized`
- 一次 `LLMClient.complete_json("speech_recognized", prompt)`
- 输出 Intent：`answer_user`、`start_focus`、`stop_focus`、`set_tts_volume`、`no_op`
- Prompt 模板：`prompts/speech_recognized.md`

### PeriodicStateHandler

- 入口：`system_triggered` + `periodic_state_check`
- 先 `build_periodic_state_summary`，再一次 LLM
- 输出：`no_op` 或提醒类 Intent（`suggest_rest`、`offer_emotion_care`、`remind_distraction`、`adjust_environment_feedback`）
- 经 `Guard.filter` 后 `ActionRealizer` 落地
- Prompt 模板：`prompts/periodic_state_check.md`

### RuleHandler

- 入口：结构化控制事件
- 0 LLM，由 reducer 前后状态确定 Intent

## 7. LLM 层

`LLMClient`（`llm/client.py`）是对 `LLMService` 的薄封装，只暴露 `complete_json(role, prompt) -> dict`。

无四角色 orchestrator、无 `fast_dialogue` / `unified_planner`。测试使用 `tests/fakes/fake_llm_service.py`。

生产配置（`.env`）：

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

## 8. Memory 层

`MemoryService`（`memory/memory_service.py`）：

| 能力 | 说明 |
| --- | --- |
| 异步写入 | `speech_recognized` 命中偏好标记词时 `submit_speech_memory`，后台线程落盘 |
| 检索 | `retrieve_preferences(user_id)` 供两个 LLM 入口拼 prompt |
| 存储 | `data/memory/preferences.json` |

已删除：`LongTermMemoryPipeline`、memory gate、LLM 记忆抽取/评审/巩固、`long_term_memory_store.py`。

## 9. Guard 层

`Guard` 只做确定性安全边界：

- 防刷屏（提醒冷却）
- TTS/语音进行中不打断
- 用户不在场不提醒
- 最近已提醒则拦截

不做复杂业务决策；业务判断是否提醒由 `periodic_state_check` LLM 完成。

## 10. Action / Device 层

### ActionType（5 种）

`speak`、`display`、`start_timer`、`stop_timer`、`set_tts_volume`

### ActionRealizer

`Intent` → `Action[]`。提醒类 Intent 附带 `kind=notification` 与 `reason`。

### DeviceAdapter

- `start_timer` / `stop_timer` → `TimerService`
- `speak` / `display` / `set_tts_volume` → output 适配器
- 未知动作 → `ActionResult(success=False, reason="unsupported_action")`

## 11. Scheduler

`AutonomousScheduler` 每 `interval_sec`（默认 30）只产生：

```json
{"type": "system_triggered", "payload": {"trigger": "periodic_state_check", "source": "agent_autonomy"}}
```

`AgentCore` 对周期检查另有 `SchedulePolicy.cooldown_sec` 门控，避免过于频繁进入 LLM。

## 12. Policy 配置

`policy_config.py` 五个 dataclass：

- `LLMRoutingPolicy`
- `SchedulePolicy`
- `GuardPolicy`
- `ActionPolicy`
- `MemoryPolicy`

## 13. Services / Storage

| 模块 | 职责 |
| --- | --- |
| `llm_service.py` | DeepSeek API |
| `timer_service.py` | 专注计时 |
| `user_profile_service.py` | 显式用户资料 CRUD |
| `user_profile_model.py` | UserProfile 数据模型 |
| `json_store.py` | AgentState 快照 |
| `user_profile_store.py` | 用户 profile JSON |

## 14. Adapters 边界

适配器只产生注册 Event、只执行注册 Action。详见 `src/adapters/README.md` 与各 `docs/integration/*.md`。

## 15. 测试

```bash
python -m pytest tests/ -q
```

核心行为测试：`tests/test_agent_refactor.py`（分流、wellness 合并、动作闭集、调度器、异步记忆等）。

`tests/scenarios/`、`tests/replay/` 等依赖旧架构的套件已删除。

## 16. 与旧架构的主要差异

| 旧 | 新 |
| --- | --- |
| 四角色 LLM orchestrator | 两个固定 LLM 入口 |
| `DecisionPipeline` + validator 链 | 三个 handler + Guard |
| `PersonalContext` + `LongTermMemoryPipeline` | `MemoryService` 偏好 + `summary_builder` |
| 5 种低频 system trigger | 仅 `periodic_state_check` |
| 11+ ActionType | 5 ActionType |
| `execution/` trace/loop | 删除；`DeviceAdapter` 直接执行 |
| `user_text_input` CLI 入口 | 删除；仅 `speech_recognized` 与结构化 CLI 命令 |
