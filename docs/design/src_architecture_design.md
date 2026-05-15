# src 架构设计文档

## 1. 项目定位

当前项目是面向嵌入式交互场景的 **LLM-centered Agent Runtime Prototype**。它不是一个只把用户输入发给模型、再把模型文本返回给用户的聊天机器人，而是围绕 `Event / State / Memory / PersonalContext / Decision / Action` 建模的单机 Agent Runtime。

核心设计原则是：

```text
语义归 prompt，边界归 runtime。
```

也就是说：

- LLM 负责场景理解、意图规划、安全自审、表达生成和长期记忆候选提取。
- Python runtime 负责事件归约、状态维护、schema validation、guard、store、action realization、device execution、trace 和 replay。

这种分层的目标不是让 LLM 绕过系统边界直接控制设备，而是让 LLM 只在受约束的语义层输出结构化中间结果，再由 deterministic runtime boundary 判断这些结果是否能进入下一层。

当前 `src/` 的主问题域包括：

| 维度 | 当前建模 |
| --- | --- |
| 外部输入 | `Event` |
| 运行状态 | `AgentState` |
| 短期上下文 | `RuntimeHistory` |
| 长期学习 | `LongTermMemory` |
| 显式用户资料 | `UserProfile` |
| 决策期个性化上下文 | `PersonalContext` |
| LLM 认知结果 | `SituationFrame / IntentPlan / SafetyReview / ResponseDraft` |
| 执行协议 | `Action` |
| 执行反馈 | `ActionResult` |
| 可观测性 | `RuntimeTrace` |

## 2. src 总体目录结构

当前 `src/` 的实际目录结构如下：

```text
src/
  main.py
  adapters/
  agent/
  services/
  storage/

src/agent/
  core.py
  reducer.py
  prompt_io.py
  action/
  config/
  decision/
  event/
  execution/
  llm_agent/
  memory/
  state/
  user/
```

> 说明：当前实现中 `event` 和 `action` 是 package，而不是单文件 `event.py` 或 `action.py`。公开导入入口仍然是 `from src.agent.event import Event` 和 `from src.agent.action import Action`。

### `src/main.py`

| 项目 | 说明 |
| --- | --- |
| 它是什么 | CLI 启动入口，创建默认 `AgentCore`，接收命令行输入，按需启动视觉适配器 |
| 它不是什么 | 不承担决策逻辑，不直接调用 LLM，不直接写 store |
| 为什么存在 | 给本地实验和嵌入式原型提供一个可运行入口 |
| 输入 | CLI 文本、`/mock` 命令、profile 命令、可选视觉事件 |
| 输出 | 标准 `Event` 进入 `AgentCore`，或将 state/profile/history 渲染到控制台 |
| 上下游 | 上游为终端/适配器，下游为 `AgentCore.handle_event()` |

### `src/agent/`

`agent/` 是 Agent Runtime 的核心目录，包含主调度、事件、状态、归约、决策、记忆、用户上下文和执行边界。

| 子目录 / 文件 | 职责 | 不应该承担的职责 | 关系 |
| --- | --- | --- | --- |
| `core.py` | 单事件调度中枢 | 不理解语义，不直接写 LLM 结果到 store/profile/state | 串联全链路 |
| `reducer.py` | `Event -> AgentState` 确定性归约 | 不调用 LLM，不生成 action | 在决策前更新状态 |
| `event/` | 统一事件协议与事件构造函数 | 不执行动作，不保存历史 | 由 adapters/services/core 产生或消费 |
| `state/` | `AgentState` 及其子状态 | 不做业务推理 | 被 reducer、guard、context builder 读取 |
| `action/` | 统一动作协议与动作构造函数 | 不表示 LLM intent，不执行设备 | 由 `ActionRealizer` 产生 |
| `decision/` | 决策链路、validator、guard、realizer | 不读 store，不写 memory，不执行设备 | 连接 LLM 语义输出与 action |
| `llm_agent/` | 四角色 LLM 编排与 schema | 不落地 action，不写 state/store | 被 `DecisionPipeline` 调用 |
| `memory/` | 长期记忆候选提取、审查、合并、校验 | 不做当前动作决策，不保存 runtime history | 写入 `LongTermMemoryStore` |
| `user/` | `UserProfile` 与 `PersonalContext` | 不调用 LLM，不直接执行动作 | 决策期 personalization 快照 |
| `execution/` | 设备适配、动作结果、trace、闭环调度 | 不做 LLM 语义判断 | 执行 `Action` 并记录结果 |
| `config/` | 策略参数 dataclass | 不是 prompt，不是协议，不是 registry | 供 guard/context/history/retrieval/action 使用 |

### `src/adapters/`

`adapters/` 是外部世界与内部 `Event / Action` 协议之间的边界。

| 文件 / 目录 | 职责 |
| --- | --- |
| `cli_input.py` | 将 CLI 文本解析为标准事件，例如专注开始/结束或普通用户文本 |
| `mock_input.py` | 将 `/mock ...` 命令转换为状态更新事件 |
| `profile_cli.py` | 将 profile CLI 命令委托给 `AgentCore` 公开 API |
| `console_output.py` | 将 `speak/display/render_pet_expression/set_light_state` 等动作映射到控制台输出 |
| `voice_adapter.py` | 将 ASR 结果包装成 `speech_recognized`，将 `speak` action 交给 TTS 后端 |
| `behavior_adapter.py` | 将行为识别结果包装为 presence/attention 事件，并做最小阈值与去抖 |
| `vision_affect/` | 摄像头、疲劳和情绪检测适配层，向内核只发送标准事件 |

适配层不应该修改 `AgentState`，也不应该直接调用 `DecisionPipeline`。它只做输入/输出协议转换。

### `src/services/`

`services/` 放置运行期服务：

| 文件 | 职责 |
| --- | --- |
| `llm_service.py` | DeepSeek Chat Completions 访问入口 |
| `runtime_history_service.py` | 维护短期运行历史窗口 |
| `timer_service.py` | 专注计时 tick 服务 |
| `user_profile_service.py` | 显式用户资料和偏好的业务入口 |

当前代码中没有独立的 `MemoryService`。长期记忆由 `LongTermMemoryPipeline` 编排，短期历史由 `RuntimeHistoryService` 维护。

### `src/storage/`

`storage/` 是 JSON 持久化层：

| 文件 | 职责 |
| --- | --- |
| `json_store.py` | 保存和恢复 `AgentState` |
| `long_term_memory_store.py` | 保存长期记忆，处理去重、冲突、confidence、decay |
| `user_profile_store.py` | 保存显式 `UserProfile` |

Store 层不调用 LLM，不执行当前决策，也不越过 service/pipeline 直接做语义推断。

## 3. 总体运行链路

主链路是单事件同步处理流程：

```text
External Input / Internal Trigger
        |
        v
      Event
        |
        v
     Reducer
        |
        v
 RuntimeHistoryService
        |
        v
 LongTermMemoryPipeline
        |
        v
 PersonalContextBuilder
        |
        v
 AgentContextBuilder
        |
        v
 LLMAgentOrchestrator
        |
        v
 IntentPlanValidator
        |
        v
 DeterministicGuard
        |
        v
 ActionRealizer
        |
        v
 DeviceAdapter
        |
        v
   ActionResult
        |
        v
   RuntimeTrace
```

更细的职责拆分如下：

| 步骤 | 输入 | 输出 | 做什么 | 不做什么 |
| --- | --- | --- | --- | --- |
| `Event` | CLI、sensor、timer、adapter | `Event` dataclass | 描述发生了什么 | 不直接触发设备动作 |
| `Reducer` | `AgentState + Event` | 新 `AgentState` | 确定性更新状态 | 不调用 LLM，不写长期记忆 |
| `RuntimeHistoryService` | `AgentState + Event/Action` | 更新 `state.runtime_history` | 记录短期窗口并裁剪 | 不沉淀长期偏好 |
| `LongTermMemoryPipeline` | `Event/Action outcome + state summary` | `LongTermMemoryRunResult` | 观察、提取、审查、合并、校验、写 store | 不参与当前动作决策 |
| `PersonalContextBuilder` | `AgentState + UserProfile + LongTermMemory` | `PersonalContext` | 生成决策期只读个性化快照 | 不调用 LLM，不写 store |
| `AgentContextBuilder` | `Event + State + PersonalContext` | `AgentContext` | 压缩成 prompt 上下文 | 不读取 store，不做语义补丁 |
| `LLMAgentOrchestrator` | `AgentContext + LLMService` | `AgentRun` | 四角色 LLM 认知编排 | 不生成设备 action |
| `IntentPlanValidator` | `IntentPlan` | validation result | 校验 schema、intent 白名单和禁止字段 | 不做自然语言判断 |
| `DeterministicGuard` | `IntentPlan + AgentContext` | `GuardDecision` | 执行 cooldown、presence、risk 等硬边界 | 不理解文本语义 |
| `ActionRealizer` | `Guarded IntentPlan + ResponseDraft` | `list[Action]` | 将 intent 确定性转成 action | 不执行设备 |
| `DeviceAdapter` | `Action` | `ActionResult` | 调用 timer/output 后端并捕获异常 | 不重新规划 |
| `RuntimeTrace` | 各阶段 payload | JSON/debug/assertion trace | 记录确定性可观测链路 | 不做分布式 tracing |

这样分层的原因是：LLM 输出可以失败、畸形或夹带越界字段；runtime 必须在每个关键边界处保持可验证、可回放、可测试。

## 4. AgentCore 设计

`src/agent/core.py` 中的 `AgentCore` 是单事件调度中枢。

### 定位

| 项目 | 说明 |
| --- | --- |
| 它是什么 | 一轮 `Event` 处理的 orchestration 中心 |
| 它不是什么 | 不理解用户语义，不直接让 LLM 写 store/profile/state，不直接读取 `LongTermMemoryStore` 做决策 |
| 为什么存在 | 保证所有输入都经过相同的数据流、边界和 trace |
| 输入 | `Event` |
| 输出 | `tuple[list[Action], list[ActionResult]]` |
| 上游 | `main.py`、`AgentLoop`、timer callback、adapter |
| 下游 | reducer、history service、memory pipeline、personal context、decision pipeline、device adapter、store |

### `handle_event()` 步骤

当前 `handle_event()` 的顺序是：

1. 复制 `previous_state`。
2. 创建 `RuntimeTrace`，记录 `event:received`。
3. 调用 `reduce_state()`，得到当前事件后的 `AgentState`。
4. 对用户相关事件调用 `UserProfileService.touch_user()`。
5. 调用 `RuntimeHistoryService.record_event()`，并在用户文本/语音事件中记录用户消息。
6. 调用 `LongTermMemoryPipeline.process_event()`，观察当前事件是否产生长期记忆。
7. 调用 `PersonalContextBuilder.build()`，构建本轮决策只读个性化快照。
8. 调用 `DecisionPipeline.decide()`，得到 intent、guard 结果、action 和决策 trace。
9. 调用 `DeviceAdapter.execute()` 执行动作，得到 `ActionResult`。
10. 对成功动作更新 runtime history、focus timer 派生状态、cooldown 和消息记录。
11. 调用 `LongTermMemoryPipeline.process_actions()`，观察 action outcome 是否产生长期记忆。
12. 更新 `last_intents`、`last_decision_result`、`last_action_results` 等调试字段。
13. 调用 `RuntimeHistoryService.trim()`，限制 runtime history 大小。
14. 调用 `JsonStore.save_state()`，保存运行时状态。
15. 保存 `last_runtime_trace`。

`AgentCore` 的关键边界是：它只调度，不把任何 LLM 输出直接落到状态或存储中。

### 默认构造

`build_default_core()` 创建默认运行组件：

| 组件 | 默认实现 |
| --- | --- |
| Runtime state store | `JsonStore("data/runtime/runtime_store.json")` |
| User profile store | `UserProfileStore("data/user/user_profiles.json")` |
| Long-term memory store | `LongTermMemoryStore("data/memory/long_term_memory.json")` |
| LLM | `LLMService()` |
| Timer | `TimerService(background=True)` |
| Output | `ConsoleOutput()` |

如果未配置 DeepSeek API，`LLMService` 会抛出配置错误；生产链路不提供本地语义 mock。测试和实验脚本使用 `tests.fakes.fake_llm_service.FakeLLMService` 或 `scripts/runtime_experiments/common.py` 中的 `ExperimentLLM`。

## 5. Event / State / Reducer 层

### Event

`src/agent/event/event_model.py` 定义 `Event`：

```python
Event(type: EventType, timestamp: int, payload: dict[str, Any])
```

Event 是外部输入和内部触发的统一抽象，例如：

- `user_text_input`
- `speech_recognized`
- `focus_start_requested`
- `focus_stop_requested`
- `timer_ticked`
- `timer_finished`
- `user_presence_updated`
- `user_attention_updated`
- `user_emotion_updated`
- `user_fatigue_updated`
- `system_triggered`

Event 不直接触发动作。它只是描述“发生了什么”，动作必须经过 reducer、decision、validator、guard 和 realizer。

### State

`src/agent/state/agent_state.py` 定义 `AgentState`，由多个子状态组成：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `user` | `UserState` | presence、attention、emotion、fatigue |
| `interaction` | `InteractionState` | 对话模式、最后响应时间 |
| `focus` | `FocusState` | 专注是否 active、剩余时间、目标时长 |
| `environment` | `EnvironmentState` | 光照、噪音、温湿度 |
| `cooldown` | `CooldownState` | 提醒类 action 的最近时间 |
| `runtime_history` | `RuntimeHistory` | 短期事件、消息、动作和摘要窗口 |
| `current_user_id` | `str` | 当前用户 |

State 不做业务推理，也不调用 LLM。它是 runtime boundary 中可持久化的当前事实快照。

### RuntimeHistory

`RuntimeHistory` 是 `AgentState` 的短期历史子结构。它保存最近事件、消息、动作、提醒记录、注意力记录、环境记录、情绪采样和滚动摘要。

它不是 `LongTermMemory`：

| 对比 | RuntimeHistory | LongTermMemory |
| --- | --- | --- |
| 生命周期 | 短期窗口 | 长期持久化 |
| 写入者 | `RuntimeHistoryService` | `LongTermMemoryPipeline` |
| 内容 | 最近发生的事实 | 经证据、validator 和 store 处理的稳定记忆 |
| 是否表达偏好真相 | 否 | 是，但带 confidence/decay/evidence |
| 是否用于 prompt | 通过 `PersonalContext` 压缩后使用 | 通过 `PersonalContext` 分桶和检索后使用 |

### Reducer

`src/agent/reducer.py` 实现 `reduce_state(state, event)`。它负责确定性地将事件归约到状态，例如更新 focus、user、environment、current user 等。

Reducer 的边界：

- 不调用 LLM。
- 不写 `LongTermMemoryStore`。
- 不生成 action。
- 不解析自然语言语义。

## 6. Memory 层设计

`src/agent/memory/` 负责长期记忆写入管线。

### LongTermMemoryPipeline

`LongTermMemoryPipeline` 是长期记忆唯一写入编排入口。它的链路是：

```text
observe
  -> extract
  -> critic
  -> consolidate
  -> validate
  -> store
```

| 阶段 | 实现 | 输入 | 输出 | 职责 | 边界 |
| --- | --- | --- | --- | --- | --- |
| observe | `_observe()` + `memory_observer.md` | `LongTermMemoryContext` | JSON metadata | 判断是否值得进入记忆提取 | 只决定是否继续，不写 store |
| extract | `_extract()` + `memory_extractor.md` | `LongTermMemoryContext` | `MemoryCandidate` 列表 | 提取候选记忆 | LLM 只能提出 candidate |
| critic | `_critic()` + `memory_critic.md` | candidates | approved candidates + reasons | 语义审查候选 | 不是最终写入边界 |
| consolidate | `MemoryConsolidator` + `memory_consolidator.md` | existing memories + candidates | consolidated candidates | 合并、去重、表达归一 | 仍输出 candidate |
| validate | `MemoryValidator` | candidate | error 或 `None` | 确定性校验证据、类型、内容 | 最终 store 前硬边界 |
| store | `LongTermMemoryStore.upsert_candidate()` | valid candidate | `LongTermMemory` | 持久化、合并、冲突、decay metadata | 不调用 LLM |

### MemoryCandidate

`MemoryCandidate` 是 LLM 或 consolidate 阶段提出的候选，不是已写入记忆。

允许的长期记忆类型定义在 `memory_candidate.py`：

- `behavior_preference`
- `behavior_pattern`
- `interaction_style`
- `active_constraint`
- `uncertain`

不在该集合中的类型会被 `MemoryValidator` 拒绝。

### MemoryValidator

`MemoryValidator` 是长期记忆写入前的确定性边界。它验证：

- `memory_type` 是否在允许集合内。
- `content` 是否非空。
- `evidence` 是否存在。
- evidence 是否能回指 event、dialogue、action outcome 或 result。
- mock evidence 是否被拒绝。
- `behavior_preference` 是否包含用户文本或语音证据，包括 `source_event_type`、`timestamp`、`source` 和 `user_text/snippet`。
- `confidence` 是否在范围内。
- 显式 profile 数据是否被错误写入长期记忆。

因此，LLM hallucination 不能直接进入长期记忆。LLM 只能产生候选，最终写入必须通过 deterministic validator。

### LongTermMemoryStore

`LongTermMemoryStore` 保存 `LongTermMemory`，并负责：

- 按 user 隔离读取。
- 只返回 active memory，除非显式 `include_inactive=True`。
- 读取时可根据 `now` 刷新 decay。
- 对同一 preference key/value 做 canonical merge。
- 合并 evidence，并将 evidence 列表限制在最近 20 条。
- 根据新增 evidence 提升 confidence。
- 处理同一 preference key 的 contradiction，把旧记忆标记为 `contradicted`。
- 写入 JSON 文件。

Store 不调用 LLM，也不做当前决策。

### 防污染机制

当前长期记忆防污染依赖多层边界：

| 风险 | 处理 |
| --- | --- |
| LLM 编造 preference | evidence 不 grounded 时 validator 拒绝 |
| mock/local evidence 被当成真实证据 | `MOCK_EVIDENCE_SOURCES` 被拒绝 |
| weak evidence | 只有 `llm/model/inference/summary` 等弱来源且无 grounded key 时拒绝 |
| contradictory preference | `LongTermMemoryStore` 标记旧 active memory 为 `contradicted` |
| 记忆无限增长 | 同类候选 upsert 合并，evidence capped，PersonalContext 分桶压缩 |
| 过时记忆影响过大 | `apply_decay()` 和读取时 `now` 可刷新 decay |

### Prompt 文件

长期记忆 LLM 阶段使用：

- `src/agent/memory/prompts/memory_observer.md`
- `src/agent/memory/prompts/memory_extractor.md`
- `src/agent/memory/prompts/memory_critic.md`
- `src/agent/memory/prompts/memory_consolidator.md`

这些 prompt 属于语义层；写入边界仍在 Python runtime。

## 7. User / PersonalContext 层设计

`src/agent/user/` 处理显式用户资料和决策期个性化上下文。

### 三类材料

| 材料 | 来源 | 权威含义 | 生命周期 |
| --- | --- | --- | --- |
| `UserProfile` | `UserProfileService` + `UserProfileStore` | 用户明确声明或系统明确配置的资料/偏好 | 长期、显式、可编辑 |
| `LongTermMemory` | `LongTermMemoryPipeline` + `LongTermMemoryStore` | 从交互中沉淀出的行为偏好、模式、约束 | 长期、证据化、带 confidence/decay |
| `RuntimeHistory` | `RuntimeHistoryService` | 最近事件、消息、动作和状态摘要 | 短期窗口 |

### PersonalContext

`PersonalContext` 是 decision-time personalization snapshot。

它是：

- `DecisionPipeline` 可以读取的只读个性化上下文。
- `UserProfile`、`LongTermMemory` 和 `RuntimeHistory` 的组合快照。
- prompt compression 和 personalization guidance 的输入。

它不是：

- store。
- profile。
- memory。
- runtime history 本体。
- LLM 可直接修改的对象。

### PersonalContext 字段

| 字段 | 说明 |
| --- | --- |
| `user_id` | 当前用户 ID |
| `user_profile` | `UserProfileService.profile_context()` 返回的显式资料 |
| `profile_items` | 从显式 preference 渲染出的高权威检索项 |
| `behavior_preferences` | 长期记忆中的行为偏好 bucket |
| `behavior_patterns` | 长期记忆中的行为模式 bucket |
| `interaction_style` | 长期记忆中的交互风格 bucket |
| `active_constraints` | 长期记忆中的当前约束 bucket |
| `uncertain_memories` | 低置信度、冲突或不确定记忆 |
| `runtime_history` | 裁剪后的短期历史快照 |
| `runtime_items` | 从最近消息、动作、事件生成的检索项 |
| `compression` | 分桶压缩元数据，例如输入/输出数量 |
| `authoritative_sources` | 说明不同信息类型的权威来源 |

### PersonalContextBuilder

`PersonalContextBuilder` 负责：

1. 读取 `LongTermMemoryStore.list(user_id, now=event.timestamp)`。
2. 读取 `UserProfileService.profile_context(user_id)`。
3. 将 profile preference 渲染为 `profile_items`。
4. 将长期记忆按类型分桶。
5. 检测 `LongTermMemory` 与 `UserProfile` 的冲突。
6. 将冲突或低置信度记忆放入 `uncertain_memories`。
7. 按 `ContextPolicyConfig.max_memory_items_per_bucket` 压缩每个 bucket。
8. 从 runtime history 中过滤噪声事件，生成短期上下文和 `runtime_items`。

它不调用 LLM，不写 store，不生成 intent。

### `retrieve_relevant()`

`PersonalContext.retrieve_relevant()` 是 prompt compression / context retrieval，而不是长期记忆价值判断。

当前检索是 deterministic hand-weighted ranking，权重来自 `RetrievalPolicyConfig`。默认返回仍然是原来的 `list[dict]`，以保持调用方兼容。

当前还提供可解释接口：

- `retrieve_relevant_with_scores(...)`
- `explain_retrieval(...)`

每个候选 item 的 `score_breakdown` 包括：

- `source_weight`
- `event_type_weight`
- `priority_score`
- `confidence_bonus`
- `evidence_bonus`
- `conflict_penalty`
- `content_term_bonus`
- `tag_term_bonus`
- `final_score`

该检索不使用 LLM 排序，不引入 embedding，也不依赖向量数据库。它的可解释性用于 debug 和实验评估，例如 `scripts/runtime_experiments/retrieval_quality_experiment.py`。

## 8. Decision 层设计

`src/agent/decision/` 是从本轮上下文到动作计划的 deterministic boundary。

核心链路：

```text
AgentContextBuilder
  -> LLMAgentOrchestrator
  -> IntentPlanValidator
  -> DeterministicGuard
  -> ActionRealizer
```

### AgentContextBuilder

| 项目 | 说明 |
| --- | --- |
| 它是什么 | 将 `Event`、`AgentState` 和 `PersonalContext` 压缩成 `AgentContext` |
| 它不是什么 | 不读取 store，不做 keyword NLP，不做语义修补 |
| 输入 | `previous_state`、`current_state`、`event`、`personal_context` |
| 输出 | `AgentContext` |
| 下游 | 四角色 LLM prompt |

`AgentContext.to_prompt_dict()` 包含：

- event type、timestamp、payload、user_text。
- current state summary。
- previous state summary。
- full `personal_context`。
- `personalization_guidance`。
- recent messages。
- relevant memories。

### IntentPlanValidator

`IntentPlanValidator` 校验 LLM 输出的结构边界：

- intent type 必须在 `REGISTERED_INTENT_TYPES` 中。
- payload 必须是 object。
- payload 不允许包含 `action`、`actions`、`state_patch`。
- plan risk_level 必须是 `low/medium/high`。

它不做自然语言语义判断，也不替 LLM 重新规划。校验失败后，`DecisionPipeline` 将 plan 降级为 `no_op_plan()`。

当前注册 intent 包括：

```text
answer_user
start_focus
stop_focus
complete_focus
suggest_rest
remind_distraction
update_status_feedback
adjust_environment_feedback
voice_interaction
display_update
continue_focus
reduce_reminder_frequency
set_tts_volume
no_op
```

### DeterministicGuard

`DeterministicGuard` 负责不能交给 LLM 的硬边界：

- high risk plan 直接阻断。
- 用户离场时阻断 interruptive intents。
- `suggest_rest`、`remind_distraction`、`adjust_environment_feedback` 等提醒类 intent 执行 cooldown。
- `focus_start_requested` 不允许混入休息提醒、降频、环境调整等 intent。
- 非用户消息事件不允许自主 `answer_user` 且 `requires_llm=True`。

它只使用结构化状态和 cooldown，不做关键词判断。

### ActionRealizer

`ActionRealizer` 将通过 guard 的 `IntentPlan` 转成注册 `Action`。

它负责：

- intent 按 priority 排序。
- 专注时长裁剪。
- TTS volume 裁剪。
- 默认文案 fallback。
- speak/display 等可见动作去重。

它不调用 LLM，不执行设备，不修改状态。

## 9. LLM Agent 四角色设计

`src/agent/llm_agent/` 定义四个 LLM 角色：

1. `SituationAnalyst`
2. `IntentPlanner`
3. `SafetyCritic`
4. `ResponseWriter`

### 四角色链路

```text
AgentContext
  -> SituationAnalyst
  -> SituationFrame
  -> IntentPlanner
  -> IntentPlan
  -> SafetyCritic
  -> SafetyReview + reviewed IntentPlan
  -> ResponseWriter
  -> ResponseDraft
  -> AgentRun
```

| 角色 | Prompt | 输入 | 输出 | 不做什么 |
| --- | --- | --- | --- | --- |
| `SituationAnalyst` | `situation_analyst.md` | `AgentContext` | `SituationFrame` | 不生成 action，不写状态 |
| `IntentPlanner` | `intent_planner.md` | `SituationFrame + AgentContext` | `IntentPlan` | 不执行设备，不绕过 registered intent |
| `SafetyCritic` | `safety_critic.md` | `SituationFrame + IntentPlan + AgentContext` | `SafetyReview + reviewed plan` | 不替代 deterministic guard |
| `ResponseWriter` | `response_writer.md` | `SituationFrame + reviewed IntentPlan + AgentContext` | `ResponseDraft` | 不做行为决策，不生成 action |

每个 role class 负责：

- 读取对应 prompt markdown。
- 拼接 `AgentContext` 或上游 schema。
- 调用 `LLMService.complete_json()`。
- 解析 JSON。
- 在失败时返回可解释 fallback metadata。

不在 Python 中写自然语言语义规则，也不通过关键词修补 ResponseWriter 的文本。

### AgentRun

`AgentRun` 聚合一轮 LLM 认知输出：

| 字段 | 说明 |
| --- | --- |
| `situation` | `SituationFrame` |
| `plan` | 最终进入 validator 的 `IntentPlan` |
| `safety_review` | `SafetyReview` |
| `response` | `ResponseDraft` |
| `used_llm` | 是否使用 LLM |
| `fallback_reason` | 哪些角色 fallback |
| `stage_metadata` | 各角色 prompt、raw output、fallback/error/skipped 信息 |

`DecisionPipeline` 会将 `stage_metadata` 中的 prompt 和 raw output 写入 `RuntimeTrace`。

## 10. Action / Execution 层设计

### Action

`src/agent/action/action_model.py` 定义 `Action`：

```python
Action(type: ActionType, payload: dict[str, Any])
```

Action 是系统内部动作协议。它不是 intent。

| 对比 | Intent | Action |
| --- | --- | --- |
| 所在层 | LLM semantic planning boundary | runtime execution protocol |
| 示例 | `suggest_rest` | `speak`、`display` |
| 生成者 | LLM `IntentPlanner` | `ActionRealizer` |
| 是否可直接执行 | 否 | 进入 `DeviceAdapter` 后执行 |
| 是否必须通过 guard | 是 | 已经在 guard 后产生 |

### DeviceAdapter

`src/agent/execution/device_adapter.py` 是设备执行边界：

- `start_timer` 调用 `TimerService.start()`。
- `stop_timer` 调用 `TimerService.stop()`。
- 其他动作交给 `ConsoleOutput.execute()` 或后续真实设备适配器。
- 捕获异常并返回失败 `ActionResult`。

`DeviceAdapter` 不重新解释语义，不生成新的 intent。

### ActionResult

`ActionResult` 记录执行结果：

```python
ActionResult(
  action_type: str,
  success: bool,
  timestamp: int,
  reason: str = "",
  payload: dict[str, Any] = {}
)
```

它会被：

- `AgentCore` 写入 `last_action_results`。
- `RuntimeHistoryService.record_action()` 间接记录成功动作。
- `LongTermMemoryPipeline.process_actions()` 作为 outcome 观察材料。
- `RuntimeTrace` 记录在 `action_result:executed` 中。

### AgentLoop 与内部事件

`src/agent/execution/loop.py` 提供 `AgentLoop`，可以将 action result 生成的内部事件继续回流处理，最大步数由 `max_steps` 限制。

`src/agent/execution/internal_events.py` 可以将成功 `speak/display/start_timer/stop_timer` 或失败 action 转换为 `system_triggered` 内部事件。

`DecisionPolicyConfig` 会过滤内部 action result trigger，避免内部完成事件反复触发 LLM 决策。

## 11. Services 层设计

### LLMService

`src/services/llm_service.py` 是生产链路唯一的真实 LLM 访问入口。

| 项目 | 说明 |
| --- | --- |
| Provider | DeepSeek Chat Completions |
| 配置来源 | `.env` 或环境变量 |
| 主要方法 | `complete_json()`、`generate_reply()`、`chat_completion()` |
| 是否 mock | 否 |
| 是否本地生成 intent/memory | 否 |
| 调用失败 | 抛出异常，由上层 role fallback、validator、guard 处理 |

需要配置：

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

测试和实验使用显式 fake/double，不属于生产 `LLMService`。

### RuntimeHistoryService

`RuntimeHistoryService` 是短期 runtime history 的唯一写入入口。它负责：

- 记录 recent events。
- 记录 user/agent/display messages。
- 记录 recent actions。
- 记录 reminder records。
- 记录 attention/environment/emotion samples。
- 按窗口策略生成 emotion summary。
- 在事件结束时裁剪所有窗口。

它属于 deterministic runtime boundary，不调用 LLM，不写长期记忆。

### TimerService

`TimerService` 提供倒计时能力：

- `start(duration_sec, callback)`
- `stop()`
- `is_active()`

当 `background=False` 时，主要用于测试和实验，不启动后台线程。真实运行默认可以使用后台 tick 线程。

### UserProfileService

`UserProfileService` 是显式用户资料入口。它负责：

- 创建和切换用户。
- 更新显式 info 和 preference。
- 更新 `last_seen_at`。
- 渲染 profile/users 文本。
- 提供 `profile_context()` 给 `PersonalContextBuilder`。
- 提供 TTS 和提醒风格等显式偏好辅助读取。

它不从 LLM 自由文本直接推断 profile，不写 LongTermMemory。

### MemoryService

当前 `src/services/` 中没有 `MemoryService`。记忆相关职责分布为：

- 短期运行历史：`RuntimeHistoryService`。
- 长期记忆提取与写入：`LongTermMemoryPipeline`。
- 长期记忆持久化：`LongTermMemoryStore`。
- 决策期记忆读取与压缩：`PersonalContextBuilder` / `PersonalContext`。

## 12. Storage 层设计

### JsonStore

`JsonStore` 保存 `AgentState`：

- `load_state_dict()`：读取 JSON 中的 `state` 字段，失败返回 `None`。
- `save_state()`：写入 `updated_at` 和 `state.to_dict()`。

它保存的是运行时状态，不保存用户 profile，也不保存长期记忆。

### LongTermMemoryStore

`LongTermMemoryStore` 保存长期记忆：

- 默认路径：`data/memory/long_term_memory.json`。
- JSON 顶层字段：`updated_at`、`memories`。
- 读取时支持 user filter、inactive filter、decay refresh。
- 写入时通过 deterministic upsert 处理重复、冲突、confidence、decay 和 evidence cap。

它不调用 LLM，不参与当前决策，只作为 `LongTermMemoryPipeline` 的写入目标和 `PersonalContextBuilder` 的读取来源。

### UserProfileStore

`UserProfileStore` 保存显式用户资料：

- 默认路径：`data/user/user_profiles.json`。
- JSON 顶层字段：`updated_at`、`profiles`。
- 只负责 load/save 原始 profile 字典。

只有 `UserProfileService` 应该直接使用它。

## 13. RuntimeTrace / Replay / 实验验证

### RuntimeTrace

`src/agent/execution/trace.py` 提供轻量 deterministic trace：

| 类型 / 方法 | 说明 |
| --- | --- |
| `RuntimeTraceEvent` | 单条 trace event，包含 `sequence/stage/label/payload` |
| `RuntimeTrace.add()` | 添加阶段事件 |
| `RuntimeTrace.extend()` | 合并其他 trace |
| `RuntimeTrace.to_dict()` | 转为稳定 dict |
| `RuntimeTrace.to_json()` | JSON dump 字符串 |
| `RuntimeTrace.dump_json()` | 写入 JSON 文件 |
| `RuntimeTrace.to_debug_string()` | 生成 debug 文本 |
| `RuntimeTrace.debug_print()` | 打印 debug 文本 |
| `RuntimeTrace.stages()` | 返回阶段 tuple，便于断言 |
| `RuntimeTrace.find()` | 按 stage/label 查找事件 |

Trace 使用 `_stable()` 将 dataclass、dict、list 转成 JSON-stable 结构，并按 key 排序，保证 replay 和测试断言稳定。

### Trace 覆盖链路

当前 trace 覆盖：

```text
Event
-> Reducer
-> MemoryPipeline
-> PersonalContext
-> AgentContext
-> Prompt
-> LLM output
-> Validator
-> Guard
-> ActionRealizer
-> Action
-> ActionResult
```

其中 prompt 和 raw LLM output 来自 LLM role 的 `stage_metadata`。

### tests/scenarios

`tests/scenarios/` 是面向真实行为链路的场景测试，不只是 unit test：

| 文件 | 验证重点 |
| --- | --- |
| `test_real_agent_behavior.py` | 多轮偏好、专注、离场、guard、trace |
| `test_memory_pollution.py` | hallucinated/weak/invalid memory 不进入 store，重复合并和 decay |
| `test_personalization_consistency.py` | 同一用户个性化一致，不同用户隔离，runtime history 不污染长期偏好 |
| `test_boundary_and_llm_failures.py` | malformed JSON、非法 intent、额外控制字段、empty response fallback、guard/action deterministic |
| `test_runtime_stability.py` | 100+ events、context bounded、cooldown、memory compression、retrieval bounded |
| `test_trace_observability.py` | trace JSON、debug print、find/stages 断言能力 |

### tests/replay

`tests/replay/` 提供 deterministic replay 验证：

- `replay_harness.py` 在临时目录构造新 runtime，用相同 event log 执行。
- `test_deterministic_replay.py` 验证同一 event log 得到一致 actions 和 trace JSON。
- 同时验证输入变化会改变 action result。

Replay 的目标是证明 deterministic boundary 稳定，而不是证明 LLM provider 输出稳定。测试中使用 fake LLM。

### scripts/runtime_experiments

`scripts/runtime_experiments/` 是单机长期运行实验入口，默认输出到 `data/experiments/runtime/` 或 `data/experiments/retrieval/`：

| 脚本 | 内容 |
| --- | --- |
| `study_session_experiment.py` | 模拟 40 分钟学习、分心、疲劳、离开、反馈和偏好强化 |
| `long_term_memory_experiment.py` | 模拟偏好变化、矛盾偏好、reinforcement、weak evidence、decay |
| `multi_user_isolation_experiment.py` | 验证用户 A/B 记忆、检索和个性化隔离 |
| `hallucination_resistance_experiment.py` | 模拟 fake memory、fake intent、malformed schema 和 guard 拦截 |
| `retrieval_quality_experiment.py` | 构造 profile、long-term memory、runtime history、冲突记忆并输出检索质量报告 |
| `common.py` | 实验环境、deterministic LLM double、recorder、报告输出工具 |

实验输出通常包含：

- `events.json`
- `action_timeline.json`
- `trace_logs.json`
- `memory_snapshots.json`
- `personalization_snapshots.json`
- `metrics.json`
- `report.md`

`retrieval_quality_experiment.py` 额外输出：

- `retrieval_cases.json`
- `retrieval_results.json`
- `score_breakdown.json`
- `metrics.json`
- `report.md`

### scripts/debug

`scripts/debug/` 提供命令行调试入口：

| 脚本 | 作用 |
| --- | --- |
| `inspect_memory.py` | 查看 `LongTermMemoryStore` 内容，支持 user filter 和 inactive memory |
| `inspect_personal_context.py` | 构造一次 PersonalContext 并查看 retrieval |
| `inspect_trace.py` | 查看 trace JSON，可按 stage filter |
| `replay_events.py` | 将 event log 通过 deterministic experiment runtime replay |

这些工具保持单机、轻量、文件驱动，不引入 tracing framework、metrics server 或 dashboard。

## 14. PolicyConfig 设计

`src/agent/config/policy_config.py` 集中保存策略参数。

| Config | 使用方 | 职责 |
| --- | --- | --- |
| `GuardPolicyConfig` | `DeterministicGuard` | interruptive intents、cooldown、presence hard boundary |
| `DecisionPolicyConfig` | `DecisionPipeline` | system trigger allow/ignore 策略 |
| `ContextPolicyConfig` | `PersonalContextBuilder`、`AgentContextBuilder` | recent window、memory bucket、噪声事件过滤、relevant memory limit |
| `ActionPolicyConfig` | `ActionRealizer` | focus duration、TTS volume 等数值边界 |
| `CopyPolicyConfig` | `ActionRealizer` | fallback 文案 |
| `RuntimeHistoryPolicyConfig` | `RuntimeHistoryService` | runtime history 各窗口大小 |
| `RetrievalPolicyConfig` | `PersonalContext.retrieve_relevant()`、`PersonalContextBuilder` | source/event/confidence/evidence/conflict/term 权重 |

PolicyConfig 是策略参数集中地。

它不是：

- Event/Intent/Action 协议。
- prompt。
- semantic patch。
- plugin/registry/manager。

它的作用是把会影响运行行为但不属于稳定协议的参数集中管理，减少 guard、realizer、history、context builder 和 retrieval 之间的散落配置。

## 15. Deterministic Boundary 设计原则

### LLM 负责

- 场景理解。
- 意图规划。
- 安全审查建议。
- 用户可见表达生成。
- 长期记忆候选提取。
- 长期记忆候选语义审查和 consolidate 建议。

### Python runtime 负责

- schema validation。
- registered type 白名单。
- guard。
- state update。
- runtime history 维护。
- store 持久化。
- action realization。
- device execution。
- trace。
- replay。
- deterministic policy。

### 明确禁止

- Python keyword NLP 替代 LLM 语义理解。
- Python 对 LLM 文本做自然语言修补。
- LLM 直接写 `AgentState`。
- LLM 直接写 `LongTermMemoryStore`。
- LLM 直接写 `UserProfileStore`。
- `DecisionPipeline` 直接读取 store。
- `Action` 绕过 `IntentPlanValidator` 和 `DeterministicGuard`。
- `ActionRealizer` 直接调用设备。

这条边界保证：语义能力可以通过 prompt 和 LLM role 演进，执行安全和可回放性则保持在确定性代码中。

## 16. 典型场景链路示例

### 示例 1：用户说“别太频繁提醒我”

```text
user_text_input
  -> Reducer 更新 interaction/user 相关状态
  -> RuntimeHistoryService 记录用户消息
  -> LongTermMemoryPipeline.observe 判断值得记
  -> extract 生成 behavior_preference candidate
  -> critic 批准或拒绝
  -> consolidate 与已有记忆合并
  -> MemoryValidator 校验证据
  -> LongTermMemoryStore 写入/合并
  -> PersonalContextBuilder 下次构建时读取
  -> retrieve_relevant 选入 relevant_memories
  -> ResponseWriter 通过 AgentContext 看到低打扰偏好
  -> ActionRealizer 生成 speak/display
  -> Trace 记录全链路
```

关键边界：

- 用户偏好必须带有 `user_text_input` 或 `speech_recognized` 证据。
- 如果后续用户表达相反偏好，`LongTermMemoryStore` 会把旧 preference 标记为 `contradicted`。
- `ResponseWriter` 只负责表达，不直接修改提醒频率策略；硬边界仍由 guard/cooldown 控制。

### 示例 2：专注中系统触发休息提醒

```text
system_triggered(trigger=focus_health_check)
  -> Reducer 保持或更新结构化状态
  -> RuntimeHistoryService 记录事件
  -> DecisionPolicyConfig 判断 trigger 是否允许进入 LLM planning
  -> PersonalContextBuilder 构造当前用户快照
  -> LLMAgentOrchestrator 产生 suggest_rest 等 intent
  -> IntentPlanValidator 校验注册 intent
  -> DeterministicGuard 检查 presence 和 cooldown
  -> ActionRealizer 生成 speak/display notification
  -> DeviceAdapter 执行
  -> RuntimeHistoryService 记录提醒和 cooldown
  -> RuntimeTrace 记录结果
```

关键边界：

- 如果用户 `presence=away`，interruptive intent 会被 guard 阻断。
- 如果 cooldown 未过，rest reminder 会被阻断。
- 内部 action result trigger 会被 `DecisionPolicyConfig` 忽略，避免循环打扰。

### 示例 3：LLM 输出非法 intent

```text
IntentPlanner raw output
  -> IntentPlan.from_dict 解析
  -> IntentPlanValidator 检查 registered type / payload / forbidden fields
  -> validation errors
  -> no_op_plan
  -> DeterministicGuard
  -> ActionRealizer
  -> no action
  -> RuntimeTrace 记录 fallback reason
```

关键边界：

- 未注册 intent 不会进入 action。
- payload 中夹带 `actions` 或 `state_patch` 会被拒绝。
- 即便 `SafetyCritic` approve，validator 和 guard 仍是最终 deterministic boundary。

## 17. 当前设计优势

当前 `src/` 架构的主要优势是：

- 单向数据流清晰：`Event -> State/Context -> Intent -> Action -> Result`。
- LLM semantic layer 与 deterministic runtime boundary 分离。
- LLM 只能提出候选，不能直接写 state/store/profile。
- 长期记忆写入有 observer、extractor、critic、consolidator、validator、store 多层边界。
- 显式 `UserProfile` 与推断型 `LongTermMemory` 分离，减少 personalization 污染。
- `PersonalContext` 将 profile、memory、runtime history 收口成只读快照。
- `retrieve_relevant()` 默认兼容旧调用，同时提供可解释 score breakdown。
- `RuntimeTrace` 覆盖 prompt、LLM raw output、validator、guard、action 等关键节点。
- `tests/scenarios/` 以真实多轮场景验证行为一致性、记忆污染、个性化隔离和长期稳定。
- `tests/replay/` 验证 deterministic boundary 对同输入的稳定性。
- `scripts/runtime_experiments/` 支持长时间运行实验和报告输出。
- prompt 文件独立存在，语义策略可调试。

## 18. 当前限制与后续演进

### 当前限制

| 限制 | 说明 |
| --- | --- |
| LLM provider 单一 | `LLMService` 当前面向 DeepSeek Chat Completions |
| 检索仍是手工权重 | `RetrievalPolicyConfig` 仍依赖 deterministic hand weights，需要实验数据持续评估 |
| 真实硬件接入仍需完善 | `ConsoleOutput` 和部分 adapter 已提供边界，但真实屏幕、语音、灯光等后端仍需集成 |
| 长时间真实运行数据不足 | 已有脚本实验和测试，但仍需要更多真实用户长期数据 |
| prompt 质量需要迭代 | LLM 行为稳定性依赖 prompt、schema 和测试共同收敛 |
| trace 是轻量实现 | 当前 trace 适合 debug、JSON dump 和测试断言，不是生产级 observability 平台 |
| replay 依赖 deterministic LLM double | 对外部真实 LLM 输出的稳定性不做保证，replay 主要验证 runtime boundary |

### 后续演进

- 基于 `retrieve_relevant_with_scores()` 和 `retrieval_quality_experiment.py` 持续校准 retrieval policy。
- 丰富 runtime metrics，但保持单机文件输出。
- 增加更多 scenario benchmark，覆盖更多用户行为和设备失败场景。
- 完善真实硬件 adapter integration。
- 将实验报告沉淀为架构汇报材料和 PPT 输入。
- 扩展 trace 分析工具，但不引入分布式 tracing 或 metrics platform。
- 增加更长周期的真实运行数据采集，用于评估 memory decay、conflict 和 personalization consistency。
