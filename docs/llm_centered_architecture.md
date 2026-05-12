# LLM-Centered Agent 架构

## 为什么移除规则中心架构

旧方向的问题不是“不够工程化”，而是智能来源错位。关键词、阈值、状态分支和配置文件可以协调系统，但不能真正理解用户意图、上下文风险和长期偏好。继续增加类、注册表和配置只会把同一组规则包得更厚。

## 为什么关键词规则不是智能

关键词规则只能发现文本里出现了某个 token，无法可靠判断用户为什么这么说、是否希望被打扰、当前状态是否冲突、哪些记忆相关。语义理解、风险分析和风格生成由 LLM 角色负责；代码只保留确定性边界。

## 主链路

`Event -> Reducer -> ProfileSnapshot -> AgentContextBuilder -> LLMAgentOrchestrator -> IntentPlanValidator -> DeterministicGuard -> ActionRealizer -> DeviceAdapter`

Event、Intent、Action 主线仍然保留，只是认知核心从规则系统切换为 LLM 多阶段推理。

## 四角色如何协作

- `SituationAnalyst`：理解当前发生了什么、用户可能意图、风险和不确定性。禁止输出 Action。
- `IntentPlanner`：把 SituationFrame 转成注册 IntentPlan。只能使用注册 intent。
- `SafetyCritic`：审查是否过度打扰、违背偏好、状态冲突或风险过高。可以 approve、revise 或 reject。
- `ResponseWriter`：只生成 speak/display 文本和 tone，不负责行为决策。

四角色是固定的认知阶段，不是为了数量而堆 agent。

## LLM 的权限边界

LLM 可以理解、规划、审查和表达，但不能直接：

- 修改 AgentState
- 写 MemoryStore
- 写 UserProfile
- 生成设备 Action
- 绕过注册 intent/action 白名单

所有模型输出都必须经过 schema validation、guard 和 action realization。

## Deterministic Guard 与 ActionRealizer

`IntentPlanValidator` 拒绝非法 JSON、未注册 intent、action 字段和 state_patch。

`DeterministicGuard` 处理硬边界：presence safety、cooldown、高风险阻断、非用户触发场景禁止自主 LLM 回复。

`ActionRealizer` 把通过边界的 IntentPlan 转成注册 Action，不调用 LLM，不做语义理解。

## LLM-managed Memory

Memory 由 `LLMMemoryManager` 管理候选、审查和整合。代码只负责 schema、evidence、持久化和 ProfileSnapshot。

详见 `docs/llm_memory_architecture.md`。

## 如何新增一个 intent

1. 在 `src/agent/decision/intent_model.py` 加入 `REGISTERED_INTENT_TYPES`。
2. 更新 `llm_agent/prompts/intent_planner.md` 中的规划说明。
3. 在 `src/agent/decision/action_realizer.py` 实现 intent 到 Action 的确定性落地。
4. 如涉及风险，补充 `DeterministicGuard`。
5. 增加 validator、guard、realizer 和端到端测试。

## 如何新增一个 action

1. 在 `src/agent/action/types.py` 注册 ActionType。
2. 在 `src/agent/action/action_builders.py` 添加构造函数。
3. 在 `runtime/device_adapter.py` 或具体 adapter 中实现执行。
4. 增加动作模型和执行边界测试。

## 如何新增一个 memory 类型

1. 在 `src/agent/memory/schemas.py` 添加 `ALLOWED_MEMORY_TYPES`。
2. 更新 memory prompts。
3. 在 `ProfileSnapshotBuilder` 中定义该类型如何进入快照。
4. 增加 MemoryValidator 和 ProfileSnapshot 测试。

## 如何调试 trace

`AgentLoop` 会记录每步事件、intent、action、result 和 decision metadata。CLI 中使用 `/trace` 或 `/last` 查看最近决策。若动作未执行，优先检查：

- `fallback_reason`
- `validator.errors`
- `guard` findings
- `action_realizer.action_count`
