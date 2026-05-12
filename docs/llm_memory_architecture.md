# LLM-managed Memory 架构

## 为什么 Memory 也要 LLM-managed

固定统计和阈值只能说明“发生了几次”，不能可靠判断“这对用户长期画像是否重要”。记忆提取需要理解语境、显式偏好、约束、语气和交互结果，因此由 LLM 负责候选生成、审查和整合。

## 主链路

`Event / Interaction / Outcome -> MemoryContextBuilder -> LLMMemoryManager -> MemoryValidator -> MemoryStore -> ProfileSnapshotBuilder`

## 四阶段记忆角色

- `MemoryObserver`：判断当前事件或 outcome 是否值得进入长期记忆流程。
- `MemoryExtractor`：生成 MemoryCandidate。
- `MemoryCritic`：拒绝模糊、无证据、隐私风险或低价值候选。
- `MemoryConsolidator`：把新候选与已有记忆合并，避免重复和冲突膨胀。

## LLM 不能做什么

LLM 不能直接写 MemoryStore，不能直接修改 UserProfile，也不能决定当前 Action。它只能输出候选记忆 JSON。

## Deterministic Boundary

`MemoryValidator` 校验：

- memory_type 必须注册
- content 不能为空
- evidence 必须存在
- confidence 必须在 0 到 1

只有通过校验的候选才能进入 `MemoryStore`。

## ProfileSnapshot

`ProfileSnapshotBuilder` 是决策层读取记忆的唯一入口。它把长期记忆压缩为：

- explicit_preferences
- behavior_patterns
- interaction_style
- active_constraints
- recent_context
- uncertain_memories

`AgentContextBuilder` 只消费 ProfileSnapshot，不直接读取 MemoryStore。

## 如何新增一个 memory 类型

1. 在 `memory/schemas.py` 注册类型。
2. 更新 `memory/prompts/*`，告诉 LLM 何时产生该类型。
3. 在 `profile_snapshot_builder.py` 中定义分桶方式。
4. 为 MemoryValidator、LLMMemoryManager 和 ProfileSnapshotBuilder 补测试。

## 调试方式

查看 `MemoryPipeline.last_result`：

- `stage_metadata.memory_observer`
- `stage_metadata.memory_extractor`
- `stage_metadata.memory_critic`
- `stage_metadata.memory_consolidator`
- `rejected`
- `stored`

如果快照缺少预期记忆，先确认候选是否有 evidence，再确认是否通过 validator 和 store upsert。
