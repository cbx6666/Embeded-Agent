# Memory

## 职责

`memory/` 实现 LLM-managed Memory：

`Event / Outcome -> MemoryContextBuilder -> LLMMemoryManager -> MemoryValidator -> MemoryStore -> ProfileSnapshotBuilder`

## 不负责什么

本目录不做当前动作决策，不允许 LLM 直接写 store/profile，不保留固定阈值画像提取器，也不让 DecisionPipeline 直接读取 MemoryStore。

## 核心文件

- `llm_memory_manager.py`：四阶段记忆观察、提取、审查、整合。
- `schemas.py`：MemoryCandidate 和允许的 memory type。
- `memory_store.py`：JSON 持久化和 evidence 合并。
- `profile_snapshot_builder.py`：生成决策层唯一可读的 ProfileSnapshot。
- `memory_pipeline.py`：AgentCore 使用的 memory 门面。
- `prompts/`：记忆角色提示词。

## 上游和下游

上游是 AgentCore 提供的事件和动作 outcome。下游是 MemoryStore 与 ProfileSnapshot。

## 扩展方式

新增 memory 类型：加入 `ALLOWED_MEMORY_TYPES`，更新 memory prompts、ProfileSnapshotBuilder 分桶逻辑，并补 MemoryValidator 测试。

## 示例

```python
snapshot = memory_pipeline.build_profile_snapshot(user_id, state, event)
```
