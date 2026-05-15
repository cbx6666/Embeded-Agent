# LongTermMemory

`memory/` 只负责长期记忆学习链路：

`Event / Outcome -> LongTermMemoryContextBuilder -> observe -> extract -> critic -> consolidate -> validate -> LongTermMemoryStore`

它不维护短期历史，不保存显式用户资料，不参与当前动作决策，也不允许 LLM 直接写 state/store/profile。

核心文件：

- `long_term_memory.py`：已经验证并持久化的长期记忆实体。
- `memory_candidate.py`：LLM 提出的候选记忆。
- `memory_validator.py`：写入仓库前的确定性边界。
- `memory_consolidator.py`：候选合并与冲突整理。
- `long_term_memory_pipeline.py`：observe -> extract -> critic -> consolidate -> validate -> store。
- `prompts/`：长期记忆 observer、extractor、critic、consolidator 的 LLM prompt。

长期记忆持久化仓库位于 `src/storage/long_term_memory_store.py`。决策层不能直接读取 store，
只能读取 `PersonalContextBuilder` 生成的 `PersonalContext`。
