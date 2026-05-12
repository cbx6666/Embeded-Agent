# Agent Architecture

当前个性化架构采用四个核心概念：

- `RuntimeHistory`：短期运行历史。
- `LongTermMemory`：有证据、可衰减、可处理冲突的长期记忆。
- `UserProfile`：用户明确声明或系统明确配置的权威资料。
- `PersonalContext`：决策层唯一读取的只读上下文快照。

主链路：

`Event -> RuntimeHistory -> LongTermMemoryPipeline -> PersonalContextBuilder -> DecisionPipeline -> Action`
