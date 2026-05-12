# Agent Personalization Architecture

主链路：

`Event -> RuntimeHistory -> LongTermMemoryPipeline -> PersonalContextBuilder -> DecisionPipeline -> Action`

## RuntimeHistory

当前运行期的短期历史。它只保存最近事件、最近消息、最近动作、提醒记录和状态滚动摘要。
它不是长期记忆，不保存稳定偏好。

## LongTermMemory

系统从长期交互中沉淀出的可证据化记忆。来源只能是 event、dialogue、action outcome
和 repeated behaviors，并且必须经过 observe -> extract -> critic -> consolidate -> validate -> store。

LLM 只能提出 `MemoryCandidate`，不能直接写 state/store/profile。

## UserProfile

用户明确声明或系统明确配置的权威资料。display_name、age、hobbies、显式偏好和 TTS 设置
只能来自 `UserProfile`。

## PersonalContext

决策层唯一允许读取的人格上下文快照，由 `PersonalContextBuilder` 组合
RuntimeHistory、LongTermMemory 和 UserProfile 生成。它是只读 snapshot，不是 store。
