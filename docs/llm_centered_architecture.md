# LLM-Centered Decision Architecture

决策主链路：

`Event -> PersonalContextBuilder -> AgentContextBuilder -> LLMAgentOrchestrator -> IntentPlanValidator -> DeterministicGuard -> ActionRealizer`

`DecisionPipeline` 只消费 `PersonalContext`，不直接读取任何长期记忆仓库或用户画像仓库。

LLM 角色负责理解、规划、安全审查和表达草稿；确定性边界负责 schema 校验、guard 过滤和 action 落地。
