# LLM Agent

`llm_agent/` 编排 SituationAnalyst、IntentPlanner、SafetyCritic 和 ResponseWriter。

它只消费 `AgentContext`，不生成设备 Action，不修改 `AgentState`，不写
`LongTermMemoryStore`，也不绕过 validator/guard。

上游是 `DecisionPipeline` 提供的事件、状态和 `PersonalContext`；下游是
`IntentPlanValidator`、`DeterministicGuard` 和 `ActionRealizer`。

每个角色读取 `prompts/` 下对应的 markdown prompt，调用 `LLMService.complete_json()`，
解析为结构化 schema，并在失败时返回带 metadata 的 fallback 结果。
