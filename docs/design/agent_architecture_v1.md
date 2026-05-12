# Superseded Architecture Note

The rule-centered design described here has been replaced by the
LLM-centered architecture in `docs/llm_centered_architecture.md`.

Current flow:

`Event -> Reducer -> ProfileSnapshot -> AgentContextBuilder -> LLMAgentOrchestrator -> IntentPlanValidator -> DeterministicGuard -> ActionRealizer -> DeviceAdapter`
