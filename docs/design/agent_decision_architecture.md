# Decision Architecture

The decision layer is now LLM-centered.

See `docs/llm_centered_architecture.md` for the full rationale and boundary
definition.

Current decision flow:

`AgentContext -> LLMAgentOrchestrator -> IntentPlanValidator -> DeterministicGuard -> ActionRealizer`
