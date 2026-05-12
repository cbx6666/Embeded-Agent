# LLM Agent

## 职责

`llm_agent/` 是 Agent 的高层认知核心。它把紧凑 `AgentContext` 交给四个角色：

`SituationAnalyst -> IntentPlanner -> SafetyCritic -> ResponseWriter`

## 不负责什么

本目录不生成设备 Action，不修改 AgentState，不写 MemoryStore，也不绕过 deterministic boundary。

## 核心文件

- `agent_context.py`：构建供 LLM 使用的压缩上下文。
- `agent_orchestrator.py`：串联四角色并输出 `AgentRun`。
- `schemas.py`：定义 SituationFrame、SafetyReview、ResponseDraft 和 fallback。
- `roles/`：四个 LLM 角色的调用与解析。
- `prompts/`：每个角色的边界提示词。

## 上游和下游

上游是 `DecisionPipeline` 提供的事件、状态和 ProfileSnapshot。下游是 `IntentPlanValidator`、`DeterministicGuard` 和 `ActionRealizer`。

## 扩展方式

不要无限增加 agent。新增认知能力时优先扩展现有角色的 schema/prompt；只有出现明确独立职责时才新增角色。

## 示例

```python
agent_run = orchestrator.decide(context, llm_service)
```
