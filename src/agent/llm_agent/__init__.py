"""LLM Agent 包公开入口。

本包导出四角色编排器、上下文构建器和角色间 schema。外部调用方应通过
LLMAgentOrchestrator.decide 使用认知链路，而不是直接让角色控制动作或状态。
"""

from src.agent.decision.agent_context_builder import AgentContext, AgentContextBuilder
from src.agent.llm_agent.agent_orchestrator import LLMAgentOrchestrator
from src.agent.llm_agent.schemas import AgentRun, ResponseDraft, SafetyReview, SituationFrame

__all__ = [
    "AgentContext",
    "AgentContextBuilder",
    "AgentRun",
    "LLMAgentOrchestrator",
    "ResponseDraft",
    "SafetyReview",
    "SituationFrame",
]
