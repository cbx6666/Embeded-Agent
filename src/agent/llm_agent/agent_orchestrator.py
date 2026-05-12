"""
LLM Agent 编排模块。

本模块负责串联 SituationAnalyst、IntentPlanner、SafetyCritic 和
ResponseWriter，完成一轮高层认知决策。上游输入是 `AgentContextBuilder`
生成的紧凑 AgentContext，下游输出是包含 SituationFrame、IntentPlan、安全审查
和表达草稿的 AgentRun。

本模块不直接生成 Action，不修改 AgentState，也不直接写入 MemoryStore。底层
动作落地由 `decision/action_realizer.py` 负责，安全边界由 validator 和 guard
负责。
"""

from __future__ import annotations

"""
LLM Agent 编排模块。

本模块负责串联 SituationAnalyst、IntentPlanner、SafetyCritic 和
ResponseWriter，完成一轮高层认知决策。上游输入是
`AgentContextBuilder` 生成的紧凑 AgentContext，下游输出是包含
SituationFrame、IntentPlan、安全审查和表达草稿的 AgentRun。

本模块不直接生成 Action，不修改 AgentState，也不直接写入 MemoryStore。
底层动作落地由 `decision/action_realizer.py` 负责，安全边界由
`decision/validator.py` 和 `decision/guard.py` 负责。
"""

from src.agent.llm_agent.agent_context import AgentContext
from src.agent.llm_agent.roles.intent_planner import IntentPlanner
from src.agent.llm_agent.roles.response_writer import ResponseWriter
from src.agent.llm_agent.roles.safety_critic import SafetyCritic
from src.agent.llm_agent.roles.situation_analyst import SituationAnalyst
from src.agent.llm_agent.schemas import AgentRun
from src.services.llm_service import LLMService


class LLMAgentOrchestrator:
    """四角色 LLM 认知编排器。

    职责是让多个 LLM 角色按固定顺序协作：先理解场景，再规划 Intent，再做
    安全审查，最后生成表达文本。它输入 AgentContext 和 LLMService，输出
    AgentRun。它不负责执行动作、不负责持久化、不负责硬件控制。
    """

    def __init__(
        self,
        *,
        situation_analyst: SituationAnalyst | None = None,
        intent_planner: IntentPlanner | None = None,
        safety_critic: SafetyCritic | None = None,
        response_writer: ResponseWriter | None = None,
    ) -> None:
        self.situation_analyst = situation_analyst or SituationAnalyst()
        self.intent_planner = intent_planner or IntentPlanner()
        self.safety_critic = safety_critic or SafetyCritic()
        self.response_writer = response_writer or ResponseWriter()

    def decide(self, context: AgentContext, llm_service: LLMService) -> AgentRun:
        """执行一轮 LLM-centered 决策。

        任一角色失败时，角色内部会产生可解释 fallback；这里汇总每个阶段的
        metadata，让 trace 能说明模型在哪一层降级。
        """

        stage_metadata: dict[str, object] = {}

        situation, situation_meta = self.situation_analyst.analyze(context, llm_service)
        stage_metadata["situation_analyst"] = situation_meta

        plan, planner_meta = self.intent_planner.plan(context, situation, llm_service)
        stage_metadata["intent_planner"] = planner_meta

        safety_review, reviewed_plan, safety_meta = self.safety_critic.review(
            context,
            situation,
            plan,
            llm_service,
        )
        stage_metadata["safety_critic"] = safety_meta

        response, response_meta = self.response_writer.write(
            context,
            situation,
            reviewed_plan,
            llm_service,
        )
        stage_metadata["response_writer"] = response_meta

        fallback_reason = _fallback_reason(stage_metadata)
        return AgentRun(
            situation=situation,
            plan=reviewed_plan,
            safety_review=safety_review,
            response=response,
            used_llm=True,
            fallback_reason=fallback_reason,
            stage_metadata=stage_metadata,
        )


def _fallback_reason(stage_metadata: dict[str, object]) -> str | None:
    """从各角色 metadata 中提取统一 fallback 原因，供 DecisionResult 记录。"""

    failed: list[str] = []
    for stage, meta in stage_metadata.items():
        if isinstance(meta, dict) and meta.get("fallback"):
            failed.append(stage)
    if not failed:
        return None
    return "llm_fallback:" + ",".join(failed)
