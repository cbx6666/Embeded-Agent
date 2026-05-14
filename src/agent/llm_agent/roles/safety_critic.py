from __future__ import annotations

"""
SafetyCritic 角色模块。

本模块负责让 LLM 审查 IntentPlan 是否过度打扰、违背偏好、与当前状态冲突或
存在风险。上游输入是 SituationFrame、IntentPlan 和 AgentContext，下游输出是
SafetyReview 以及可能被修订的 IntentPlan。

本模块不执行动作、不修改状态，也不替代 DeterministicGuard；它负责认知层
自审，硬边界仍由代码执行。
"""

from pathlib import Path

from src.agent.decision.intent_model import IntentPlan
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.prompt_io import prompt_path, read_prompt
from src.agent.llm_agent.schemas import SafetyReview, SituationFrame, parse_json_object
from src.services.llm_service import LLMService

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class SafetyCritic:
    """安全审查 LLM 角色。

    输入初步 IntentPlan，输出 approve/revise/reject。模型失败时默认保留原计划
    进入确定性边界，而不是直接放行到设备。
    """

    role_name = "safety_critic"

    def __init__(self, prompt_path_override: Path | None = None) -> None:
        self.prompt_path = prompt_path_override or prompt_path(_PROMPTS_DIR, "safety_critic.md")

    def review(
        self,
        context: AgentContext,
        situation: SituationFrame,
        plan: IntentPlan,
        llm_service: LLMService,
    ) -> tuple[SafetyReview, IntentPlan, dict[str, object]]:
        """审查并可能修订 IntentPlan。

        reject 会降级成 no_op；revise 必须提供完整 revised_plan。后续仍会经过
        IntentPlanValidator 和 DeterministicGuard。
        """

        prompt = (
            f"{read_prompt(self.prompt_path)}\n\n"
            f"SituationFrame JSON:\n{situation.to_dict()}\n\n"
            f"IntentPlan JSON:\n{plan.to_dict()}\n\n"
            f"Context JSON:\n{context.to_prompt_json()}"
        )
        try:
            raw = llm_service.complete_json(self.role_name, prompt)
            review = SafetyReview.from_dict(parse_json_object(raw))
        except Exception as exc:
            review = SafetyReview(decision="approve", reason="SafetyCritic fallback approval.")
            return review, plan, {"prompt": prompt, "fallback": True, "error": str(exc)}

        if review.decision == "reject":
            from src.agent.decision.intent_model import no_op_plan

            return review, no_op_plan(review.reason or "SafetyCritic rejected the plan."), {
                "prompt": prompt,
                "raw": raw,
                "fallback": False,
            }
        if review.decision == "revise" and review.revised_plan is not None:
            return review, review.revised_plan, {"prompt": prompt, "raw": raw, "fallback": False}
        return review, plan, {"prompt": prompt, "raw": raw, "fallback": False}
