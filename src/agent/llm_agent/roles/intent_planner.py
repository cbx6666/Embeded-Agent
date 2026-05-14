from __future__ import annotations

"""
IntentPlanner 角色模块。

本模块负责把 SituationFrame 转换为 IntentPlan。上游输入是
SituationAnalyst 的结构化理解和 AgentContext，下游输出是只包含注册 intent
的 IntentPlan。

本模块不生成 Action、不修改 AgentState、不控制设备。未注册 intent 会在
后续 IntentPlanValidator 被拒绝。
"""

from pathlib import Path

from src.agent.decision.intent_model import IntentPlan, REGISTERED_INTENT_TYPES
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.prompt_io import prompt_path, read_prompt
from src.agent.llm_agent.schemas import SituationFrame, fallback_plan_for_event, parse_json_object
from src.services.llm_service import LLMService

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class IntentPlanner:
    """意图规划 LLM 角色。

    输入 SituationFrame，输出 IntentPlan。它承担语义规划，但最终执行权仍在
    validator、guard 和 realizer 的确定性边界中。
    """

    role_name = "intent_planner"

    def __init__(self, prompt_path_override: Path | None = None) -> None:
        self.prompt_path = prompt_path_override or prompt_path(_PROMPTS_DIR, "intent_planner.md")

    def plan(
        self,
        context: AgentContext,
        situation: SituationFrame,
        llm_service: LLMService,
    ) -> tuple[IntentPlan, dict[str, object]]:
        """调用 LLM 生成 IntentPlan。

        prompt 明确列出注册 intent 类型，模型失败或输出畸形时回退到事件级安全
        fallback plan。
        """

        prompt = (
            f"{read_prompt(self.prompt_path)}\n\n"
            f"Registered intent types: {sorted(REGISTERED_INTENT_TYPES)}\n\n"
            f"SituationFrame JSON:\n{situation.to_dict()}\n\n"
            f"Context JSON:\n{context.to_prompt_json()}"
        )
        try:
            raw = llm_service.complete_json(self.role_name, prompt)
            plan = IntentPlan.from_dict(parse_json_object(raw))
            return plan, {"prompt": prompt, "raw": raw, "fallback": False}
        except Exception as exc:
            plan = fallback_plan_for_event(context.event_type, context.user_text)
            return plan, {"prompt": prompt, "fallback": True, "error": str(exc)}
