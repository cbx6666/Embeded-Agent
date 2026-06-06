from __future__ import annotations

"""单次 LLM 完成场景理解、Intent 规划和可选表达草稿。"""

from pathlib import Path

from src.agent.config.policy_config import LLMRolePolicyConfig
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.decision.intent_model import IntentPlan, REGISTERED_INTENT_TYPES
from src.agent.llm_agent.schemas import (
    ResponseDraft,
    SituationFrame,
    fallback_plan_for_event,
    fallback_situation,
    parse_json_object,
)
from src.agent.prompt_io import prompt_path, read_prompt
from src.services.llm_service import LLMService


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class UnifiedPlanner:
    """fast 模式的单调用认知入口。"""

    role_name = "unified_planner"

    def __init__(
        self,
        *,
        prompt_path_override: Path | None = None,
        policy_config: LLMRolePolicyConfig | None = None,
    ) -> None:
        self.prompt_path = prompt_path_override or prompt_path(
            _PROMPTS_DIR,
            "unified_planner.md",
        )
        self.policy_config = policy_config or LLMRolePolicyConfig()

    def decide(
        self,
        context: AgentContext,
        llm_service: LLMService,
    ) -> tuple[SituationFrame, IntentPlan, ResponseDraft, dict[str, object]]:
        """一次调用返回理解、计划和表达；失败时使用确定性安全 fallback。"""

        prompt = (
            f"{read_prompt(self.prompt_path)}\n\n"
            f"Registered intent types: {sorted(REGISTERED_INTENT_TYPES)}\n\n"
            f"Context JSON:\n{context.to_prompt_json()}"
        )
        try:
            raw = llm_service.complete_json(self.role_name, prompt)
            data = parse_json_object(raw)
            situation = SituationFrame.from_dict(data.get("situation", {}))
            plan = IntentPlan.from_dict(data.get("plan", {}))
            response = ResponseDraft.from_dict(data.get("response", {}))
            if any(intent.type == "answer_user" for intent in plan.intents):
                if not response.speak_text and not response.display_text:
                    raise ValueError("answer_user requires response text")
            return situation, plan, response, {
                "prompt": prompt,
                "raw": raw,
                "fallback": False,
                "model": str(getattr(llm_service, "model", "unknown")),
            }
        except Exception as exc:
            situation = fallback_situation(context.event_type, context.user_text)
            plan = fallback_plan_for_event(context.event_type, context.user_text)
            response = ResponseDraft()
            if context.user_text:
                response = ResponseDraft(
                    speak_text=self.policy_config.deterministic_fast_fallback_text,
                    display_text=self.policy_config.deterministic_fast_fallback_text,
                )
            return situation, plan, response, {
                "prompt": prompt,
                "fallback": True,
                "error": str(exc),
                "model": str(getattr(llm_service, "model", "unknown")),
            }
