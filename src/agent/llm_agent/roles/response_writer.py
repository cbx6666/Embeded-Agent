from __future__ import annotations

"""
ResponseWriter 角色模块。

本模块负责根据已批准的 IntentPlan 生成用户可见文本。上游输入是
SituationFrame、IntentPlan 和 AgentContext，下游输出是 ResponseDraft。

本模块不决定行为、不生成 Action、不修改状态；ActionRealizer 会把文本草稿
确定性地放入 speak/display 等注册动作中。
"""

from pathlib import Path

from src.agent.decision.intent_model import IntentPlan
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.prompt_io import prompt_path, read_prompt
from src.agent.llm_agent.schemas import ResponseDraft, SituationFrame, parse_json_object
from src.services.llm_service import LLMService

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class ResponseWriter:
    """表达生成 LLM 角色。

    输入已审查的 IntentPlan，输出 speak_text/display_text/tone。没有需要表达的
    intent 时跳过，模型失败时使用 LLMService 的文本 fallback。
    """

    role_name = "response_writer"

    def __init__(self, prompt_path_override: Path | None = None) -> None:
        self.prompt_path = prompt_path_override or prompt_path(_PROMPTS_DIR, "response_writer.md")


    def write(
        self,
        context: AgentContext,
        situation: SituationFrame,
        plan: IntentPlan,
        llm_service: LLMService,
    ) -> tuple[ResponseDraft, dict[str, object]]:
        """生成用户可见表达文本。

        如果模型输出空文本或非法 JSON，会回退到 `generate_reply`，确保对话不会
        因表达层失败而中断。
        """

        if not _needs_response(plan):
            return ResponseDraft(), {"skipped": True}

        prompt = (
            f"{read_prompt(self.prompt_path)}\n\n"
            f"SituationFrame JSON:\n{situation.to_dict()}\n\n"
            f"IntentPlan JSON:\n{plan.to_dict()}\n\n"
            f"Context JSON:\n{context.to_prompt_json()}"
        )
        try:
            raw = llm_service.complete_json(self.role_name, prompt)
            draft = ResponseDraft.from_dict(parse_json_object(raw))
            if not draft.speak_text and not draft.display_text:
                raise ValueError("response is empty")
            return draft, {"raw": raw, "fallback": False}
        except Exception as exc:
            fallback = llm_service.generate_reply(context.user_text or situation.summary, None)
            draft = ResponseDraft(speak_text=fallback, display_text=fallback)
            return draft, {
                "fallback": True,
                "error": str(exc),
            }


def _needs_response(plan: IntentPlan) -> bool:
    """判断计划中是否存在需要用户可见表达的 intent。"""

    return any(intent.type in {"answer_user", "suggest_rest", "remind_distraction", "update_status_feedback", "display_update"} for intent in plan.intents)
