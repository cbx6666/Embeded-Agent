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
from src.agent.llm_agent.schemas import ResponseDraft, SituationFrame, parse_json_object
from src.services.llm_service import LLMService


class ResponseWriter:
    """表达生成 LLM 角色。

    输入已审查的 IntentPlan，输出 speak_text/display_text/tone。没有需要表达的
    intent 时跳过，模型失败时使用 LLMService 的文本 fallback。
    """

    role_name = "response_writer"

    def __init__(self, prompt_path: Path | None = None) -> None:
        self.prompt_path = prompt_path or _prompt_path("response_writer.md")

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
            f"{_read_prompt(self.prompt_path)}\n\n"
            f"SituationFrame JSON:\n{situation.to_dict()}\n\n"
            f"IntentPlan JSON:\n{plan.to_dict()}\n\n"
            f"Context JSON:\n{context.to_prompt_json()}"
        )
        try:
            raw = llm_service.complete_json(self.role_name, prompt)
            draft = ResponseDraft.from_dict(parse_json_object(raw))
            draft = _remove_false_preference_commitments(draft, plan)
            if not draft.speak_text and not draft.display_text:
                raise ValueError("response is empty")
            return draft, {"raw": raw, "fallback": False}
        except Exception as exc:
            fallback = llm_service.generate_reply(context.user_text or situation.summary, None)
            draft = _remove_false_preference_commitments(
                ResponseDraft(speak_text=fallback, display_text=fallback),
                plan,
            )
            return draft, {
                "fallback": True,
                "error": str(exc),
            }


def _needs_response(plan: IntentPlan) -> bool:
    """判断计划中是否存在需要用户可见表达的 intent。"""

    return any(intent.type in {"answer_user", "suggest_rest", "remind_distraction", "update_status_feedback", "display_update"} for intent in plan.intents)


def _remove_false_preference_commitments(draft: ResponseDraft, plan: IntentPlan) -> ResponseDraft:
    """Keep acknowledgements truthful when no persistent preference action exists."""

    if _has_persistent_preference_update(plan):
        return draft
    if not any(_contains_false_commitment(text) for text in (draft.speak_text, draft.display_text)):
        return draft
    replacement = "好的，我会尽量少打扰你。"
    return ResponseDraft(speak_text=replacement, display_text=replacement, tone=draft.tone or "calm")


def _has_persistent_preference_update(plan: IntentPlan) -> bool:
    return any(
        intent.type in {"update_user_profile", "update_user_preference", "set_user_preference"}
        for intent in plan.intents
    )


def _contains_false_commitment(text: str) -> bool:
    normalized = str(text).strip().lower()
    if not normalized:
        return False
    forbidden = (
        "我已经记住",
        "我记住了",
        "以后一定",
        "我已经设置",
        "我会长期调整",
        "我会调整提醒方式",
        "以后我会少提醒",
        "以后我都会少提醒你",
        "i will remember",
        "i've remembered",
        "i have remembered",
        "i've set",
        "i have set",
        "from now on",
    )
    return any(phrase in normalized for phrase in forbidden)


def _prompt_path(name: str) -> Path:
    """定位本角色 prompt 文件。"""

    return Path(__file__).resolve().parents[1] / "prompts" / name


def _read_prompt(path: Path) -> str:
    """读取 prompt；缺失时使用严格 JSON 的最小兜底提示。"""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return "Return strict JSON for the requested role."
