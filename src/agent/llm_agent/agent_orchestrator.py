"""
LLM Agent 编排模块。

本模块负责串联 SituationAnalyst、IntentPlanner、SafetyCritic 和
ResponseWriter，完成一轮高层认知决策。上游输入是 `AgentContextBuilder`
生成的紧凑 AgentContext，下游输出是包含 SituationFrame、IntentPlan、安全审查
和表达草稿的 AgentRun。

本模块不直接生成 Action，不修改 AgentState，也不直接写入 LongTermMemoryStore。底层
动作落地由 `decision/action_realizer.py` 负责，安全边界由 validator 和 guard
负责。
"""

from __future__ import annotations

from src.agent.decision.agent_context_builder import AgentContext
from src.agent.llm_agent.roles.intent_planner import IntentPlanner
from src.agent.llm_agent.roles.response_writer import ResponseWriter
from src.agent.llm_agent.roles.safety_critic import SafetyCritic
from src.agent.llm_agent.roles.situation_analyst import SituationAnalyst
from src.agent.llm_agent.fast_dialogue import build_fast_dialogue_prompt, fast_dialogue_role_name
from src.adapters.voice.voice_streaming import SentenceChunker
from src.agent.llm_agent.schemas import (
    AgentRun,
    ResponseDraft,
    SafetyReview,
    SituationFrame,
    fallback_plan_for_event,
    parse_json_object,
)
from src.services.llm_service import LLMService

_FAST_DIALOGUE_EVENT_TYPES = frozenset({"speech_recognized", "user_text_input"})


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

    def decide(
        self,
        context: AgentContext,
        llm_service: LLMService,
        *,
        llm_mode: str = "fast",
    ) -> AgentRun:
        """执行一轮 LLM-centered 决策。

        任一角色失败时，角色内部会产生可解释 fallback；这里汇总每个阶段的
        metadata，让 trace 能说明模型在哪一层降级。
        """

        if llm_mode == "fast" and _should_use_fast_dialogue(context):
            return self._decide_fast_dialogue(context, llm_service)

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

    def _decide_fast_dialogue(self, context: AgentContext, llm_service: LLMService) -> AgentRun:
        """语音/文本对话快路径：单次 LLM，携带状态/偏好/对话等关键上下文。"""

        plan = fallback_plan_for_event(context.event_type, context.user_text)
        situation = SituationFrame(
            summary="Direct user dialogue (fast mode).",
            user_intent=context.user_text[:120],
            should_respond=True,
        )
        prompt = build_fast_dialogue_prompt(context)
        role_name = fast_dialogue_role_name()
        stage_metadata: dict[str, object] = {
            role_name: {"prompt": prompt, "skipped_roles": list(_FOUR_ROLE_NAMES)},
        }
        sink = getattr(llm_service, "voice_stream_sink", None)
        try:
            if sink is not None and context.event_type == "speech_recognized":
                draft, stream_meta = self._stream_fast_dialogue_text(llm_service, prompt, sink)
                stage_metadata[role_name] = {
                    **stage_metadata[role_name],
                    **stream_meta,
                    "streaming": True,
                    "fallback": False,
                    "model": llm_service.model,
                }
            else:
                raw = llm_service.complete_json(role_name, prompt)
                draft = ResponseDraft.from_dict(parse_json_object(raw))
                if not draft.speak_text and not draft.display_text:
                    raise ValueError("empty reply")
                if not draft.speak_text:
                    draft.speak_text = draft.display_text
                if not draft.display_text:
                    draft.display_text = draft.speak_text
                stage_metadata[role_name] = {
                    **stage_metadata[role_name],
                    "raw": raw,
                    "streaming": False,
                    "fallback": False,
                    "model": llm_service.model,
                }
        except Exception as exc:
            draft = ResponseDraft(
                speak_text="我在，请再说一遍。",
                display_text="我在，请再说一遍。",
            )
            stage_metadata[role_name] = {
                **stage_metadata[role_name],
                "fallback": True,
                "error": str(exc),
                "model": llm_service.model,
            }
            return AgentRun(
                situation=situation,
                plan=plan,
                safety_review=SafetyReview(decision="approve", reason="fast_dialogue_fallback"),
                response=draft,
                used_llm=True,
                fallback_reason="llm_fallback:fast_dialogue",
                stage_metadata=stage_metadata,
            )

        return AgentRun(
            situation=situation,
            plan=plan,
            safety_review=SafetyReview(decision="approve", reason="fast_dialogue_mode"),
            response=draft,
            used_llm=True,
            fallback_reason=None,
            stage_metadata=stage_metadata,
        )

    def _stream_fast_dialogue_text(
        self,
        llm_service: LLMService,
        prompt: str,
        sink: object,
    ) -> tuple[ResponseDraft, dict[str, object]]:
        """流式生成中文回复，并按句触发 TTS sink。"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful embedded desk assistant. "
                    "Reply in concise spoken Chinese. Plain text only, no JSON or markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "请用 1-3 句中文口语直接回答用户；不要 JSON，不要声称已执行设备动作。"
                ),
            },
        ]
        chunker = SentenceChunker()
        parts: list[str] = []
        for delta in llm_service.chat_completion_stream(messages, temperature=0.4):
            parts.append(delta)
            for sentence in chunker.feed(delta):
                if hasattr(sink, "on_sentence"):
                    sink.on_sentence(sentence)
        tail = chunker.flush()
        if tail and hasattr(sink, "on_sentence"):
            sink.on_sentence(tail)

        reply = "".join(parts).strip()
        if not reply:
            reply = tail.strip()
        if not reply:
            raise ValueError("empty streamed reply")
        spoke = bool(getattr(sink, "spoke_any", False))
        return (
            ResponseDraft(
                speak_text=reply,
                display_text=reply,
                already_spoken=spoke,
            ),
            {"raw": reply, "streamed_sentences": spoke},
        )


_FOUR_ROLE_NAMES = (
    "situation_analyst",
    "intent_planner",
    "safety_critic",
    "response_writer",
)


def _should_use_fast_dialogue(context: AgentContext) -> bool:
    return context.event_type in _FAST_DIALOGUE_EVENT_TYPES and bool(context.user_text.strip())


def _fallback_reason(stage_metadata: dict[str, object]) -> str | None:
    """从各角色 metadata 中提取统一 fallback 原因，供 DecisionResult 记录。"""

    failed: list[str] = []
    for stage, meta in stage_metadata.items():
        if isinstance(meta, dict) and meta.get("fallback"):
            failed.append(stage)
    if not failed:
        return None
    return "llm_fallback:" + ",".join(failed)
