"""
LLM Agent 编排模块。

本模块提供两种认知编排：fast 模式用 UnifiedPlanner 单次生成 Situation、IntentPlan
和 Response，仅在复杂/高风险时追加 SafetyCritic；full 模式保留
SituationAnalyst、IntentPlanner、SafetyCritic、ResponseWriter 四角色链。

本模块不直接生成 Action，不修改 AgentState，也不直接写入 LongTermMemoryStore。底层
动作落地由 `decision/action_realizer.py` 负责，安全边界由 validator 和 guard
负责。
"""

from __future__ import annotations

from src.agent.config.policy_config import LLMRolePolicyConfig
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.llm_agent.roles.intent_planner import IntentPlanner
from src.agent.llm_agent.roles.response_writer import ResponseWriter
from src.agent.llm_agent.roles.safety_critic import SafetyCritic
from src.agent.llm_agent.roles.situation_analyst import SituationAnalyst
from src.agent.llm_agent.fast_dialogue import build_fast_dialogue_prompt, fast_dialogue_role_name
from src.agent.llm_agent.unified_planner import UnifiedPlanner
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

class LLMAgentOrchestrator:
    """在单调用 fast path 与完整四角色链之间选择。"""

    def __init__(
        self,
        *,
        situation_analyst: SituationAnalyst | None = None,
        intent_planner: IntentPlanner | None = None,
        safety_critic: SafetyCritic | None = None,
        response_writer: ResponseWriter | None = None,
        unified_planner: UnifiedPlanner | None = None,
        role_policy: LLMRolePolicyConfig | None = None,
    ) -> None:
        self.situation_analyst = situation_analyst or SituationAnalyst()
        self.intent_planner = intent_planner or IntentPlanner()
        self.safety_critic = safety_critic or SafetyCritic()
        self.response_writer = response_writer or ResponseWriter()
        self.role_policy = role_policy or LLMRolePolicyConfig()
        self.unified_planner = unified_planner or UnifiedPlanner(
            policy_config=self.role_policy
        )

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

        if llm_mode == "fast":
            # 语音流式回复仍走纯文本快路径；其他请求用一次 UnifiedPlanner 同时
            # 产出 Situation、IntentPlan 和 Response。
            if _should_stream_fast_dialogue(context, llm_service):
                return self._decide_fast_dialogue(context, llm_service)
            return self._decide_adaptive(context, llm_service)

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

    def _decide_adaptive(
        self,
        context: AgentContext,
        llm_service: LLMService,
    ) -> AgentRun:
        """单次统一规划；只有复杂或高风险计划再追加 SafetyCritic。"""

        situation, plan, response, planner_meta = self.unified_planner.decide(
            context,
            llm_service,
        )
        stage_metadata: dict[str, object] = {
            "unified_planner": planner_meta,
            "situation_analyst": {"skipped": True, "reason": "merged_into_unified_planner"},
            "intent_planner": {"skipped": True, "reason": "merged_into_unified_planner"},
            "response_writer": {"skipped": True, "reason": "response_provided_by_unified_planner"},
        }
        safety_review = SafetyReview(
            decision="approve",
            reason="conditional safety review not required",
        )
        reviewed_plan = plan
        if self._needs_safety_review(plan):
            safety_review, reviewed_plan, safety_meta = self.safety_critic.review(
                context,
                situation,
                plan,
                llm_service,
            )
            stage_metadata["safety_critic"] = safety_meta
        else:
            stage_metadata["safety_critic"] = {
                "skipped": True,
                "reason": "low_risk_single_intent",
            }

        return AgentRun(
            situation=situation,
            plan=reviewed_plan,
            safety_review=safety_review,
            response=response,
            used_llm=True,
            fallback_reason=_fallback_reason(stage_metadata),
            stage_metadata=stage_metadata,
        )

    def _needs_safety_review(self, plan: object) -> bool:
        """根据结构化计划决定是否值得追加第二次 LLM 审查。"""

        intents = list(getattr(plan, "intents", []) or [])
        risk_level = str(getattr(plan, "risk_level", "low"))
        if risk_level in self.role_policy.safety_review_risk_levels:
            return True
        if len(intents) >= self.role_policy.safety_review_min_intents:
            return True
        return any(
            str(getattr(intent, "type", "")) in self.role_policy.safety_review_intent_types
            for intent in intents
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
        model_name = str(getattr(llm_service, "model", "unknown"))
        sink = getattr(llm_service, "voice_stream_sink", None)
        try:
            if sink is not None and context.event_type == "speech_recognized":
                draft, stream_meta = self._stream_fast_dialogue_text(llm_service, prompt, sink)
                stage_metadata[role_name] = {
                    **stage_metadata[role_name],
                    **stream_meta,
                    "streaming": True,
                    "fallback": False,
                    "model": model_name,
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
                    "model": model_name,
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
                "model": model_name,
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


def _should_stream_fast_dialogue(
    context: AgentContext,
    llm_service: LLMService,
) -> bool:
    return bool(
        context.event_type == "speech_recognized"
        and context.user_text.strip()
        and getattr(llm_service, "voice_stream_sink", None) is not None
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
