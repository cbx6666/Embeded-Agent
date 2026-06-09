from __future__ import annotations

import re

"""wellness_care_check 疲劳 / 负面情绪 / 姿态关怀处理器（每 30s，优先级 2）。

职责边界：
- 只负责疲劳、负面情绪、姿态不佳、长时间精神状态异常；不碰环境光照/温度/湿度/噪声。
- 疲劳、情绪、姿态是 OR 关系：任一在窗口内触发即关怀。
- 媒体建议作为高优先级关怀候选；媒体不可建议时回到原有 fatigue/emotion/posture 关怀（LLM 文案），不能跳过关怀。
- Python（``build_wellness_care_summary``）算出 ``should_care`` 与关怀方向；
  LLM 只负责文案，不能把强触发改成 no_op，也不能改成环境提醒。
- Guard 做防刷屏（按 reason 冷却）/ 不在场 / TTS 硬边界的确定性兜底。
"""

from src.adapters.voice.arbitration.session_probe import should_defer_autonomous_speak
from src.agent.decision.autonomous_check_meta import apply_defer_metadata
from src.agent.action.realizer import ActionRealizer
from src.agent.core.models import DecisionResult, Event, Intent
from src.agent.guard.guard import Guard
from src.agent.llm.client import LLMClient
from src.agent.llm.prompt_builder import build_wellness_prompt
from src.agent.llm.reply_validator import normalize_reply, validate_tts_reply
from src.agent.policy_config import LLMRoutingPolicy, WellnessCareCheckPolicy
from src.agent.state.agent_state import AgentState
from src.agent.state.summary_builder import build_wellness_care_summary

# 关怀方向 -> (intent_type, reason_key, 默认文案字段) — 媒体建议由 media_policy 优先选择。
_FOCUS_INTENT = {
    "fatigue": ("suggest_rest", "rest_reminder", "rest_reminder_text"),
    "emotion": ("offer_emotion_care", "emotion_reminder", "emotion_care_text"),
    "posture": ("suggest_rest", "posture_reminder", "posture_reminder_text"),
}

_WELLNESS_REMINDER_REASONS = frozenset(
    {"rest_reminder", "emotion_reminder", "posture_reminder", "media_suggestion"}
)


def _user_context_for_wellness_reply(
    user_context: dict,
    *,
    intent_type: str,
) -> dict:
    """非 suggest_media 时仅标记禁止媒体询问，保留完整个性化候选供 LLM 自然选用。"""

    if intent_type == "suggest_media":
        return user_context

    ctx = dict(user_context)
    hints = ctx.get("memory_usage_hints")
    if isinstance(hints, dict):
        hints = dict(hints)
        hints["media_ask_forbidden"] = True
        ctx["memory_usage_hints"] = hints
    return ctx


def _recent_reminder_texts(state: AgentState, *, limit: int = 5) -> list[str]:
    records = list(getattr(state.runtime_history, "reminder_records", []) or [])
    texts: list[str] = []
    for item in reversed(records):
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "")
        text = str(item.get("text") or "").strip()
        if not text and reason not in _WELLNESS_REMINDER_REASONS:
            continue
        if text:
            texts.append(text[:80])
        if len(texts) >= limit:
            break
    return texts


def _build_trigger_summary(summary: dict, focus: str) -> str:
    if focus == "posture":
        block = summary.get("posture") if isinstance(summary.get("posture"), dict) else {}
        posture = str(block.get("current_posture") or "unknown")
        sustained = block.get("sustained_bad_posture_sec")
        return f"体态偏不良（{posture}，持续约 {sustained or 0}s）"
    if focus == "fatigue":
        block = summary.get("fatigue") if isinstance(summary.get("fatigue"), dict) else {}
        level = str(block.get("current_level") or "unknown")
        return f"疲劳程度 {level}"
    if focus == "emotion":
        block = summary.get("emotion") if isinstance(summary.get("emotion"), dict) else {}
        emotion = str(
            block.get("dominant_negative_emotion") or block.get("current_emotion") or "unknown"
        )
        return f"情绪偏负面（{emotion}）"
    return str(summary.get("care_reason") or focus)


def _build_wellness_reply_context(
    *,
    summary: dict,
    focus: str,
    user_context: dict,
    state: AgentState,
) -> dict:
    hints = user_context.get("memory_usage_hints")
    hints = hints if isinstance(hints, dict) else {}
    candidates = hints.get("personalization_candidates")
    if not isinstance(candidates, list):
        candidates = hints.get("suggestion_candidates") if isinstance(hints.get("suggestion_candidates"), list) else []
    return {
        "trigger_focus": focus,
        "trigger_summary": _build_trigger_summary(summary, focus),
        "personalization_candidates": candidates,
        "recent_reminder_texts": _recent_reminder_texts(state),
        "recent_personalization_used": list(hints.get("recently_used_angles") or []),
        "memory_usage_instruction": str(hints.get("memory_usage_instruction") or "").strip(),
    }


class WellnessCareHandler:
    def __init__(
        self,
        *,
        realizer: ActionRealizer | None = None,
        guard: Guard | None = None,
        policy: LLMRoutingPolicy | None = None,
        check_policy: WellnessCareCheckPolicy | None = None,
        media_controller: object | None = None,
    ) -> None:
        self.realizer = realizer or ActionRealizer()
        self.guard = guard or Guard()
        self.policy = policy or LLMRoutingPolicy()
        self.check_policy = check_policy or WellnessCareCheckPolicy()
        self.media_controller = media_controller

    def decide(
        self,
        *,
        state: AgentState,
        event: Event,
        llm_client: LLMClient,
        user_context: dict[str, object] | None = None,
    ) -> DecisionResult:
        ctx = user_context or {}
        memories = ctx.get("memories") if isinstance(ctx, dict) else None
        summary = build_wellness_care_summary(
            state,
            memories=memories if isinstance(memories, dict) else None,
            policy=self.check_policy,
            check_time=event.timestamp,
        )
        log = self._base_log(summary)

        if state.user.presence != self.check_policy.require_presence:
            return self._no_op(log, "user_away", "user not present", summary)

        if not summary["should_care"]:
            return self._no_op(log, "no_trigger", summary["care_reason"], summary)

        if should_defer_autonomous_speak(dialogue_state=state.interaction.dialogue_state):
            apply_defer_metadata(
                log,
                outcome="voice_session_deferred",
                defer_reason="voice_session_active",
                trigger="wellness_care_check",
            )
            return DecisionResult(
                intents=[Intent("no_op", "voice session active; wellness deferred without speak")],
                source="wellness_care",
                reason="voice_session_deferred: voice session active; wellness deferred without speak",
                log_fields=log,
            )

        focus = summary["recommended_care_focus"]
        intent_type, reason_key, _default_attr = self._choose_care_strategy(
            focus, state, ctx, event.timestamp, log
        )
        log["selected_intent"] = intent_type

        reply_ctx = _user_context_for_wellness_reply(ctx, intent_type=intent_type)
        wellness_reply_context = _build_wellness_reply_context(
            summary=summary,
            focus=focus,
            user_context=reply_ctx,
            state=state,
        )
        reply, used_llm = self._generate_wellness_reply(
            llm_client=llm_client,
            summary=summary,
            focus=focus,
            intent_type=intent_type,
            user_context=reply_ctx,
            wellness_reply_context=wellness_reply_context,
            log=log,
            media_type=log.get("media_type"),
            media_category=log.get("media_category"),
            media_ask_allowed=intent_type == "suggest_media",
        )

        if not reply.strip() or log.get("llm_failed"):
            log["invalid_llm_reply"] = True
            log["fallback_suppressed"] = True
            return self._no_op(
                log,
                "llm_failed_no_hardcoded_fallback",
                "llm failed or empty reply; no hardcoded fallback",
                summary,
            )

        valid, invalid_reason = validate_tts_reply(reply)
        if not valid:
            log["llm_failed"] = True
            log["invalid_llm_reply"] = True
            log["invalid_llm_reply_reason"] = invalid_reason
            log["fallback_suppressed"] = True
            return self._no_op(
                log,
                "llm_failed_no_hardcoded_fallback",
                f"invalid llm reply ({invalid_reason}); no hardcoded fallback",
                summary,
            )

        intent = self._build_wellness_intent(
            intent_type, reason_key, focus, log.get("media_type"), log.get("media_category")
        )
        allowed, findings = self.guard.filter([intent], state=state, timestamp=event.timestamp)
        if not allowed:
            guard_reason = next((f.reason for f in findings if not f.allowed), "guard blocked")
            if "cooldown" in guard_reason:
                log["cooldown_result"] = "blocked"
                outcome = "cooldown"
            else:
                log["guard_result"] = "blocked"
                outcome = "guard_blocked"
            log["final_action_reason"] = outcome
            return DecisionResult(
                intents=[Intent("no_op", guard_reason)],
                source="wellness_care",
                reason=f"{outcome}: {guard_reason}",
                used_llm=used_llm,
                reply_text=reply,
                log_fields=log,
            )

        log["cooldown_result"] = "pass"
        log["guard_result"] = "pass"
        log["final_action_reason"] = reason_key
        if intent_type == "suggest_media":
            mc = self.media_controller
            if mc is not None:
                media_type = log.get("media_type")
                category = log.get("media_category")
                if media_type and category:
                    mc.record_suggestion(
                        media_type=str(media_type),
                        category=str(category),
                        timestamp=event.timestamp,
                    )
        actions = self.realizer.realize(allowed, reply_text=reply)
        return DecisionResult(
            intents=allowed,
            actions=actions,
            used_llm=used_llm,
            source="wellness_care",
            reason=f"wellness_care:{focus}",
            reply_text=reply,
            log_fields=log,
        )

    def _generate_wellness_reply(
        self,
        *,
        llm_client: LLMClient,
        summary: dict,
        focus: str,
        intent_type: str,
        user_context: dict,
        wellness_reply_context: dict,
        log: dict,
        media_type: str | None = None,
        media_category: str | None = None,
        media_ask_allowed: bool = True,
    ) -> tuple[str, bool]:
        media_prompt = None
        if intent_type == "suggest_media" and media_type and media_category:
            media_prompt = {
                "media_type": media_type,
                "media_category": media_category,
                "suggestion_role": "ask_only",
            }
        reply = ""
        used_llm = False
        try:
            prompt = build_wellness_prompt(
                wellness_summary=summary,
                selected_intent=intent_type,
                care_focus=focus,
                user_context=user_context,
                wellness_reply_context=wellness_reply_context,
                media_suggestion=media_prompt,
                media_ask_allowed=media_ask_allowed,
            )
            data = llm_client.complete_json(
                self.policy.wellness_prompt, prompt, temperature=self.policy.reply_temperature
            )
            used_llm = True
            reply = normalize_reply(data.get("reply", ""))
        except Exception as exc:  # noqa: BLE001
            used_llm = True
            log["llm_failed"] = True
            log["llm_note"] = f"llm_failed:{exc}"

        if not reply.strip():
            log["llm_failed"] = True

        candidates = wellness_reply_context.get("personalization_candidates")
        if not isinstance(candidates, list):
            candidates = []
        log["trigger_focus"] = focus
        log["personalization_candidates_count"] = len(candidates)
        log["personalization_candidates_preview"] = [
            str(c.get("label") or "")[:40] for c in candidates[:5] if isinstance(c, dict)
        ]
        log["recent_reminder_texts"] = list(wellness_reply_context.get("recent_reminder_texts") or [])
        log["llm_reply"] = reply
        return reply, used_llm

    @staticmethod
    def _build_wellness_intent(
        intent_type: str,
        reason_key: str,
        focus: str,
        media_type: str | None,
        media_category: str | None,
    ) -> Intent:
        intent = Intent(intent_type, f"wellness_care:{focus}", payload={"reason": reason_key})
        if intent_type == "suggest_media" and media_type and media_category:
            intent.payload["media_type"] = media_type
            intent.payload["category"] = media_category
        return intent

    def _choose_care_strategy(
        self,
        focus: str,
        state: AgentState,
        user_context: dict,
        timestamp: int,
        log: dict,
    ) -> tuple[str, str, str]:
        mc = self.media_controller
        if mc is not None:
            mc.build_selection_context(
                agent_state=state, user_context=user_context, care_focus=focus
            )
            cd = state.cooldown
            choice = mc.policy.try_media_suggestion(
                care_focus=focus,
                media_suggestion_ever_asked=bool(cd.media_suggestion_ever_asked),
                wellness_cares_since_media_ask=int(cd.wellness_cares_since_media_ask),
                library=mc.library,
            )
            if choice is not None and choice.strategy == "media_suggestion":
                log["care_strategy"] = choice.strategy
                log["media_type"] = choice.media_type
                log["media_category"] = choice.category
                return (
                    choice.intent_type,
                    choice.reason_key,
                    choice.default_text_attr,
                )
            check = mc.policy.can_suggest_media(
                media_suggestion_ever_asked=bool(cd.media_suggestion_ever_asked),
                wellness_cares_since_media_ask=int(cd.wellness_cares_since_media_ask),
            )
            log["care_strategy"] = f"wellness_{focus}"
            log["media_unavailable"] = True
            log["media_unavailable_reason"] = check.cooldown_reason or "unavailable"
            log["wellness_cares_since_media_ask"] = int(cd.wellness_cares_since_media_ask)

        return _FOCUS_INTENT.get(focus, ("suggest_rest", "rest_reminder", "rest_reminder_text"))

    @staticmethod
    def _base_log(summary: dict) -> dict:
        fatigue = summary.get("fatigue", {})
        emotion = summary.get("emotion", {})
        posture = summary.get("posture", {})
        return {
            "fatigue_level": fatigue.get("current_level"),
            "fatigue_confidence": fatigue.get("current_confidence"),
            "sustained_high_sec": fatigue.get("sustained_high_sec"),
            "high_ratio_recent_window": fatigue.get("high_ratio_recent_window"),
            "emotion": emotion.get("current_emotion"),
            "negative_ratio_recent_window": emotion.get("negative_ratio_recent_window"),
            "negative_streak_sec": emotion.get("negative_streak_sec"),
            "posture": posture.get("current_posture"),
            "care_triggers": [c.get("type") for c in summary.get("care_triggers", [])],
            "recommended_care_focus": summary.get("recommended_care_focus"),
            "should_care": summary.get("should_care"),
            "selected_intent": None,
            "cooldown_result": "pass",
            "guard_result": "pass",
            "final_action_reason": None,
        }

    def _no_op(self, log: dict, outcome: str, detail: str, summary: dict) -> DecisionResult:
        log["final_action_reason"] = outcome
        return DecisionResult(
            intents=[Intent("no_op", detail)],
            source="wellness_care",
            reason=f"{outcome}: {detail}",
            log_fields=log,
        )
