from __future__ import annotations

"""behavior_distraction_check 玩手机分心处理器（每 20s，调度优先级最高）。

判定流程（全部在 Python，LLM 只写提醒措辞）：
1. 用户语音会话 listening/thinking 中 → 延后，不消耗提醒机会。
2. 汇总最近 30s 行为记录，统计 phone_use 条数、YOLO 检出手机的条数、
   phone_use 在 phone_use+working 中的占比、最近一次 phone_use 距今秒数、置信度。
3. 全部满足才 trigger_candidate=true：
   - 用户在场（presence=present；窗口内持续 phone_use 可容忍 pose 误判 away）
   - phone_use 条数 ≥ 1
   - YOLO 真正框到手机的条数 ≥ 1（姿态/手腕推断 alone 不算）
   - phone_use 占比 ≥ 15%
   - 最近 15s 内仍有 phone_use（还在玩）
   - 最近一条 phone_use 必须带 YOLO 手机框（require_yolo_phone_on_latest）
   - 行为置信度 ≥ 0.5
4. trigger 后 Guard 按 distraction_reminder 冷却 60s；TTS 入队按优先级等待，不丢弃。
"""

from src.adapters.voice.arbitration.session_probe import should_defer_autonomous_speak
from src.agent.decision.autonomous_check_meta import apply_defer_metadata
from src.agent.action.realizer import ActionRealizer
from src.agent.core.models import DecisionResult, Event, Intent
from src.agent.guard.guard import Guard
from src.agent.llm.client import LLMClient
from src.agent.llm.prompt_builder import build_behavior_distraction_prompt
from src.agent.llm.reply_validator import normalize_reply, validate_tts_reply
from src.agent.policy_config import BehaviorDistractionCheckPolicy, LLMRoutingPolicy
from src.agent.state.agent_state import AgentState
from src.agent.state.summary_builder import build_behavior_distraction_summary


class BehaviorDistractionHandler:
    def __init__(
        self,
        *,
        realizer: ActionRealizer | None = None,
        guard: Guard | None = None,
        policy: LLMRoutingPolicy | None = None,
        check_policy: BehaviorDistractionCheckPolicy | None = None,
    ) -> None:
        self.realizer = realizer or ActionRealizer()
        self.guard = guard or Guard()
        self.policy = policy or LLMRoutingPolicy()
        self.check_policy = check_policy or BehaviorDistractionCheckPolicy()

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
        summary = build_behavior_distraction_summary(
            state,
            memories=memories if isinstance(memories, dict) else None,
            policy=self.check_policy,
            check_time=event.timestamp,
        )
        log = {
            "trigger_candidate": bool(summary.get("trigger_candidate")),
            "selected_intent": None,
            "cooldown_result": "pass",
            "guard_result": "pass",
            "final_action_reason": None,
        }

        # 用户语音会话 / 放歌 / 会话后保护期内不走 LLM，下轮周期再判。
        if should_defer_autonomous_speak(dialogue_state=state.interaction.dialogue_state):
            apply_defer_metadata(
                log,
                outcome="voice_session_active_deferred",
                defer_reason="voice_session_active",
                trigger="behavior_distraction_check",
            )
            return DecisionResult(
                intents=[Intent("no_op", "voice session active; reminder deferred")],
                source="behavior_distraction",
                reason="voice session active; reminder deferred",
                log_fields=log,
            )

        if not summary["trigger_candidate"]:
            detail = summary.get("trigger_detail", {}) if isinstance(summary, dict) else {}
            reason = str(detail.get("reason", "no distraction signal"))
            stats = (
                f"phone_use={summary.get('window_phone_use_events')} "
                f"yolo={summary.get('window_yolo_phone_events')} "
                f"ratio={detail.get('phone_use_ratio')} "
                f"last_age={detail.get('last_phone_age_sec')}s"
            )
            log["final_action_reason"] = "no_trigger"
            return DecisionResult(
                intents=[Intent("no_op", "behavior distraction precheck -> no signal")],
                source="behavior_distraction",
                reason=f"{reason} | {stats}",
                log_fields=log,
            )

        prompt = build_behavior_distraction_prompt(
            distraction_summary=summary,
            user_context=ctx,
        )

        # Python 已确认 trigger_candidate：必须提醒，LLM 只负责措辞，不能否决为 no_op。
        reply = ""
        used_llm = False
        try:
            data = llm_client.complete_json(
                self.policy.behavior_distraction_prompt,
                prompt,
                temperature=self.policy.reply_temperature,
            )
            used_llm = True
            reply = normalize_reply(data.get("reply", ""))
        except Exception as exc:  # noqa: BLE001
            used_llm = True
            log["llm_failed"] = True
            log["fallback_suppressed"] = True
            log["final_action_reason"] = "llm_failed_no_hardcoded_fallback"
            return DecisionResult(
                intents=[Intent("no_op", f"llm_failed:{exc}")],
                source="behavior_distraction",
                reason=f"llm_failed_no_hardcoded_fallback:{exc}",
                used_llm=used_llm,
                log_fields=log,
            )

        if not reply:
            log["llm_failed"] = True
            log["invalid_llm_reply"] = True
            log["fallback_suppressed"] = True
            log["final_action_reason"] = "llm_failed_no_hardcoded_fallback"
            return DecisionResult(
                intents=[Intent("no_op", "llm empty reply")],
                source="behavior_distraction",
                reason="llm_failed_no_hardcoded_fallback: empty reply",
                used_llm=used_llm,
                log_fields=log,
            )

        valid, invalid_reason = validate_tts_reply(reply)
        if not valid:
            log["llm_failed"] = True
            log["invalid_llm_reply"] = True
            log["invalid_llm_reply_reason"] = invalid_reason
            log["fallback_suppressed"] = True
            log["final_action_reason"] = "llm_failed_no_hardcoded_fallback"
            return DecisionResult(
                intents=[Intent("no_op", f"invalid llm reply: {invalid_reason}")],
                source="behavior_distraction",
                reason=f"llm_failed_no_hardcoded_fallback: invalid reply ({invalid_reason})",
                used_llm=used_llm,
                log_fields=log,
            )

        llm_note = "trigger confirmed -> remind"
        intent = Intent("remind_distraction", llm_note)
        log["selected_intent"] = "remind_distraction"
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
                source="behavior_distraction",
                reason=f"guard blocked: {guard_reason}",
                used_llm=used_llm,
                reply_text=reply,
                log_fields=log,
            )

        log["final_action_reason"] = "distraction_reminder"
        actions = self.realizer.realize(allowed, reply_text=reply)
        return DecisionResult(
            intents=allowed,
            actions=actions,
            used_llm=used_llm,
            source="behavior_distraction",
            reason=llm_note,
            reply_text=reply,
            log_fields=log,
        )
