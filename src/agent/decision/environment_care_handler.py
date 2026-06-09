from __future__ import annotations

"""environment_care_check 环境关怀处理器（每 60s，优先级 3）。

职责边界：
- 只负责环境关怀（光照 / 温度 / 湿度 / 噪声），由 LLM 判断是否需要一句关怀提醒。
- 这条链路可以 no_op。
- 只能产 adjust_environment_feedback（reason=environment_warning），
  绝不能生成 rest_reminder / emotion_reminder / posture_reminder。
- 撞上用户语音会话或 TTS 时延后到下一轮重新判断，不静默误报已播。
"""

from src.adapters.voice.arbitration.session_probe import should_defer_autonomous_speak
from src.agent.decision.autonomous_check_meta import apply_defer_metadata
from src.agent.action.realizer import ActionRealizer
from src.agent.core.models import DecisionResult, Event, Intent
from src.agent.guard.guard import Guard
from src.agent.llm.client import LLMClient
from src.agent.llm.prompt_builder import build_environment_care_prompt
from src.agent.llm.reply_validator import normalize_reply, validate_tts_reply
from src.agent.policy_config import EnvironmentCareCheckPolicy, LLMRoutingPolicy
from src.agent.state.agent_state import AgentState
from src.agent.state.summary_builder import build_environment_care_summary

_ALLOWED_INTENTS = {"no_op", "adjust_environment_feedback"}


class EnvironmentCareHandler:
    def __init__(
        self,
        *,
        realizer: ActionRealizer | None = None,
        guard: Guard | None = None,
        policy: LLMRoutingPolicy | None = None,
        check_policy: EnvironmentCareCheckPolicy | None = None,
    ) -> None:
        self.realizer = realizer or ActionRealizer()
        self.guard = guard or Guard()
        self.policy = policy or LLMRoutingPolicy()
        self.check_policy = check_policy or EnvironmentCareCheckPolicy()

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
        summary = build_environment_care_summary(
            state,
            memories=memories if isinstance(memories, dict) else None,
            policy=self.check_policy,
            check_time=event.timestamp,
        )
        log = {
            "environment_triggers": [t.get("type") for t in summary.get("environment_triggers", [])],
            "should_consider_care": summary.get("should_consider_care"),
            "llm_intent": None,
            "cooldown_result": "pass",
            "guard_result": "pass",
            "final_action_reason": None,
        }

        if state.user.presence != self.check_policy.require_presence:
            return self._no_op(log, "user_away", "user not present")

        # 用户会话保护区 / 放歌期间：延后到下一轮，不调用 LLM。
        # environment defer 不回退 60s 调度周期（见 autonomous_check_meta），仅跳过准入冷却。
        if should_defer_autonomous_speak(dialogue_state=state.interaction.dialogue_state):
            if state.interaction.dialogue_state == "speaking":
                apply_defer_metadata(
                    log,
                    outcome="tts_speaking_deferred",
                    defer_reason="user_reply_speaking",
                    trigger="environment_care_check",
                )
                detail = "tts in progress; deferred"
            else:
                apply_defer_metadata(
                    log,
                    outcome="voice_session_active_deferred",
                    defer_reason="voice_session_active",
                    trigger="environment_care_check",
                )
                detail = "voice session active; deferred"
            return DecisionResult(
                intents=[Intent("no_op", detail)],
                source="environment_care",
                reason=f"{log['final_action_reason']}: {detail}",
                log_fields=log,
            )

        if not self._has_environment_data(state):
            return self._no_op(log, "no_environment_data", "no environment readings")

        if not summary.get("should_consider_care"):
            return self._no_op(log, "no_trigger", summary.get("care_reason", "no abnormal environment"))

        try:
            prompt = build_environment_care_prompt(environment_summary=summary, user_context=ctx)
            data = llm_client.complete_json(
                self.policy.environment_care_prompt, prompt, temperature=self.policy.reply_temperature
            )
        except Exception as exc:  # noqa: BLE001 - LLM 失败不提醒
            return self._no_op(log, "llm_failed", f"llm fallback: {exc}", used_llm=True)

        intent_type = str(data.get("intent", "no_op")).strip()
        if intent_type not in _ALLOWED_INTENTS:
            intent_type = "no_op"
        reply = normalize_reply(data.get("reply", ""))
        log["llm_intent"] = intent_type

        if intent_type == "no_op":
            return self._no_op(log, "llm_no_op", "llm decided no environment care", used_llm=True)

        if not reply:
            log["llm_failed"] = True
            log["invalid_llm_reply"] = True
            log["fallback_suppressed"] = True
            return self._no_op(log, "empty_reply", "llm intent set but reply empty", used_llm=True)

        valid, invalid_reason = validate_tts_reply(reply)
        if not valid:
            log["llm_failed"] = True
            log["invalid_llm_reply"] = True
            log["invalid_llm_reply_reason"] = invalid_reason
            log["fallback_suppressed"] = True
            return self._no_op(
                log,
                "invalid_reply",
                f"llm reply invalid ({invalid_reason})",
                used_llm=True,
            )

        intent = Intent(
            "adjust_environment_feedback",
            "environment_care llm decision",
            payload={"reason": "environment_warning"},
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
                source="environment_care",
                reason=f"{outcome}: {guard_reason}",
                used_llm=True,
                reply_text=reply,
                log_fields=log,
            )

        log["final_action_reason"] = "environment_warning"
        actions = self.realizer.realize(allowed, reply_text=reply)
        return DecisionResult(
            intents=allowed,
            actions=actions,
            used_llm=True,
            source="environment_care",
            reason="environment_care",
            reply_text=reply,
            log_fields=log,
        )

    @staticmethod
    def _has_environment_data(state: AgentState) -> bool:
        env = state.environment
        return any(
            v is not None
            for v in (env.light_lux, env.temperature_c, env.humidity_pct, env.noise_db)
        )

    def _no_op(
        self, log: dict, outcome: str, detail: str, *, used_llm: bool = False
    ) -> DecisionResult:
        log["final_action_reason"] = outcome
        return DecisionResult(
            intents=[Intent("no_op", detail)],
            source="environment_care",
            reason=f"{outcome}: {detail}",
            used_llm=used_llm,
            log_fields=log,
        )
