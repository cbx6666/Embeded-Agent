from __future__ import annotations

"""P1 自主检查的确定性前置门控。

Scheduler 只负责按系统时间产生检查事件；本模块再结合当前状态、rolling summary
和 cooldown 判断是否值得进入昂贵的 LLM 决策。手工注入 P1 事件同样必须经过这里，
因此不会形成绕过调度频率的通用 LLM 入口。
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from src.agent.config.policy_config import (
    AutonomousCheckPolicyConfig,
    GuardPolicyConfig,
)
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.event import Event
from src.agent.state import AgentState


AutonomousCheckMode = Literal["skip", "rule", "llm"]


@dataclass(frozen=True)
class AutonomousCheckDecision:
    """一次 P1 前置判断结果。"""

    mode: AutonomousCheckMode
    reason: str
    trigger: str
    evidence: dict[str, Any] = field(default_factory=dict)
    plan: IntentPlan | None = None

    @property
    def should_enter_decision(self) -> bool:
        return self.mode in {"rule", "llm"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "trigger": self.trigger,
            "evidence": dict(self.evidence),
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


class AutonomousCheckPolicy:
    """决定 P1 检查应跳过、走规则还是进入 LLM。"""

    def __init__(
        self,
        *,
        policy_config: AutonomousCheckPolicyConfig | None = None,
        guard_policy_config: GuardPolicyConfig | None = None,
    ) -> None:
        self.config = policy_config or AutonomousCheckPolicyConfig()
        self.guard_config = guard_policy_config or GuardPolicyConfig()

    def evaluate(self, *, event: Event, state: AgentState) -> AutonomousCheckDecision:
        """只使用结构化事实执行低成本 gate，不调用 LLM、不修改状态。"""

        trigger = str(event.payload.get("trigger", "")).strip()
        source = str(event.payload.get("source", "")).strip()
        if event.type != "system_triggered":
            return self._skip(trigger, "not_system_triggered")
        if trigger not in self.config.check_cooldown_sec:
            return self._skip(trigger, "unsupported_autonomous_trigger")
        if source != self.config.trusted_source:
            return self._skip(trigger, "untrusted_autonomous_source")

        cooldown_reason = self._check_cooldown_reason(trigger, event.timestamp, state)
        if cooldown_reason:
            return self._skip(trigger, cooldown_reason)

        if trigger == "focus_health_check":
            return self._evaluate_focus_health(event, state)
        if trigger == "environment_check":
            return self._evaluate_environment(event, state)
        if trigger == "user_idle_check":
            return self._evaluate_user_idle(event, state)
        if trigger == "periodic_check":
            return self._evaluate_periodic(event, state)
        return self._skip(trigger, "unsupported_autonomous_trigger")

    def _evaluate_focus_health(
        self,
        event: Event,
        state: AgentState,
    ) -> AutonomousCheckDecision:
        trigger = "focus_health_check"
        if not state.focus.active:
            return self._skip(trigger, "focus_not_active")
        if str(state.user.presence) not in self.config.present_values:
            return self._skip(trigger, "user_not_present")
        if state.focus.elapsed_sec < self.config.focus_min_elapsed_sec:
            return self._skip(trigger, "focus_elapsed_too_short")

        abnormal = self._sustained_abnormal_signals(
            state,
            signals=("fatigue", "attention", "posture"),
        )
        if not abnormal:
            return self._skip(trigger, "no_sustained_focus_abnormality")

        available = self._signals_outside_reminder_cooldown(
            abnormal,
            now_ts=event.timestamp,
            state=state,
        )
        if not available:
            return self._skip(trigger, "focus_reminder_cooldown_active")
        if len(available) == 1:
            signal = next(iter(available))
            intent_type = "remind_distraction" if signal == "attention" else "suggest_rest"
            return self._rule(
                trigger=trigger,
                reason="single_sustained_focus_abnormality",
                intent_type=intent_type,
                intent_reason=f"sustained {signal} abnormality during focus",
                evidence={"abnormal_signals": available},
            )
        return AutonomousCheckDecision(
            mode="llm",
            reason="multiple_sustained_focus_abnormalities",
            trigger=trigger,
            evidence={"abnormal_signals": available},
        )

    def _evaluate_environment(
        self,
        event: Event,
        state: AgentState,
    ) -> AutonomousCheckDecision:
        trigger = "environment_check"
        if str(state.user.presence) not in self.config.present_values:
            return self._skip(trigger, "user_not_present")
        if not self._is_working(state):
            return self._skip(trigger, "user_not_in_working_context")

        abnormal = self._sustained_abnormal_signals(
            state,
            signals=("light", "temperature", "humidity", "noise"),
        )
        if not abnormal:
            return self._skip(trigger, "no_sustained_environment_abnormality")
        available = self._signals_outside_reminder_cooldown(
            abnormal,
            now_ts=event.timestamp,
            state=state,
        )
        if not available:
            return self._skip(trigger, "environment_reminder_cooldown_active")
        return self._rule(
            trigger=trigger,
            reason="sustained_environment_abnormality",
            intent_type="adjust_environment_feedback",
            intent_reason="sustained environment abnormality",
            evidence={"abnormal_signals": available},
        )

    def _evaluate_user_idle(
        self,
        event: Event,
        state: AgentState,
    ) -> AutonomousCheckDecision:
        trigger = "user_idle_check"
        if str(state.user.presence) not in self.config.present_values:
            return self._skip(trigger, "user_not_present")
        if not self._is_working(state):
            return self._skip(trigger, "no_active_work_or_focus")
        last_user_time = state.interaction.last_user_time
        if last_user_time is None:
            return self._skip(trigger, "idle_baseline_missing")
        idle_duration = max(0, int(event.timestamp) - int(last_user_time))
        if idle_duration < self.config.idle_min_duration_sec:
            return self._skip(trigger, "idle_duration_too_short")
        if self._reminder_cooldown_active("idle", event.timestamp, state):
            return self._skip(trigger, "idle_reminder_cooldown_active")
        return AutonomousCheckDecision(
            mode="llm",
            reason="user_idle_during_active_work",
            trigger=trigger,
            evidence={"idle_duration_sec": idle_duration},
        )

    def _evaluate_periodic(
        self,
        event: Event,
        state: AgentState,
    ) -> AutonomousCheckDecision:
        trigger = "periodic_check"
        if not self.config.periodic_check_enabled:
            return self._skip(trigger, "periodic_check_disabled")
        if str(state.user.presence) not in self.config.present_values:
            return self._skip(trigger, "user_not_present")

        # periodic_check 不能成为无条件 LLM 入口；即使显式开启，也必须先发现持续异常。
        abnormal = self._sustained_abnormal_signals(
            state,
            signals=("fatigue", "attention", "posture", "light", "temperature", "humidity", "noise"),
        )
        if not abnormal:
            return self._skip(trigger, "periodic_check_has_no_actionable_signal")
        available = self._signals_outside_reminder_cooldown(
            abnormal,
            now_ts=event.timestamp,
            state=state,
        )
        if not available:
            return self._skip(trigger, "periodic_reminder_cooldown_active")
        return AutonomousCheckDecision(
            mode="llm",
            reason="periodic_check_found_sustained_abnormality",
            trigger=trigger,
            evidence={"abnormal_signals": available},
        )

    def _sustained_abnormal_signals(
        self,
        state: AgentState,
        *,
        signals: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        findings: dict[str, dict[str, Any]] = {}
        trends = state.runtime_history.signal_trends
        for signal in signals:
            trend = trends.get(signal)
            if not isinstance(trend, dict):
                continue
            abnormal_values = self.config.abnormal_values_by_signal.get(signal, frozenset())
            recent = [
                item
                for item in trend.get("recent_values", [])
                if isinstance(item, dict)
            ]
            if len(recent) < self.config.sustained_min_samples:
                continue
            abnormal_count = sum(
                1
                for item in recent
                if str(item.get("value", "")).strip().lower() in abnormal_values
            )
            ratio = abnormal_count / len(recent)
            current = str(trend.get("current", "")).strip().lower()
            consecutive = int(trend.get("consecutive_same_count", 0))
            confidence = trend.get("confidence_summary", {})
            average_confidence = (
                confidence.get("average")
                if isinstance(confidence, dict)
                else None
            )
            if current not in abnormal_values:
                continue
            if ratio < self.config.sustained_min_ratio:
                continue
            if consecutive < self.config.sustained_min_consecutive:
                continue
            if (
                average_confidence is not None
                and float(average_confidence) < self.config.minimum_average_confidence
            ):
                continue
            findings[signal] = {
                "current": current,
                "sample_count": len(recent),
                "abnormal_ratio": round(ratio, 4),
                "consecutive_same_count": consecutive,
                "average_confidence": average_confidence,
            }
        return findings

    def _signals_outside_reminder_cooldown(
        self,
        findings: dict[str, dict[str, Any]],
        *,
        now_ts: int,
        state: AgentState,
    ) -> dict[str, dict[str, Any]]:
        return {
            signal: evidence
            for signal, evidence in findings.items()
            if not self._reminder_cooldown_active(signal, now_ts, state)
        }

    def _reminder_cooldown_active(
        self,
        signal: str,
        now_ts: int,
        state: AgentState,
    ) -> bool:
        reason = self.config.reminder_reasons_by_signal.get(signal)
        if not reason:
            return False
        last_ts = state.cooldown.reminder_last_ts.get(reason)
        if last_ts is None:
            return False
        return int(now_ts) - int(last_ts) < self.guard_config.reminder_cooldown_sec

    def _check_cooldown_reason(
        self,
        trigger: str,
        now_ts: int,
        state: AgentState,
    ) -> str | None:
        last_ts = state.cooldown.autonomous_check_last_ts.get(trigger)
        if last_ts is None:
            return None
        cooldown_sec = int(self.config.check_cooldown_sec.get(trigger, 0))
        if int(now_ts) - int(last_ts) < cooldown_sec:
            return "autonomous_check_cooldown_active"
        return None

    def _is_working(self, state: AgentState) -> bool:
        return bool(
            state.focus.active
            or str(state.interaction.mode) in self.config.working_modes
            or str(state.user.current_activity) in self.config.working_activities
        )

    @staticmethod
    def _skip(trigger: str, reason: str) -> AutonomousCheckDecision:
        return AutonomousCheckDecision(mode="skip", reason=reason, trigger=trigger)

    @staticmethod
    def _rule(
        *,
        trigger: str,
        reason: str,
        intent_type: str,
        intent_reason: str,
        evidence: dict[str, Any],
    ) -> AutonomousCheckDecision:
        plan = IntentPlan(
            intents=[
                AgentIntent(
                    type=intent_type,
                    priority=60,
                    reason=intent_reason,
                    payload={},
                    requires_llm=False,
                )
            ],
            reasoning=f"AutonomousCheckPolicy mapped {trigger} to {intent_type}.",
            risk_level="low",
            interrupt_user=True,
        )
        return AutonomousCheckDecision(
            mode="rule",
            reason=reason,
            trigger=trigger,
            evidence=evidence,
            plan=plan,
        )
