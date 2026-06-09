from __future__ import annotations

"""确定性 Guard。

只负责系统级硬边界，不做业务决策、不理解自然语言：

- 防刷屏：同一提醒在冷却时间内不重复执行。
- TTS / 语音正在进行时不打断（dialogue_state == "speaking"）。
- 用户不在场时不提醒。

输入是决策处理器产出的意图列表 + 当前状态，输出过滤后的意图与可解释 findings。
"""

from dataclasses import dataclass

from src.agent.core.models import Intent
from src.agent.policy_config import GuardPolicy
from src.agent.state.agent_state import AgentState


@dataclass(frozen=True)
class GuardFinding:
    intent_type: str
    allowed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"intent_type": self.intent_type, "allowed": self.allowed, "reason": self.reason}


class Guard:
    def __init__(self, policy: GuardPolicy | None = None) -> None:
        self.policy = policy or GuardPolicy()

    def filter(
        self,
        intents: list[Intent],
        *,
        state: AgentState,
        timestamp: int,
    ) -> tuple[list[Intent], list[GuardFinding]]:
        allowed: list[Intent] = []
        findings: list[GuardFinding] = []
        for intent in intents:
            reason = self._block_reason(intent, state, timestamp)
            if reason:
                findings.append(GuardFinding(intent.type, False, reason))
                continue
            allowed.append(intent)
            findings.append(GuardFinding(intent.type, True, "allowed"))
        return allowed, findings

    def _block_reason(self, intent: Intent, state: AgentState, timestamp: int) -> str | None:
        if intent.type not in self.policy.interruptive_intents:
            return None

        if state.user.presence == self.policy.block_interruptive_when_presence:
            return "user is away; interruptive reminder blocked"

        if (
            self.policy.block_interruptive_when_speaking
            and intent.type not in self.policy.speaking_exempt_intents
            and state.interaction.dialogue_state == "speaking"
        ):
            return "tts in progress; reminder must not interrupt"

        if intent.type in self.policy.cooldown_exempt_intents:
            return None

        reason_key = str(intent.payload.get("reason") or self.policy.cooldown_reasons.get(intent.type, ""))
        if reason_key:
            cooldown_sec = self._cooldown_sec_for(reason_key)
            last_ts = state.cooldown.reminder_last_ts.get(reason_key)
            if last_ts is not None:
                try:
                    elapsed = int(timestamp) - int(last_ts)
                except (TypeError, ValueError):
                    elapsed = cooldown_sec
                if elapsed < 0:
                    elapsed = cooldown_sec
                if elapsed < cooldown_sec:
                    return f"cooldown active for {reason_key} ({elapsed}s < {cooldown_sec}s)"
        return None

    def _cooldown_sec_for(self, reason_key: str) -> int:
        """按 reason 取冷却秒数；未配置时回退到默认 reminder_cooldown_sec。"""

        return int(self.policy.cooldown_by_reason.get(reason_key, self.policy.reminder_cooldown_sec))
