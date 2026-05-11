from __future__ import annotations

"""长期记忆生命周期策略。

MemoryPolicy 负责判断画像候选是否允许写入长期 profile，以及已有画像是否需要
置信度衰减或废弃。它不修改 Action、不修改 Planner、不修改 AgentState。
"""

from dataclasses import replace

from src.agent.memory.memory_candidate import MemoryCandidate, MemoryPolicyDecision
from src.agent.state.user_profile_state import UserProfileInsight


class MemoryPolicy:
    """长期画像写入与生命周期治理规则。"""

    def __init__(
        self,
        *,
        evidence_threshold: int = 3,
        confidence_threshold: float = 0.6,
        confirmation_confidence_threshold: float = 0.45,
        decay_after_days: float = 30.0,
        decay_rate_per_period: float = 0.92,
        discard_confidence_threshold: float = 0.2,
    ) -> None:
        self.evidence_threshold = evidence_threshold
        self.confidence_threshold = confidence_threshold
        self.confirmation_confidence_threshold = confirmation_confidence_threshold
        self.decay_after_days = decay_after_days
        self.decay_rate_per_period = decay_rate_per_period
        self.discard_confidence_threshold = discard_confidence_threshold

    def evaluate(
        self,
        candidate: MemoryCandidate,
        existing_insights: list[UserProfileInsight],
    ) -> MemoryPolicyDecision:
        """判断一个画像候选是否可以写入长期 profile。"""
        if candidate.evidence_count < self.evidence_threshold:
            return MemoryPolicyDecision(
                candidate=candidate,
                allow_write=False,
                reason=f"证据数不足：{candidate.evidence_count} < {self.evidence_threshold}",
            )

        if candidate.confidence < self.confirmation_confidence_threshold:
            return MemoryPolicyDecision(
                candidate=candidate,
                allow_write=False,
                reason=f"置信度过低：{candidate.confidence:.2f}",
            )

        if candidate.confidence < self.confidence_threshold:
            return MemoryPolicyDecision(
                candidate=candidate,
                allow_write=False,
                requires_confirmation=True,
                reason=f"置信度需要确认：{candidate.confidence:.2f}",
            )

        contradicted = self._contradicted_contents(candidate, existing_insights)
        if contradicted and not self._candidate_is_stronger(candidate, existing_insights):
            return MemoryPolicyDecision(
                candidate=candidate,
                allow_write=False,
                reason="存在更强的矛盾画像，暂不覆盖",
                contradicted_contents=contradicted,
            )

        return MemoryPolicyDecision(
            candidate=candidate,
            allow_write=True,
            reason=candidate.explanation,
            contradicted_contents=contradicted,
        )

    def decay_insights(
        self,
        insights: list[UserProfileInsight],
        *,
        now: float,
    ) -> tuple[list[UserProfileInsight], bool]:
        """对长期未更新的 insight 做置信度衰减，并丢弃过低置信度画像。"""
        changed = False
        kept: list[UserProfileInsight] = []
        seconds_per_period = self.decay_after_days * 24 * 3600

        for insight in insights:
            updated_at = now if insight.updated_at is None else float(insight.updated_at)
            age_sec = max(0.0, now - updated_at)
            if seconds_per_period <= 0 or age_sec < seconds_per_period:
                kept.append(insight)
                continue

            periods = int(age_sec // seconds_per_period)
            decayed_confidence = insight.confidence * (self.decay_rate_per_period ** periods)
            changed = True
            if decayed_confidence < self.discard_confidence_threshold:
                continue
            kept.append(replace(insight, confidence=decayed_confidence))

        return kept, changed

    def _contradicted_contents(
        self,
        candidate: MemoryCandidate,
        existing_insights: list[UserProfileInsight],
    ) -> list[str]:
        """找出同类型但内容不同的画像，作为矛盾候选。"""
        if candidate.contradiction_group is None:
            return []
        return [
            insight.content
            for insight in existing_insights
            if insight.insight_type == candidate.insight_type
            and insight.content != candidate.content
        ]

    def _candidate_is_stronger(
        self,
        candidate: MemoryCandidate,
        existing_insights: list[UserProfileInsight],
    ) -> bool:
        """判断新候选是否足以替代已有矛盾画像。"""
        strongest = 0.0
        for insight in existing_insights:
            if insight.insight_type == candidate.insight_type and insight.content != candidate.content:
                strongest = max(strongest, insight.confidence)
        return candidate.confidence >= strongest + 0.05
