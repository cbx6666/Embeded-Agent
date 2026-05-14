from __future__ import annotations

"""Cognitive retrieval policy values for PersonalContext ranking.

These numbers express the agent's retrieval worldview: source authority,
event-specific relevance, confidence/evidence influence, and conflict or term
adjustments. They are policy, not protocol.
"""

from dataclasses import dataclass, field


def _default_source_weights() -> dict[str, float]:
    return {
        "UserProfile": 100.0,
        "LongTermMemory": 50.0,
        "RuntimeHistory": 25.0,
    }


def _default_event_type_weights() -> dict[str, dict[str, float]]:
    dialogue_weights = {
        "explicit_user_preference": 14.0,
        "interaction_style": 12.0,
        "behavior_preference": 10.0,
        "recent_message": 5.0,
    }
    focus_weights = {
        "active_constraint": 14.0,
        "behavior_pattern": 10.0,
        "behavior_preference": 8.0,
        "recent_action": 4.0,
    }
    user_state_weights = {
        "behavior_pattern": 12.0,
        "active_constraint": 10.0,
        "recent_event": 5.0,
    }
    system_weights = {
        "active_constraint": 12.0,
        "behavior_pattern": 8.0,
        "recent_action": 5.0,
    }
    return {
        "user_text_input": dict(dialogue_weights),
        "speech_recognized": dict(dialogue_weights),
        "focus_start_requested": dict(focus_weights),
        "focus_stop_requested": dict(focus_weights),
        "timer_ticked": dict(focus_weights),
        "timer_finished": dict(focus_weights),
        "user_presence_updated": dict(user_state_weights),
        "user_attention_updated": dict(user_state_weights),
        "user_emotion_updated": dict(user_state_weights),
        "user_fatigue_updated": dict(user_state_weights),
        "system_triggered": dict(system_weights),
    }


@dataclass(frozen=True)
class RetrievalPolicyConfig:
    """Policy knobs used by PersonalContext.retrieve_relevant ranking."""

    source_weights: dict[str, float] = field(default_factory=_default_source_weights)
    event_type_weights: dict[str, dict[str, float]] = field(default_factory=_default_event_type_weights)
    confidence_weight: float = 10.0
    evidence_weight: float = 0.5
    conflict_penalty: float = 30.0
    content_term_weight: float = 3.0
    tag_term_weight: float = 2.0
    max_evidence_bonus: float = 4.0
