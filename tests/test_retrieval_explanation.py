from __future__ import annotations

import copy

from src.agent.user.personal_context import PersonalContext


def test_retrieval_explanation_contains_score_breakdown() -> None:
    context = PersonalContext(
        user_id="u1",
        behavior_preferences=(
            {
                "source": "LongTermMemory",
                "memory_type": "behavior_preference",
                "content": "User prefers gentle reminders.",
                "effective_confidence": 0.8,
                "evidence_count": 2,
                "priority_score": 18.0,
                "tags": ["behavior_preference", "reminder_style"],
            },
        ),
    )

    explained = context.retrieve_relevant_with_scores(
        event_type="user_text_input",
        text="gentle reminder_style",
        limit=1,
    )

    breakdown = explained[0]["score_breakdown"]
    assert set(breakdown) == {
        "source_weight",
        "event_type_weight",
        "priority_score",
        "confidence_bonus",
        "evidence_bonus",
        "conflict_penalty",
        "content_term_bonus",
        "tag_term_bonus",
        "final_score",
    }
    assert breakdown["final_score"] == sum(
        value for key, value in breakdown.items() if key != "final_score"
    )


def test_retrieve_relevant_preserves_existing_order() -> None:
    context = _context_with_mixed_items()

    original = context.retrieve_relevant(event_type="user_text_input", text="gentle reminders", limit=5)
    explained = context.retrieve_relevant_with_scores(
        event_type="user_text_input",
        text="gentle reminders",
        limit=5,
    )

    assert [item["content"] for item in explained] == [item["content"] for item in original]


def test_conflict_penalty_visible_in_breakdown() -> None:
    context = PersonalContext(
        user_id="u1",
        uncertain_memories=(
            {
                "source": "LongTermMemory",
                "memory_type": "behavior_preference",
                "content": "User prefers loud reminders.",
                "effective_confidence": 0.9,
                "evidence_count": 2,
                "priority_score": 20.0,
                "conflict_with": "UserProfile:reminder_style",
                "tags": ["behavior_preference", "reminder_style"],
            },
        ),
    )

    explained = context.retrieve_relevant_with_scores(
        event_type="user_text_input",
        text="loud reminder_style",
        limit=1,
    )

    assert explained[0]["score_breakdown"]["conflict_penalty"] < 0


def test_profile_item_source_weight_visible() -> None:
    context = PersonalContext(
        user_id="u1",
        profile_items=(
            {
                "source": "UserProfile",
                "item_type": "explicit_user_preference",
                "content": "reminder_style: gentle",
                "priority_score": 40.0,
                "tags": ["reminder_style", "explicit_user_preference"],
            },
        ),
    )

    explained = context.retrieve_relevant_with_scores(
        event_type="user_text_input",
        text="reminder_style",
        limit=1,
    )

    assert explained[0]["score_breakdown"]["source_weight"] == 100.0


def test_retrieval_explanation_does_not_mutate_items() -> None:
    item = {
        "source": "LongTermMemory",
        "memory_type": "behavior_preference",
        "content": "User prefers gentle reminders.",
        "effective_confidence": 0.8,
        "evidence_count": 1,
        "priority_score": 18.0,
        "metadata": {"preference_key": "reminder_style"},
    }
    original = copy.deepcopy(item)
    context = PersonalContext(user_id="u1", behavior_preferences=(item,))

    explained = context.retrieve_relevant_with_scores(
        event_type="user_text_input",
        text="gentle",
        limit=1,
    )
    explained[0]["metadata"]["preference_key"] = "changed"

    assert item == original
    assert "score_breakdown" not in item


def _context_with_mixed_items() -> PersonalContext:
    return PersonalContext(
        user_id="u1",
        profile_items=(
            {
                "source": "UserProfile",
                "item_type": "explicit_user_preference",
                "content": "reminder_style: gentle",
                "priority_score": 40.0,
                "tags": ["reminder_style", "explicit_user_preference"],
            },
        ),
        behavior_preferences=(
            {
                "source": "LongTermMemory",
                "memory_type": "behavior_preference",
                "content": "User prefers gentle reminders.",
                "effective_confidence": 0.9,
                "evidence_count": 3,
                "priority_score": 20.0,
                "tags": ["behavior_preference", "reminder_style"],
            },
        ),
        runtime_items=(
            {
                "source": "RuntimeHistory",
                "item_type": "recent_message",
                "content": "user: please be gentle",
                "priority_score": 4.0,
                "tags": ["recent_message", "user"],
            },
        ),
    )
