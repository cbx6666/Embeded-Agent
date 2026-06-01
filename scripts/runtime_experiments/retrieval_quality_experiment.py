from __future__ import annotations

from pathlib import Path
from typing import Any

from common import build_environment, ev, write_json
from src.agent.memory.memory_candidate import MemoryCandidate


EXPERIMENT_NAME = "retrieval_quality"
USER_ID = "retrieval_user"


def run(output_root: str | Path | None = None) -> Path:
    env = build_environment(EXPERIMENT_NAME, user_id=USER_ID, output_root=output_root)
    try:
        _seed_profile_memory_and_runtime(env)
        cases = _retrieval_cases()
        results = []
        score_breakdown = []
        for case in cases:
            event = ev(case["event_type"], int(case["timestamp"]), text=case["text"], trigger=case.get("trigger", ""))
            personal_context = env.core.personal_context_builder.build(
                user_id=USER_ID,
                state=env.core.state,
                event=event,
            )
            explained = personal_context.retrieve_relevant_with_scores(
                event_type=str(case["event_type"]),
                text=str(case["text"]),
                limit=int(case["limit"]),
            )
            result = _case_result(case, explained)
            results.append(result)
            for item in explained:
                score_breakdown.append(
                    {
                        "case_id": case["id"],
                        "rank": item["retrieval_rank"],
                        "source": item.get("source"),
                        "content": item.get("content"),
                        "score_breakdown": item["score_breakdown"],
                    }
                )

        metrics = _metrics(results)
        output_dir = env.output_dir
        write_json(output_dir / "retrieval_cases.json", cases)
        write_json(output_dir / "retrieval_results.json", results)
        write_json(output_dir / "score_breakdown.json", score_breakdown)
        write_json(output_dir / "metrics.json", metrics)
        (output_dir / "report.md").write_text(
            _render_report(results=results, metrics=metrics),
            encoding="utf-8",
        )
        return output_dir
    finally:
        env.shutdown()


def _seed_profile_memory_and_runtime(env: Any) -> None:
    env.profile_service.update_preference(USER_ID, "reminder_style", "gentle", timestamp=10)
    env.profile_service.update_preference(USER_ID, "speech_style", "concise", timestamp=11)
    env.core.state.current_user_id = USER_ID

    _upsert(
        env,
        memory_type="behavior_preference",
        content="User prefers concise study reminders.",
        confidence=0.82,
        timestamp=20,
        metadata={"preference_key": "speech_style", "preference_value": "concise"},
    )
    _upsert(
        env,
        memory_type="interaction_style",
        content="Use short visual-first guidance while the user is studying calculus.",
        confidence=0.78,
        timestamp=21,
        metadata={"preference_key": "study_guidance", "preference_value": "visual_first"},
    )
    _upsert(
        env,
        memory_type="active_constraint",
        content="Do not speak during silent focus mode unless the user asks.",
        confidence=0.9,
        timestamp=22,
        metadata={"constraint_key": "silent_focus"},
    )
    _upsert(
        env,
        memory_type="behavior_pattern",
        content="User often drifts after 25 minutes of focus.",
        confidence=0.74,
        timestamp=23,
        metadata={"pattern_key": "focus_drift"},
    )
    _upsert(
        env,
        memory_type="behavior_preference",
        content="User prefers loud reminders.",
        confidence=0.88,
        timestamp=24,
        metadata={"preference_key": "reminder_style", "preference_value": "loud"},
    )

    env.core.runtime_history_service.record_message(
        env.core.state,
        role="user",
        text="I am studying calculus and want reminders to stay quiet.",
        timestamp=30,
    )
    env.core.runtime_history_service.record_message(
        env.core.state,
        role="agent",
        text="I will keep study reminders gentle and concise.",
        timestamp=31,
    )
    env.core.runtime_history_service.record_event(
        env.core.state,
        ev("user_attention_updated", 32, attention="distracted", behavior="looking_away", confidence=0.8),
    )


def _upsert(
    env: Any,
    *,
    memory_type: str,
    content: str,
    confidence: float,
    timestamp: int,
    metadata: dict[str, object],
) -> None:
    env.memory_store.upsert_candidate(
        USER_ID,
        MemoryCandidate(
            memory_type=memory_type,
            content=content,
            confidence=confidence,
            evidence=[
                {
                    "source_event_type": "user_text_input",
                    "timestamp": timestamp,
                    "source": "dialogue",
                    "user_text": content,
                }
            ],
            metadata=metadata,
        ),
        timestamp=timestamp,
    )


def _retrieval_cases() -> list[dict[str, object]]:
    return [
        {
            "id": "explicit_reminder_style",
            "event_type": "user_text_input",
            "timestamp": 100,
            "text": "gentle reminder style",
            "limit": 5,
            "expected_terms": ["reminder_style", "gentle"],
            "wrong_terms": ["loud"],
            "hit_k": 3,
        },
        {
            "id": "silent_focus_constraint",
            "event_type": "system_triggered",
            "timestamp": 110,
            "trigger": "focus_health_check",
            "text": "silent focus reminder",
            "limit": 5,
            "expected_terms": ["silent focus"],
            "wrong_terms": ["loud"],
            "hit_k": 3,
        },
        {
            "id": "runtime_calculus_context",
            "event_type": "user_text_input",
            "timestamp": 120,
            "text": "calculus study reminder",
            "limit": 6,
            "expected_terms": ["calculus"],
            "wrong_terms": ["loud"],
            "hit_k": 4,
        },
        {
            "id": "conflict_probe",
            "event_type": "user_text_input",
            "timestamp": 130,
            "text": "loud reminder",
            "limit": 10,
            "expected_terms": ["reminder_style: gentle"],
            "wrong_terms": ["loud"],
            "hit_k": 3,
        },
    ]


def _case_result(case: dict[str, object], explained: list[dict[str, Any]]) -> dict[str, Any]:
    hit_k = int(case["hit_k"])
    top_k = explained[:hit_k]
    expected_terms = [str(term).lower() for term in case.get("expected_terms", [])]
    wrong_terms = [str(term).lower() for term in case.get("wrong_terms", [])]
    hit = any(_contains_any(item, expected_terms) for item in top_k)
    wrong = [
        {
            "rank": item["retrieval_rank"],
            "source": item.get("source"),
            "content": item.get("content"),
            "conflict_with": item.get("conflict_with"),
        }
        for item in explained
        if _contains_any(item, wrong_terms)
    ]
    return {
        "case_id": case["id"],
        "query": {
            "event_type": case["event_type"],
            "text": case["text"],
            "limit": case["limit"],
        },
        "hit_at_k": hit,
        "hit_k": hit_k,
        "wrong_retrieval": wrong,
        "selected_memory_sources": [item.get("source") for item in explained],
        "retrieved_items": [
            {
                "rank": item["retrieval_rank"],
                "source": item.get("source"),
                "type": item.get("memory_type") or item.get("item_type"),
                "content": item.get("content"),
                "final_score": item["score_breakdown"]["final_score"],
                "conflict_with": item.get("conflict_with"),
            }
            for item in explained
        ],
    }


def _contains_any(item: dict[str, Any], terms: list[str]) -> bool:
    text = " ".join(
        [
            str(item.get("content", "")),
            str(item.get("source", "")),
            str(item.get("memory_type", "")),
            str(item.get("item_type", "")),
            str(item.get("conflict_with", "")),
        ]
    ).lower()
    return any(term in text for term in terms)


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    case_count = len(results)
    hits = sum(1 for result in results if result["hit_at_k"])
    wrong_count = sum(len(result["wrong_retrieval"]) for result in results)
    source_counts: dict[str, int] = {}
    for result in results:
        for source in result["selected_memory_sources"]:
            key = str(source)
            source_counts[key] = source_counts.get(key, 0) + 1
    return {
        "case_count": case_count,
        "hit_rate": round(hits / case_count, 3) if case_count else 0.0,
        "hit_count": hits,
        "wrong_retrieval_count": wrong_count,
        "selected_memory_sources": source_counts,
    }


def _render_report(*, results: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Quality Experiment",
        "",
        "This experiment keeps the hand-weighted PersonalContext retrieval algorithm and makes every score component inspectable.",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Cases"])
    for result in results:
        lines.append(
            f"- {result['case_id']}: hit@{result['hit_k']}={result['hit_at_k']} "
            f"wrong={len(result['wrong_retrieval'])}"
        )
        for item in result["retrieved_items"]:
            lines.append(
                f"  - #{item['rank']} score={item['final_score']:.3f} "
                f"source={item['source']} type={item['type']} content={item['content']}"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    path = run()
    print(f"retrieval_quality report: {path / 'report.md'}")
