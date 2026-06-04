from __future__ import annotations

from pathlib import Path

from common import ExperimentLLM, ExperimentRecorder, build_environment, ev


def build_events() -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = [
        (
            "declare gentle low-frequency preference",
            ev("user_text_input", 1000, text="我喜欢温和提醒，也希望你少提醒我。", source="experiment"),
        ),
        ("start 40 minute study session", ev("focus_start_requested", 1010, duration_sec=2400, source="experiment")),
    ]

    for minute in range(1, 31):
        ts = 1010 + minute * 60
        events.append((f"focus timer minute {minute}", ev("timer_ticked", ts, remaining_sec=max(0, 2400 - minute * 60))))
        if minute in {6, 13, 20, 27}:
            events.append(
                (
                    f"attention drift minute {minute}",
                    ev("user_attention_updated", ts + 1, attention="distracted", behavior="looking_away", confidence=0.82),
                )
            )
            events.append(
                (
                    f"health check after drift {minute}",
                    ev("system_triggered", ts + 2, trigger="focus_health_check", source="experiment"),
                )
            )
        if minute in {10, 22}:
            events.append(
                (
                    f"focused again minute {minute}",
                    ev("user_attention_updated", ts + 3, attention="focused", behavior="reading", confidence=0.9),
                )
            )
        if minute == 16:
            events.append(("fatigue high", ev("user_fatigue_updated", ts + 4, fatigue_level="high", source="vision")))
            events.append(
                (
                    "rest suggestion opportunity",
                    ev("system_triggered", ts + 5, trigger="focus_health_check", source="experiment"),
                )
            )
        if minute == 17:
            events.append(
                (
                    "explicit feedback against frequent interruption",
                    ev("user_text_input", ts + 6, text="刚才提醒可以，但不要频繁打断我。", source="experiment"),
                )
            )
        if minute == 23:
            events.append(("user leaves briefly", ev("user_presence_updated", ts + 7, presence="away", confidence=0.95)))
            events.append(
                (
                    "blocked reminder while away",
                    ev("system_triggered", ts + 8, trigger="focus_health_check", source="experiment"),
                )
            )
        if minute == 25:
            events.append(("user returns", ev("user_presence_updated", ts + 9, presence="present", confidence=0.95)))

    events.extend(
        [
            ("timer complete", ev("timer_finished", 1010 + 40 * 60, timer="focus")),
            (
                "post session reinforcement",
                ev("user_text_input", 1010 + 40 * 60 + 20, text="这次温和提醒有帮助，以后保持这种方式。", source="experiment"),
            ),
            ("manual rest", ev("focus_stop_requested", 1010 + 40 * 60 + 60, source="experiment")),
        ]
    )
    return events


def run(output_root: str | Path | None = None) -> Path:
    env = build_environment("study_session_experiment", llm=ExperimentLLM(), output_root=output_root)
    recorder = ExperimentRecorder("study_session_experiment", env)
    try:
        for label, event in build_events():
            recorder.run_event(event, label=label)
        return recorder.write_outputs(
            title="Study Session Runtime Experiment",
            notes=[
                "Simulates a 40-minute study session with distraction, fatigue, absence, rest and user feedback.",
                "Expected: reminders stay bounded by guard/cooldown and personalization keeps gentle reminders visible in prompts.",
            ],
        )
    finally:
        env.shutdown()


if __name__ == "__main__":
    path = run()
    print(f"study_session_experiment report: {path / 'report.md'}")
