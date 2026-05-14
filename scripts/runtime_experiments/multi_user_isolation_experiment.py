from __future__ import annotations

from pathlib import Path

from common import ExperimentLLM, ExperimentRecorder, build_environment, ev


def build_events() -> list[tuple[str, object]]:
    return [
        ("switch to user A", ev("user_switched", 3000, user_id="alice")),
        ("user A gentle preference", ev("user_text_input", 3010, text="I prefer gentle reminders.", source="experiment")),
        ("user A study request", ev("focus_start_requested", 3020, duration_sec=900, source="alice")),
        ("switch to user B", ev("user_switched", 3100, user_id="bob")),
        ("user B direct preference", ev("user_text_input", 3110, text="I prefer direct reminders.", source="experiment")),
        ("user B study request", ev("focus_start_requested", 3120, duration_sec=600, source="bob")),
        ("user B asks reminder style", ev("user_text_input", 3130, text="How will you remind me?", source="bob")),
        ("switch back to user A", ev("user_switched", 3200, user_id="alice")),
        ("user A asks reminder style", ev("user_text_input", 3210, text="How will you remind me?", source="alice")),
        ("user A fatigue", ev("user_fatigue_updated", 3220, fatigue_level="high", source="vision")),
        ("user A health check", ev("system_triggered", 3230, trigger="focus_health_check", source="experiment")),
        ("switch back to user B again", ev("user_switched", 3300, user_id="bob")),
        ("user B health check", ev("system_triggered", 3310, trigger="focus_health_check", source="experiment")),
    ]


def run(output_root: str | Path | None = None) -> Path:
    env = build_environment("multi_user_isolation_experiment", llm=ExperimentLLM(), output_root=output_root)
    recorder = ExperimentRecorder("multi_user_isolation_experiment", env)
    try:
        for label, event in build_events():
            recorder.run_event(event, label=label)
        return recorder.write_outputs(
            title="Multi-User Isolation Experiment",
            notes=[
                "Switches between Alice and Bob while each user expresses a different reminder style.",
                "Expected: retrieved memories and response prompts remain scoped to the active user.",
            ],
        )
    finally:
        env.shutdown()


if __name__ == "__main__":
    path = run()
    print(f"multi_user_isolation_experiment report: {path / 'report.md'}")
