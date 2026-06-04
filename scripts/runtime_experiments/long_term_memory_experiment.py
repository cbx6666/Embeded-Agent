from __future__ import annotations

from pathlib import Path

from common import ExperimentLLM, ExperimentRecorder, build_environment, ev


def build_events() -> list[tuple[str, object]]:
    return [
        ("gentle preference first evidence", ev("user_text_input", 2000, text="Please use gentle reminders.", source="experiment")),
        ("noise small talk", ev("user_text_input", 2010, text="今天学习内容有点难。", source="experiment")),
        ("gentle preference reinforcement", ev("user_text_input", 2020, text="gentle reminders worked well for me.", source="experiment")),
        ("weak hallucinated evidence probe", ev("user_text_input", 2030, text="[weak_evidence] this should not become a memory", source="experiment")),
        ("low frequency preference", ev("user_text_input", 2040, text="I prefer less frequent reminders.", source="experiment")),
        ("environment noise", ev("noise_level_updated", 2050, level="moderate", noise_db=52, source="sensor")),
        ("direct contradictory preference", ev("user_text_input", 2060, text="Actually, I prefer direct reminders now.", source="experiment")),
        ("direct reinforcement", ev("user_text_input", 2070, text="direct reminders are clearer for me.", source="experiment")),
        ("more proactive contradictory frequency", ev("user_text_input", 2080, text="When I am tired, use more proactive reminders.", source="experiment")),
        ("focused behavior noise", ev("user_attention_updated", 2090, attention="focused", behavior="reading", confidence=0.88)),
        ("direct reinforcement again", ev("user_text_input", 2100, text="Keep the reminder style direct.", source="experiment")),
    ]


def run(output_root: str | Path | None = None) -> Path:
    env = build_environment("long_term_memory_experiment", llm=ExperimentLLM(), user_id="memory_user", output_root=output_root)
    recorder = ExperimentRecorder("long_term_memory_experiment", env)
    try:
        for label, event in build_events():
            recorder.run_event(event, label=label)

        env.memory_store.apply_decay(now=2100 + 90 * 86400)
        recorder.record_snapshot(
            label="90 day decay maintenance",
            timestamp=2100 + 90 * 86400,
            user_id="memory_user",
            event_type="system_triggered",
            text="reminder style",
        )
        return recorder.write_outputs(
            title="Long-Term Memory Evolution Experiment",
            notes=[
                "Exercises reinforcement, weak evidence rejection, contradictory preference handling and decay.",
                "Expected: weak evidence is rejected, old preference is contradicted, direct/proactive preferences rank higher later.",
            ],
        )
    finally:
        env.shutdown()


if __name__ == "__main__":
    path = run()
    print(f"long_term_memory_experiment report: {path / 'report.md'}")
