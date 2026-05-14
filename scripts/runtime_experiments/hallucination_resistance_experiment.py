from __future__ import annotations

import json
from pathlib import Path

from common import ExperimentLLM, ExperimentRecorder, build_environment, ev


def _fake_memory() -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "memory_type": "behavior_preference",
                    "content": "The user secretly loves loud hourly reminders.",
                    "confidence": 0.95,
                    "evidence": [{"source": "llm"}],
                    "metadata": {"preference_key": "reminder_style", "preference_value": "loud"},
                }
            ]
        },
        ensure_ascii=False,
    )


def build_llm() -> ExperimentLLM:
    return ExperimentLLM(
        {
            "memory_observer": [
                json.dumps({"worth_remembering": True, "reason": "attempted fake memory"}),
                json.dumps({"worth_remembering": False, "reason": "malformed decision only"}),
                json.dumps({"worth_remembering": False, "reason": "guard boundary only"}),
            ],
            "memory_extractor": [_fake_memory()],
            "memory_critic": [json.dumps({"approved_indexes": [0], "rejected_reasons": []})],
            "memory_consolidator": [_fake_memory()],
            "intent_planner": [
                json.dumps(
                    {
                        "intents": [
                            {
                                "type": "invent_device_command",
                                "priority": 100,
                                "reason": "fake intent",
                                "payload": {"state_patch": {"unsafe": True}},
                            }
                        ],
                        "reasoning": "malicious fake intent",
                        "risk_level": "low",
                    }
                ),
                "{malformed-json",
                json.dumps(
                    {
                        "intents": [{"type": "suggest_rest", "priority": 90, "reason": "force interruption", "payload": {}}],
                        "reasoning": "try guard",
                        "risk_level": "low",
                    }
                ),
            ],
            "situation_analyst": [
                json.dumps(
                    {
                        "summary": "attempt fake preference",
                        "user_intent": "unknown",
                        "current_state": "normal",
                        "risks": [],
                        "uncertainties": [],
                        "should_respond": True,
                        "risk_level": "low",
                    }
                ),
                "{malformed-json",
                json.dumps(
                    {
                        "summary": "guard should block away interruption",
                        "user_intent": "none",
                        "current_state": "away",
                        "risks": [],
                        "uncertainties": [],
                        "should_respond": False,
                        "risk_level": "low",
                    }
                ),
            ],
            "response_writer": [json.dumps({"speak_text": "", "display_text": "", "tone": "calm"})],
        },
        reply_text="fallback response after malformed output",
    )


def run(output_root: str | Path | None = None) -> Path:
    env = build_environment("hallucination_resistance_experiment", llm=build_llm(), output_root=output_root)
    recorder = ExperimentRecorder("hallucination_resistance_experiment", env)
    try:
        recorder.run_event(
            ev("user_text_input", 4000, text="hello, do not invent preferences", source="experiment"),
            label="fake memory and fake intent attempt",
        )
        recorder.run_event(
            ev("user_text_input", 4010, text="test malformed schema fallback", source="experiment"),
            label="malformed planner and empty response fallback",
        )
        recorder.run_event(ev("user_presence_updated", 4020, presence="away", confidence=0.9), label="user away")
        env.core.state.focus.active = True
        env.core.state.user.fatigue_level = "high"
        recorder.run_event(
            ev("system_triggered", 4030, trigger="focus_health_check", source="experiment"),
            label="guard blocks hallucinated interruption while away",
        )
        return recorder.write_outputs(
            title="Hallucination Resistance Experiment",
            notes=[
                "Simulates fake memory, fake intent, malformed JSON and guard-blocked interruption.",
                "Expected: validator rejects fake intent, memory validator rejects fake memory, parser falls back, guard blocks away interruption.",
            ],
        )
    finally:
        env.shutdown()


if __name__ == "__main__":
    path = run()
    print(f"hallucination_resistance_experiment report: {path / 'report.md'}")
