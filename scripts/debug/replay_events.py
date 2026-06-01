from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_experiments.common import ExperimentLLM, ExperimentRecorder, build_environment, write_json
from src.agent.event import Event


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an event log through the deterministic experiment runtime.")
    parser.add_argument("events", help="Path to events.json.")
    parser.add_argument("--name", default="debug_replay", help="Output run name.")
    parser.add_argument("--output-root", default="data/experiments/replay", help="Output root directory.")
    args = parser.parse_args()

    events = [_event_from_dict(item) for item in json.loads(Path(args.events).read_text(encoding="utf-8"))]
    env = build_environment(args.name, llm=ExperimentLLM(), output_root=args.output_root)
    recorder = ExperimentRecorder(args.name, env)
    try:
        for index, event in enumerate(events, start=1):
            recorder.run_event(event, label=f"replay #{index} {event.type}")
        out = recorder.write_outputs(
            title=f"Replay Report: {args.name}",
            notes=[f"Replayed {len(events)} events from {args.events}."],
        )
        write_json(out / "replay_summary.json", recorder.metrics())
        print(f"replay report: {out / 'report.md'}")
    finally:
        env.shutdown()


def _event_from_dict(item: dict[str, object]) -> Event:
    payload = item.get("payload", {})
    return Event(
        type=str(item.get("type", "")),
        timestamp=int(item.get("timestamp", 0)),
        payload=dict(payload) if isinstance(payload, dict) else {},
    )


if __name__ == "__main__":
    main()
