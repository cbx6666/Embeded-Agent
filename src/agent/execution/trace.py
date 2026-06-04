from __future__ import annotations

"""Lightweight deterministic trace models."""

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeTraceEvent:
    """One deterministic trace entry for runtime and behavior tests."""

    sequence: int
    stage: str
    label: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "stage": self.stage,
            "label": self.label,
            "payload": _stable(self.payload),
        }


@dataclass
class RuntimeTrace:
    """A tiny trace collector with JSON/debug output and assertion helpers."""

    events: list[RuntimeTraceEvent] = field(default_factory=list)

    def add(self, stage: str, label: str, payload: dict[str, Any] | None = None, **fields: Any) -> RuntimeTraceEvent:
        data: dict[str, Any] = {}
        if payload:
            data.update(payload)
        data.update(fields)
        event = RuntimeTraceEvent(
            sequence=len(self.events) + 1,
            stage=str(stage),
            label=str(label),
            payload=_stable(data),
        )
        self.events.append(event)
        return event

    def extend(self, other: RuntimeTrace | dict[str, Any] | None) -> None:
        if other is None:
            return
        if isinstance(other, RuntimeTrace):
            raw_events = [event.to_dict() for event in other.events]
        else:
            value = other.get("events", []) if isinstance(other, dict) else []
            raw_events = value if isinstance(value, list) else []
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload", {})
            self.add(
                stage=str(item.get("stage", "")),
                label=str(item.get("label", "")),
                payload=payload if isinstance(payload, dict) else {"value": payload},
            )

    def to_dict(self) -> dict[str, Any]:
        return {"events": [event.to_dict() for event in self.events]}

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    def dump_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    def to_debug_string(self) -> str:
        lines: list[str] = []
        for event in self.events:
            keys = ", ".join(sorted(event.payload)) if event.payload else "-"
            lines.append(f"{event.sequence:02d} {event.stage}:{event.label} [{keys}]")
        return "\n".join(lines)

    def debug_print(self) -> None:
        print(self.to_debug_string())

    def stages(self) -> tuple[str, ...]:
        return tuple(event.stage for event in self.events)

    def find(self, stage: str, label: str | None = None) -> list[RuntimeTraceEvent]:
        return [
            event
            for event in self.events
            if event.stage == stage and (label is None or event.label == label)
        ]


@dataclass
class AgentDecisionTrace:
    """Compact loop trace retained for AgentLoop recent-trace compatibility."""

    event_type: str
    timestamp: int
    state_summary: dict[str, Any]
    intents: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    decision_metadata: dict[str, Any] = field(default_factory=dict)
    loop_step: int = 0


def _stable(value: Any) -> Any:
    """Convert values to JSON-stable structures without adding timestamps."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _stable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _stable(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _stable(value.to_dict())
    return str(value)
