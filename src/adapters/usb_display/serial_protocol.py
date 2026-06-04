"""
USB Serial JSON-line protocol for Atlas board ↔ PC desktop pet communication.

Each message is a single JSON object terminated by a newline (\\n).
The board sends display commands; the PC optionally sends back sensor/ack frames.

Direction: Board → PC

Command types:
  - expression: render pet expression
  - display: show text bubble
  - light: set light/background state

Message format:
  {"type": "<command>", "ts": 1234567890, "payload": {...}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

MESSAGE_DELIMITER = "\n"

CommandType = Literal["expression", "display", "light"]


@dataclass
class DisplayCommand:
    """Normalized command sent from board to PC display app."""

    type: CommandType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {"type": self.type, "ts": self.ts, "payload": self.payload},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(line: str) -> "DisplayCommand | None":
        try:
            obj = json.loads(line.strip())
            return DisplayCommand(
                type=obj["type"],
                payload=obj.get("payload", {}),
                ts=float(obj.get("ts", 0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


def encode_expression(
    expression: str,
    *,
    style: Optional[str] = None,
    intensity: Optional[float] = None,
    duration_ms: Optional[int] = None,
    ts: Optional[float] = None,
) -> str:
    import time

    return DisplayCommand(
        type="expression",
        ts=ts or time.time(),
        payload={
            "expression": expression,
            "style": style,
            "intensity": intensity,
            "duration_ms": duration_ms,
        },
    ).to_json()


def encode_display(
    text: str,
    *,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    duration_ms: Optional[int] = None,
    ts: Optional[float] = None,
) -> str:
    import time

    return DisplayCommand(
        type="display",
        ts=ts or time.time(),
        payload={
            "text": text,
            "kind": kind,
            "status": status,
            "duration_ms": duration_ms,
        },
    ).to_json()


def encode_light(
    state: str,
    *,
    color: Optional[str] = None,
    pattern: Optional[str] = None,
    brightness: Optional[int] = None,
    duration_ms: Optional[int] = None,
    ts: Optional[float] = None,
) -> str:
    import time

    return DisplayCommand(
        type="light",
        ts=ts or time.time(),
        payload={
            "state": state,
            "color": color,
            "pattern": pattern,
            "brightness": brightness,
            "duration_ms": duration_ms,
        },
    ).to_json()


def parse_command(line: str) -> DisplayCommand | None:
    return DisplayCommand.from_json(line)
