from __future__ import annotations

"""Agent 事件桥：向 AgentCore 注入标准语音生命周期事件。"""

import time
from typing import Any, Protocol

from src.agent.event.event_model import Event


class EventSink(Protocol):
    def handle_event(self, event: Any) -> Any: ...


class AgentBridge:
    def __init__(self, sink: EventSink | None = None) -> None:
        self._sink = sink

    def bind(self, sink: EventSink) -> None:
        self._sink = sink

    def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        if self._sink is None:
            return
        self._sink.handle_event(
            Event(
                type=event_type,
                timestamp=int(time.time()),
                payload=payload,
            )
        )

    def emit_voice_input_started(self, session_id: str) -> None:
        self._emit("voice_input_started", {"session_id": session_id, "source": "board_voice"})

    def emit_voice_input_stopped(self, session_id: str) -> None:
        self._emit("voice_input_stopped", {"session_id": session_id, "source": "board_voice"})

    def emit_tts_started(self, text: str, *, source: str = "board_voice") -> None:
        self._emit("tts_started", {"text": text, "source": source})

    def emit_tts_finished(
        self,
        text: str,
        *,
        source: str = "board_voice",
        reason: str = "",
        kind: str = "",
        cancelled: bool = False,
    ) -> None:
        payload: dict[str, object] = {"text": text, "source": source}
        if reason:
            payload["reason"] = reason
        if kind:
            payload["kind"] = kind
        if cancelled:
            payload["cancelled"] = True
        self._emit("tts_finished", payload)

    def emit_tts_cancelled(self, text: str, *, reason: str, kind: str = "") -> None:
        self.emit_tts_finished(
            text,
            reason=reason,
            kind=kind,
            cancelled=True,
        )

    def emit_speech_recognized(
        self,
        *,
        text: str,
        session_id: str,
        confidence: float = 0.9,
    ) -> None:
        self._emit(
            "speech_recognized",
            {
                "text": text,
                "source": "board_voice",
                "session_id": session_id,
                "confidence": confidence,
                "language": "zh",
                "is_final": True,
            },
        )
