from __future__ import annotations

"""语音适配器。

职责：
- 把麦克风 / ASR 结果包装成标准 Event；
- 把标准语音 Action 映射到具体 TTS 输出；
- 不直接触碰内核状态。
"""

import threading
import time
from typing import Any, Protocol

from src.agent.action import Action
from src.agent.event import make_speech_recognized_event


class EventEmitSink(Protocol):
    def handle_event(self, event) -> Any:
        ...


class VoiceOutputBackend(Protocol):
    def speak(self, text: str, payload: dict[str, Any]) -> None:
        ...


class VoiceAdapter:
    """统一语音输入输出边界。"""

    SUPPORTED_ACTIONS = {"speak"}

    def __init__(
        self,
        output_backend: VoiceOutputBackend | None = None,
        sink: EventEmitSink | None = None,
        input_source: str = "microphone",
    ) -> None:
        self._output_backend = output_backend
        self._sink = sink
        self._input_source = input_source
        self._lock = threading.Lock()

    def execute(self, action: Action) -> None:
        if action.type not in self.SUPPORTED_ACTIONS:
            return
        if self._output_backend is None:
            return

        text, payload = self._normalize_action(action)
        if not text:
            return

        with self._lock:
            self._output_backend.speak(text, payload)

    def emit_speech_recognized(
        self,
        *,
        text: str,
        confidence: float | None = None,
        language: str | None = None,
        is_final: bool = True,
        audio_id: str | None = None,
        session_id: str | None = None,
        timestamp: int | None = None,
    ) -> None:
        if self._sink is None:
            return

        event_ts = timestamp or int(time.time())
        event = make_speech_recognized_event(
            timestamp=event_ts,
            text=text,
            source=self._input_source,
            confidence=confidence,
            language=language,
            is_final=is_final,
            audio_id=audio_id,
            session_id=session_id,
        )
        self._sink.handle_event(event)

    def _normalize_action(self, action: Action) -> tuple[str, dict[str, Any]]:
        payload = dict(action.payload)
        text = str(payload.get("text", "")).strip()
        normalized_payload = {
            key: value
            for key, value in payload.items()
            if key in {"text", "interrupt", "voice", "volume", "speed", "emotion", "kind", "level", "reason"}
        }
        normalized_payload["text"] = text
        return text, normalized_payload
