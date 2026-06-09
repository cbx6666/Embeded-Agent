"""Screen display adapter - connects Agent actions to pygame window."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Protocol

from src.agent.action import Action
from src.adapters.console_output import ConsoleOutput
from src.adapters.screen.expression_styles import resolve_expression
from src.adapters.screen.pet_display_context import PetDisplayContext
from src.adapters.screen.pet_renderer import reason_status_label

_EMOTION_LEVEL = {"neutral": 1.0, "happy": 2.0, "tired": 1.5, "stressed": 2.5}
_FATIGUE_LEVEL = {"none": 0.0, "mild": 1.0, "moderate": 2.0, "high": 3.0}
_UI_PUSH_MIN_INTERVAL_SEC = 1.0
_STATS_CACHE_TTL_SEC = 5.0


class DisplayHardware(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def update(self, context: PetDisplayContext) -> None:
        ...


class EventEmitSink(Protocol):
    def handle_event(self, event) -> Any:
        ...


_DIALOGUE_TO_PET_STATE = {
    "idle": "idle",
    "listening": "listening",
    "thinking": "thinking",
    "speaking": "speaking",
}


class ScreenDisplayAdapter:
    """Adapter that consumes display actions and updates pygame dashboard."""

    SUPPORTED_ACTIONS = {"display"}

    def __init__(
        self,
        hardware: DisplayHardware,
        sink: EventEmitSink | None = None,
        source: str = "screen_display",
        console_output: ConsoleOutput | None = None,
        stats_provider: Callable[[], dict[str, dict[str, int]]] | None = None,
    ) -> None:
        self._hardware = hardware
        self._sink = sink
        self._source = source
        self._lock = threading.Lock()
        self._current_state = "idle"
        self._speak_text = ""
        self._focus_remaining = 0
        self._focus_duration = 0
        self._status_label = ""
        self._console_output = console_output or ConsoleOutput()
        self._stats_provider = stats_provider

        self._user_speech_text = ""
        self._agent_speech_text = ""
        self._speech_mode = ""
        self._emotion = "neutral"
        self._fatigue = "none"
        self._temperature_c: float | None = None
        self._humidity_pct: float | None = None
        self._light_lux: int | None = None
        self._noise_db: int | None = None
        self._temperature_level = ""
        self._humidity_level = ""
        self._light_level = ""
        self._noise_level = ""
        self._emotion_confidence: float | None = None
        self._fatigue_confidence: float | None = None
        self._emotion_timeline: deque[tuple[int, float]] = deque(maxlen=180)
        self._fatigue_timeline: deque[tuple[int, float]] = deque(maxlen=180)
        self._last_timeline_ts = 0
        self._last_hardware_push_mono = 0.0
        self._stats_cache: dict[str, dict[str, int]] | None = None
        self._stats_cache_mono = 0.0

    def execute(self, action: Action) -> None:
        try:
            self._console_output.execute(action)
        except Exception:
            pass

        if action.type not in self.SUPPORTED_ACTIONS:
            return

        with self._lock:
            self._update_from_action(action)
            self._push_hardware_update()

    def show_text(self, text: str) -> None:
        self._console_output.show_text(text)

    def feed_event(self, event: Any) -> None:
        """跟踪语音生命周期事件，驱动聆听/播报字幕。"""
        etype = str(getattr(event, "type", ""))
        payload = dict(getattr(event, "payload", {}) or {})

        force_push = False
        with self._lock:
            if etype == "voice_input_started":
                self._speech_mode = "listening"
                self._user_speech_text = ""
                force_push = True
            elif etype == "voice_input_stopped":
                self._speech_mode = "recognizing"
                force_push = True
            elif etype == "speech_recognized":
                text = str(payload.get("text", "")).strip()
                if text:
                    self._user_speech_text = text
                    self._speech_mode = "user"
                    force_push = True
            elif etype == "tts_started":
                text = str(payload.get("text", "")).strip()
                if text:
                    self._agent_speech_text = text
                    self._speech_mode = "agent"
                    force_push = True
            elif etype == "tts_finished":
                if self._speech_mode == "agent":
                    self._speech_mode = ""
                    force_push = True
            elif etype in {"user_emotion_updated", "user_fatigue_updated"}:
                pass  # 由 sync_visual_state 从 AgentState 刷新
            if force_push:
                self._maybe_push_hardware_update(force=True)

    def sync_visual_state(
        self,
        *,
        dialogue_state: str = "idle",
        focus_active: bool = False,
        focus_remaining: int = 0,
        focus_duration: int = 0,
        temperature_c: float | None = None,
        humidity_pct: float | None = None,
        light_lux: int | None = None,
        noise_db: int | None = None,
        temperature_level: str = "",
        humidity_level: str = "",
        light_level: str = "",
        noise_level: str = "",
        emotion: str = "neutral",
        fatigue: str = "none",
        emotion_confidence: float | None = None,
        fatigue_confidence: float | None = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            if focus_active:
                self._current_state = "focus_mode"
                self._focus_remaining = max(0, int(focus_remaining))
                self._focus_duration = max(0, int(focus_duration))
            else:
                self._current_state = _DIALOGUE_TO_PET_STATE.get(dialogue_state, "idle")

            if temperature_c is not None:
                self._temperature_c = temperature_c
            if humidity_pct is not None:
                self._humidity_pct = humidity_pct
            if light_lux is not None:
                self._light_lux = int(light_lux)
            if noise_db is not None:
                self._noise_db = int(noise_db)
            if temperature_level:
                self._temperature_level = str(temperature_level)
            if humidity_level:
                self._humidity_level = str(humidity_level)
            if light_level:
                self._light_level = str(light_level)
            if noise_level:
                self._noise_level = str(noise_level)

            self._emotion = emotion or "neutral"
            self._fatigue = fatigue or "none"
            if emotion_confidence is not None:
                self._emotion_confidence = float(emotion_confidence)
            if fatigue_confidence is not None:
                self._fatigue_confidence = float(fatigue_confidence)

            if dialogue_state == "listening" and not self._user_speech_text:
                self._speech_mode = "listening"
            elif dialogue_state == "thinking" and self._speech_mode == "listening":
                self._speech_mode = "recognizing"
            elif dialogue_state == "speaking" and self._agent_speech_text:
                self._speech_mode = "agent"
            elif dialogue_state == "idle" and self._speech_mode in {"listening", "recognizing"}:
                self._speech_mode = ""

            self._maybe_push_hardware_update(force=force)

    def _maybe_push_hardware_update(self, *, force: bool = False) -> None:
        """将缓存状态推送到 pygame 线程；默认最多 1 次/秒，避免感知高频事件拖慢 Agent。"""
        now = time.monotonic()
        if not force and now - self._last_hardware_push_mono < _UI_PUSH_MIN_INTERVAL_SEC:
            return
        self._last_hardware_push_mono = now
        self._append_timeline_sample()
        self._push_hardware_update()

    def update_focus_timer(self, remaining: int | None, duration: int | None) -> None:
        self.sync_visual_state(
            focus_active=True,
            focus_remaining=max(0, int(remaining or 0)),
            focus_duration=max(0, int(duration or 0)),
        )

    def _append_timeline_sample(self) -> None:
        now = int(time.time())
        if now == self._last_timeline_ts:
            return
        self._last_timeline_ts = now
        self._emotion_timeline.append((now, _EMOTION_LEVEL.get(self._emotion, 1.0)))
        self._fatigue_timeline.append((now, _FATIGUE_LEVEL.get(self._fatigue, 0.0)))

    def _build_context(self) -> PetDisplayContext:
        emotion_pie: dict[str, int] = {}
        fatigue_pie: dict[str, int] = {}
        stats = self._cached_stats()
        if stats:
            emotion_pie = dict(stats.get("emotion_seconds") or {})
            fatigue_pie = dict(stats.get("fatigue_seconds") or {})
        if not emotion_pie:
            emotion_pie = {self._emotion: 60}
        if not fatigue_pie:
            fatigue_pie = {self._fatigue: 60}

        return PetDisplayContext(
            agent_state=self._current_state,
            expression=resolve_expression(agent_state=self._current_state, user_emotion=self._emotion),
            status_label=self._status_label,
            speak_text=self._speak_text,
            user_speech_text=self._user_speech_text,
            agent_speech_text=self._agent_speech_text or self._speak_text,
            speech_mode=self._speech_mode,
            focus_remaining=self._focus_remaining,
            focus_duration=self._focus_duration,
            temperature_c=self._temperature_c,
            humidity_pct=self._humidity_pct,
            light_lux=self._light_lux,
            noise_db=self._noise_db,
            temperature_level=self._temperature_level,
            humidity_level=self._humidity_level,
            light_level=self._light_level,
            noise_level=self._noise_level,
            emotion=self._emotion,
            fatigue=self._fatigue,
            emotion_confidence=self._emotion_confidence,
            fatigue_confidence=self._fatigue_confidence,
            emotion_pie=emotion_pie,
            fatigue_pie=fatigue_pie,
            emotion_timeline=list(self._emotion_timeline),
            fatigue_timeline=list(self._fatigue_timeline),
        )

    def _cached_stats(self) -> dict[str, dict[str, int]]:
        if self._stats_provider is None:
            return {}
        now = time.monotonic()
        if (
            self._stats_cache is not None
            and now - self._stats_cache_mono < _STATS_CACHE_TTL_SEC
        ):
            return self._stats_cache
        try:
            stats = self._stats_provider() or {}
            self._stats_cache = {
                "emotion_seconds": dict(stats.get("emotion_seconds") or {}),
                "fatigue_seconds": dict(stats.get("fatigue_seconds") or {}),
            }
            self._stats_cache_mono = now
            return self._stats_cache
        except Exception:
            return self._stats_cache or {}

    def _push_hardware_update(self) -> None:
        self._hardware.update(self._build_context())

    def _update_from_action(self, action: Action) -> None:
        payload = dict(action.payload)

        if action.type == "display":
            text = str(payload.get("text", "")).strip()
            kind = str(payload.get("kind", payload.get("status", "status"))).strip()

            if text:
                self._speak_text = text
                if kind in {"speaking", "reminder", "notification", "status_report"}:
                    self._agent_speech_text = text
                    self._speech_mode = "agent"

            state_map = {
                "listening": "listening",
                "thinking": "thinking",
                "speaking": "speaking",
                "focus": "focus_mode",
                "focus_mode": "focus_mode",
                "idle": "idle",
                "status": self._current_state,
            }
            if kind in state_map:
                self._current_state = state_map[kind]
                self._status_label = ""
            elif kind == "notification":
                reason = str(payload.get("reason", "")).strip()
                self._status_label = reason_status_label(reason)
                self._speak_text = text
            elif kind == "reminder":
                self._speak_text = text

    def get_current_state(self) -> tuple[str, str, int, int]:
        with self._lock:
            return self._current_state, self._speak_text, self._focus_remaining, self._focus_duration
