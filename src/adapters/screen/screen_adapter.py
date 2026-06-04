"""Screen display adapter - connects Agent actions to pygame window."""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol

from src.agent.action import Action
from src.agent.event import make_display_sensor_event
from src.adapters.console_output import ConsoleOutput


class DisplayHardware(Protocol):
    """Protocol for display hardware."""

    def start(self) -> None:
        """Start display."""
        ...

    def stop(self) -> None:
        """Stop display."""
        ...

    def update(
        self,
        agent_state: str,
        speak_text: str = "",
        focus_remaining: int = 0,
        focus_duration: int = 0,
    ) -> None:
        """Update display state."""
        ...


class EventEmitSink(Protocol):
    """Protocol for event emission sink."""

    def handle_event(self, event) -> Any:
        ...


class ScreenDisplayAdapter:
    """Adapter that consumes display actions and updates pygame window.

    This adapter handles:
    - display: show text/status
    - render_pet_expression: update pet face state
    - set_light_state: control LED (placeholder)
    """

    SUPPORTED_ACTIONS = {"display", "render_pet_expression", "set_light_state"}

    def __init__(
        self,
        hardware: DisplayHardware,
        sink: EventEmitSink | None = None,
        source: str = "screen_display",
        console_output: ConsoleOutput | None = None,
    ) -> None:
        self._hardware = hardware
        self._sink = sink
        self._source = source
        self._lock = threading.Lock()
        self._current_state = "idle"
        self._speak_text = ""
        self._focus_remaining = 0
        self._focus_duration = 0
        self._console_output = console_output or ConsoleOutput()

    def execute(self, action: Action) -> None:
        """Execute a display action (thread-safe)."""
        # 先传递给 console_output
        try:
            self._console_output.execute(action)
        except:
            pass
            
        if action.type not in self.SUPPORTED_ACTIONS:
            return

        with self._lock:
            if action.type == "set_light_state":
                self._execute_light_action(action)
                return

            self._update_from_action(action)
            self._hardware.update(
                agent_state=self._current_state,
                speak_text=self._speak_text,
                focus_remaining=self._focus_remaining,
                focus_duration=self._focus_duration,
            )

    def show_text(self, text: str) -> None:
        """Print text to console (for CLI compatibility)."""
        self._console_output.show_text(text)

    def update_focus_timer(self, remaining: int, duration: int) -> None:
        """Update focus timer display."""
        with self._lock:
            self._focus_remaining = remaining
            self._focus_duration = duration
            self._hardware.update(
                agent_state=self._current_state,
                speak_text=self._speak_text,
                focus_remaining=self._focus_remaining,
                focus_duration=self._focus_duration,
            )

    def emit_sensor_snapshot(
        self,
        *,
        expression: str | None = None,
        brightness: int | None = None,
        fps: int | None = None,
        sensor_values: dict[str, object] | None = None,
        screen_id: str | None = None,
        timestamp: int | None = None,
    ) -> None:
        """Emit display sensor event."""
        if self._sink is None:
            return
        event = make_display_sensor_event(
            timestamp=timestamp or int(time.time()),
            expression=expression or self._current_state,
            source=self._source,
            brightness=brightness,
            fps=fps,
            sensor_values=sensor_values,
            screen_id=screen_id,
        )
        self._sink.handle_event(event)

    def _update_from_action(self, action: Action) -> None:
        """Extract display state from action payload."""
        payload = dict(action.payload)

        if action.type == "display":
            text = str(payload.get("text", "")).strip()
            kind = str(payload.get("kind", payload.get("status", "status"))).strip()

            if text:
                self._speak_text = text

            # Map kind to agent state
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
            elif kind == "reminder":
                self._speak_text = text

        elif action.type == "render_pet_expression":
            expression = str(payload.get("expression", "neutral")).strip()
            expr_state_map = {
                "neutral": "idle",
                "happy": "speaking",
                "listening": "listening",
                "thinking": "thinking",
                "focused": "focus_mode",
                "tired": "idle",
                "stressed": "idle",
            }
            self._current_state = expr_state_map.get(expression, "idle")

    def _execute_light_action(self, action: Action) -> None:
        """Execute light state action (placeholder for LED control)."""
        # Placeholder for actual LED hardware
        pass

    def get_current_state(self) -> tuple[str, str, int, int]:
        """Get current display state (for debugging)."""
        with self._lock:
            return self._current_state, self._speak_text, self._focus_remaining, self._focus_duration