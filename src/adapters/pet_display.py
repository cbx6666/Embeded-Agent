from __future__ import annotations

"""桌宠显示屏适配器。

职责：
- 消费标准 Action，把桌宠表情渲染到具体显示设备；
- 从显示侧或挂载传感器采样后，转换成标准 Event；
- 不直接修改 core/state，只通过注入的 sink 发事件。
"""

import threading
import time
from typing import Any, Protocol

from src.agent.action import Action
from src.agent.event import display_sensor_updated


class EventEmitSink(Protocol):
    def handle_event(self, event) -> Any:
        ...


class DisplayHardware(Protocol):
    def render_expression(self, expression: str, payload: dict[str, Any]) -> None:
        ...

    def read_sensor_snapshot(self) -> dict[str, Any] | None:
        ...


class PetDisplayAdapter:
    """将标准动作映射到显示硬件，并为传感器预留事件出口。"""

    def __init__(
        self,
        hardware: DisplayHardware,
        sink: EventEmitSink | None = None,
        source: str = "pet_display",
    ) -> None:
        self._hardware = hardware
        self._sink = sink
        self._source = source
        self._lock = threading.Lock()

    def execute(self, action: Action) -> None:
        if action.type not in {"display", "render_pet_expression"}:
            return
        expression, payload = self._normalize_action(action)
        with self._lock:
            self._hardware.render_expression(expression, payload)

    def emit_sensor_snapshot(
        self,
        *,
        expression: str,
        brightness: int | None = None,
        fps: int | None = None,
        sensor_values: dict[str, object] | None = None,
        screen_id: str | None = None,
        timestamp: int | None = None,
    ) -> None:
        if self._sink is None:
            return
        event = display_sensor_updated(
            timestamp=timestamp or int(time.time()),
            expression=expression,
            source=self._source,
            brightness=brightness,
            fps=fps,
            sensor_values=sensor_values,
            screen_id=screen_id,
        )
        self._sink.handle_event(event)

    def poll_and_emit_sensor_snapshot(
        self,
        *,
        expression: str,
        screen_id: str | None = None,
        timestamp: int | None = None,
    ) -> None:
        snapshot = self._hardware.read_sensor_snapshot() or {}
        brightness = snapshot.get("brightness")
        fps = snapshot.get("fps")
        sensor_values = snapshot.get("sensor_values")
        self.emit_sensor_snapshot(
            expression=expression,
            brightness=int(brightness) if brightness is not None else None,
            fps=int(fps) if fps is not None else None,
            sensor_values=sensor_values if isinstance(sensor_values, dict) else None,
            screen_id=screen_id,
            timestamp=timestamp,
        )

    def _normalize_action(self, action: Action) -> tuple[str, dict[str, Any]]:
        payload = dict(action.payload)
        if action.type == "render_pet_expression":
            expression = str(payload.get("expression", "neutral")).strip() or "neutral"
            return expression, payload
        text = str(payload.get("text", "")).strip()
        expression = str(payload.get("kind", "status")).strip() or "status"
        normalized_payload = {"text": text, "kind": expression}
        return expression, normalized_payload
