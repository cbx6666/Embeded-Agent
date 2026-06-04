from __future__ import annotations

"""
Board-side USB display hardware adapter.

Implements the DisplayHardware protocol by sending JSON-line commands
over a USB CDC/ACM serial port to the PC display application.

On Atlas 200I DK A2, the USB Type-C configured as a gadget appears as
/dev/ttyGS0 (gadget) or /dev/ttyACM0 (cdc-acm). The port path can be
configured via constructor or the EMBED_USB_DISPLAY_PORT env var.
"""

import logging
import os
import threading
import time
from typing import Any, Optional

from src.adapters.usb_display.serial_protocol import (
    encode_display,
    encode_expression,
    encode_light,
)

logger = logging.getLogger(__name__)


class USBDisplayHardware:
    """Sends display commands over USB serial to the PC pet app."""

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 0.5,
    ) -> None:
        self._port = port or os.environ.get(
            "EMBED_USB_DISPLAY_PORT", "/dev/ttyGS0"
        )
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial = None
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────

    def open(self) -> None:
        try:
            import serial  # pyserial
        except ImportError:
            logger.error("pyserial not installed; cannot open USB display port")
            return

        with self._lock:
            if self._serial is not None and self._serial.is_open:
                return
            try:
                self._serial = serial.Serial(
                    self._port,
                    baudrate=self._baudrate,
                    timeout=self._timeout,
                    write_timeout=self._timeout,
                )
                logger.info("USB display port %s opened @ %d baud", self._port, self._baudrate)
            except serial.SerialException as exc:
                logger.error("Failed to open USB display port %s: %s", self._port, exc)

    def close(self) -> None:
        with self._lock:
            if self._serial is not None and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None

    # ── DisplayHardware protocol ───────────────────────────────

    def render_expression(self, expression: str, payload: dict[str, Any]) -> None:
        text = payload.get("text", "").strip()
        if text and expression in ("status", "display"):
            line = encode_display(
                text=text,
                kind=str(payload.get("kind", "") or ""),
                status=str(payload.get("status", "") or ""),
                duration_ms=payload.get("duration_ms"),
            )
        else:
            line = encode_expression(
                expression=expression,
                style=str(payload.get("style", "") or ""),
                intensity=payload.get("intensity"),
                duration_ms=payload.get("duration_ms"),
            )
        self._write_line(line)

    def read_sensor_snapshot(self) -> dict[str, Any] | None:
        return None

    def set_light_state(self, payload: dict[str, Any]) -> None:
        line = encode_light(
            state=str(payload.get("state", "off")),
            color=str(payload.get("color", "") or ""),
            pattern=str(payload.get("pattern", "") or ""),
            brightness=payload.get("brightness"),
            duration_ms=payload.get("duration_ms"),
        )
        self._write_line(line)

    # ── internal ───────────────────────────────────────────────

    def _write_line(self, line: str) -> None:
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                logger.warning("USB display port not open; dropping: %s", line[:80])
                return
            try:
                self._serial.write((line + "\n").encode("utf-8"))
                self._serial.flush()
            except Exception as exc:
                logger.error("USB display write error: %s", exc)
