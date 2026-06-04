from __future__ import annotations

"""
PC-side USB serial reader.

Runs in a background thread, reads JSON-line commands from a serial port
(USB CDC/ACM from the Atlas board), and pushes DisplayCommand objects
into the pet renderer's command queue.
"""

import logging
import threading
import time
from typing import Callable, Optional

from src.adapters.usb_display.serial_protocol import DisplayCommand, parse_command
from src.pc_display.pet_renderer import PetRenderer

logger = logging.getLogger(__name__)


class SerialReader:
    """Reads display commands from USB serial and dispatches to renderer."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.5,
        reconnect_delay: float = 2.0,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._reconnect_delay = reconnect_delay

        self._serial = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._renderer: Optional[PetRenderer] = None

        # line buffer for partial reads
        self._buf = ""

    def bind_renderer(self, renderer: PetRenderer) -> None:
        self._renderer = renderer

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("SerialReader started on %s", self._port)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_serial()
        logger.info("SerialReader stopped")

    # ── internal ───────────────────────────────────────────────

    def _open_serial(self) -> bool:
        try:
            import serial
        except ImportError:
            logger.error("pyserial not installed")
            return False

        try:
            self._serial = serial.Serial(
                self._port,
                baudrate=self._baudrate,
                timeout=self._timeout,
            )
            logger.info("Connected to %s @ %d baud", self._port, self._baudrate)
            return True
        except serial.SerialException as exc:
            logger.warning("Cannot open %s: %s (will retry)", self._port, exc)
            return False

    def _close_serial(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def _run(self) -> None:
        while self._running:
            if self._serial is None or not self._serial.is_open:
                self._close_serial()
                if not self._open_serial():
                    time.sleep(self._reconnect_delay)
                    continue
                self._buf = ""

            try:
                raw = self._serial.read(256)
                if not raw:
                    continue
                self._buf += raw.decode("utf-8", errors="replace")

                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    self._handle_line(line.strip())
            except serial.SerialException as exc:
                logger.warning("Serial error: %s", exc)
                self._close_serial()
                time.sleep(self._reconnect_delay)
            except Exception as exc:
                logger.error("Unexpected error in SerialReader: %s", exc)
                time.sleep(0.5)

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        cmd = parse_command(line)
        if cmd is None:
            logger.debug("Unparseable line: %s", line[:80])
            return
        if self._renderer is not None:
            self._renderer.push_command(cmd)
        else:
            logger.debug("No renderer bound; dropping command: %s", cmd.type)
