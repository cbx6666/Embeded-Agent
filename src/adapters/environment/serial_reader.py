from __future__ import annotations

"""串口逐行读取：优先 pyserial，回退 Linux termios。"""

import logging
import os
import select
import time
from typing import Iterator, Protocol

logger = logging.getLogger(__name__)

BAUDRATE_MAP: dict[int, int] = {}


def _init_baudrate_map() -> dict[int, int]:
    import termios

    return {
        9600: getattr(termios, "B9600", 0),
        19200: getattr(termios, "B19200", 0),
        38400: getattr(termios, "B38400", 0),
        57600: getattr(termios, "B57600", 0),
        115200: getattr(termios, "B115200", 0),
    }


class LineSerialReader(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def readline(self, timeout: float = 1.0) -> str | None: ...


class PySerialLineReader:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2.0) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._ser = None

    def open(self) -> None:
        import serial

        self._ser = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
        )
        logger.debug("environment serial: pyserial on %s @ %d", self._port, self._baudrate)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def readline(self, timeout: float = 1.0) -> str | None:
        if self._ser is None:
            return None
        old_timeout = self._ser.timeout
        try:
            self._ser.timeout = timeout
            raw = self._ser.readline()
        finally:
            self._ser.timeout = old_timeout
        if not raw:
            return None
        line = raw.decode("utf-8", errors="replace").strip()
        return line or None


class TermiosLineReader:
    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._port = port
        self._baudrate = baudrate
        self._fd = -1

    def open(self) -> None:
        global BAUDRATE_MAP
        if not BAUDRATE_MAP:
            BAUDRATE_MAP = _init_baudrate_map()
        if self._baudrate not in BAUDRATE_MAP:
            supported = ", ".join(str(v) for v in sorted(BAUDRATE_MAP))
            raise RuntimeError(f"termios 模式下仅支持这些波特率：{supported}")
        self._fd = os.open(self._port, os.O_RDWR | os.O_NOCTTY)
        self._configure_termios()
        logger.debug("environment serial: termios on %s @ %d", self._port, self._baudrate)

    def _configure_termios(self) -> None:
        import termios

        attrs = termios.tcgetattr(self._fd)
        speed = BAUDRATE_MAP[self._baudrate]
        attrs[4] = speed
        attrs[5] = speed
        attrs[2] = attrs[2] & ~termios.CSIZE | termios.CS8
        attrs[2] = attrs[2] & ~(termios.PARENB | termios.CSTOPB)
        attrs[3] = attrs[3] & ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def readline(self, timeout: float = 1.0) -> str | None:
        if self._fd < 0:
            return None
        chunks: list[bytes] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remain = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._fd], [], [], min(remain, 0.2))
            if not ready:
                continue
            ch = os.read(self._fd, 1)
            if not ch:
                break
            chunks.append(ch)
            if ch == b"\n":
                break
        if not chunks:
            return None
        return b"".join(chunks).decode("utf-8", errors="replace").strip()


def open_line_serial_reader(port: str, baudrate: int = 115200, timeout: float = 2.0) -> LineSerialReader:
    try:
        import serial  # noqa: F401

        return PySerialLineReader(port, baudrate, timeout)
    except ImportError:
        return TermiosLineReader(port, baudrate)


def iter_serial_lines(*, port: str, baudrate: int, timeout: float = 2.0) -> Iterator[str]:
    reader = open_line_serial_reader(port, baudrate, timeout)
    reader.open()
    try:
        while True:
            line = reader.readline(timeout=timeout)
            if line:
                yield line
    finally:
        reader.close()
