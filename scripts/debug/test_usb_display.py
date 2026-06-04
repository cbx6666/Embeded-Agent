"""
Standalone USB display hardware test — bypasses Agent core entirely.

Usage:
  python scripts/debug/test_usb_display.py --port /dev/ttyGS0
  python scripts/debug/test_usb_display.py --port /dev/ttyGS0 --interactive
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import time
from typing import Optional, Union

# ── add project root so "src.adapters..." imports work ──────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.usb_display.serial_protocol import (
    encode_display,
    encode_expression,
    encode_light,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_usb_display")


# ── port I/O (pyserial preferred, raw fd fallback) ──────────────

_SerialType = Union["serial.Serial", "RawSerial", "StdoutSerial"]


class StdoutSerial:
    """Fake serial that prints JSON to stdout — for local dry-run testing."""

    def __init__(self) -> None:
        pass

    def open(self) -> None:
        logger.info("Dry-run mode — JSON lines printed to stdout, read from stdin")

    def close(self) -> None:
        pass

    def write(self, data: bytes) -> int:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return len(data)

    def readline(self, timeout: float = 1.0) -> Optional[bytes]:
        import select
        r, _, _ = select.select([sys.stdin], [], [], min(timeout, 0.2))
        if r:
            line = sys.stdin.buffer.readline()
            return line if line else None
        return None

    def flush(self) -> None:
        sys.stdout.buffer.flush()

    @property
    def is_open(self) -> bool:
        return True


class RawSerial:
    """Minimal serial-like wrapper using raw file descriptors on Linux.

    USB gadget serial ports (/dev/ttyGS*, /dev/ttyACM*) are virtual —
    baud rate is ignored, so plain open() works fine.
    """

    def __init__(self, port: str) -> None:
        self._port = port
        self._fd = -1

    def open(self) -> None:
        if self._fd >= 0:
            return
        # O_NOCTTY: don't make this the controlling terminal
        fd = os.open(self._port, os.O_RDWR | os.O_NOCTTY)
        self._fd = fd
        logger.info("Raw fd opened: %s (fd=%d)", self._port, fd)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def write(self, data: bytes) -> int:
        return os.write(self._fd, data)

    def readline(self, timeout: float = 1.0) -> Optional[bytes]:
        """Read one line with timeout. Returns None on timeout."""
        chunks = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remain = max(0, deadline - time.monotonic())
            r, _, _ = select.select([self._fd], [], [], min(remain, 0.1))
            if not r:
                break
            ch = os.read(self._fd, 1)
            if not ch:
                break
            chunks.append(ch)
            if ch == b"\n":
                return b"".join(chunks)
        return b"".join(chunks) if chunks else None

    def flush(self) -> None:
        pass  # raw fd writes are unbuffered

    @property
    def is_open(self) -> bool:
        return self._fd >= 0


def _open_port_pyserial(port: str, baudrate: int) -> "serial.Serial":
    import serial
    logger.info("Opening %s @ %d baud (pyserial)...", port, baudrate)
    ser = serial.Serial(port, baudrate, timeout=0.5, write_timeout=0.5)
    logger.info("Port opened: %s", ser)
    return ser


def _list_tty_devices() -> list[str]:
    import glob
    patterns = ["/dev/ttyGS*", "/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyS*"]
    found: list[str] = []
    for pat in patterns:
        found.extend(sorted(glob.glob(pat)))
    return found


def _open_port_raw(port: str, _baudrate: int) -> RawSerial:
    logger.info("Opening %s (raw fd, pyserial unavailable)...", port)
    ser = RawSerial(port)
    try:
        ser.open()
    except OSError as exc:
        logger.error("Cannot open %s: %s", port, exc)
        available = _list_tty_devices()
        if available:
            logger.info("Available tty devices: %s", ", ".join(available))
        else:
            logger.info("No /dev/tty* serial devices found. Is the USB gadget configured?")
        sys.exit(1)
    return ser


def try_open_port(port: str, baudrate: int = 115200) -> _SerialType:
    try:
        import serial  # noqa: F401
        return _open_port_pyserial(port, baudrate)
    except ImportError:
        logger.info("pyserial not installed — using raw file descriptor fallback")
        return _open_port_raw(port, baudrate)


def send_line(ser: _SerialType, line: str) -> None:
    raw = (line + "\n").encode("utf-8")
    logger.info("TX (%d bytes): %s", len(raw), line)
    ser.write(raw)
    ser.flush()


def read_line(ser: _SerialType, timeout: float = 1.0) -> Optional[str]:
    raw = ser.readline(timeout)
    if raw:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw.rstrip("\n\r")
        logger.info("RX: %s", line)
        return line
    return None


# ── test sequences ───────────────────────────────────────────────

def run_basic_test(ser: _SerialType) -> None:
    logger.info("=== Basic expression / display / light test ===")

    expressions = [
        ("happy", {"intensity": 0.9, "duration_ms": 2000}),
        ("idle", {"duration_ms": 1500}),
        ("surprised", {"intensity": 0.7, "duration_ms": 2000}),
        ("sad", {"style": "anime", "duration_ms": 2000}),
        ("blink", {"duration_ms": 500}),
    ]

    for expr, kwargs in expressions:
        line = encode_expression(expr, **kwargs)
        send_line(ser, line)
        time.sleep(0.3)

    texts = [
        ("你好！", {"kind": "speech"}),
        ("今天天气不错", {"kind": "thought", "duration_ms": 3000}),
        ("测试完成 ✅", {"kind": "speech", "duration_ms": 4000}),
    ]
    for text, kwargs in texts:
        line = encode_display(text, **kwargs)
        send_line(ser, line)
        time.sleep(0.3)

    light_states = [
        ("on", {"color": "#FF9900", "brightness": 200, "duration_ms": 2000}),
        ("blink", {"color": "#00FF00", "pattern": "slow", "duration_ms": 3000}),
        ("off", {}),
    ]
    for state, kwargs in light_states:
        line = encode_light(state, **kwargs)
        send_line(ser, line)
        time.sleep(0.3)

    for _ in range(10):
        read_line(ser, timeout=0.3)

    logger.info("=== Basic test done ===")


def run_interactive(ser: _SerialType) -> None:
    logger.info("=== Interactive mode ===")
    logger.info("Commands: e:<name>[:intensity]  d:<text>  l:<state>  q=quit")
    logger.info("Example: e:happy:0.8  d:Hello!  l:on")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        if cmd.lower() in ("q", "quit", "exit"):
            break

        line: Optional[str] = None
        if cmd.startswith("e:"):
            parts = cmd[2:].split(":", 1)
            expr = parts[0]
            intensity = float(parts[1]) if len(parts) > 1 else None
            line = encode_expression(expr, intensity=intensity, duration_ms=3000)
        elif cmd.startswith("d:"):
            text = cmd[2:]
            line = encode_display(text, kind="speech", duration_ms=3000)
        elif cmd.startswith("l:"):
            state = cmd[2:]
            line = encode_light(state, duration_ms=2000)
        else:
            logger.warning("Unknown command: %s", cmd)
            continue

        send_line(ser, line)
        read_line(ser, timeout=0.2)


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="USB Display standalone test")
    parser.add_argument(
        "--list-ports", action="store_true",
        help="List available /dev/tty* devices and exit",
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("EMBED_USB_DISPLAY_PORT", "/dev/ttyGS0"),
        help="Serial port path (default: $EMBED_USB_DISPLAY_PORT or /dev/ttyGS0)",
    )
    parser.add_argument(
        "--baudrate", type=int, default=115200, help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Enter interactive command mode after basic test",
    )
    parser.add_argument(
        "--no-basic", action="store_true",
        help="Skip the basic test sequence",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print JSON to stdout instead of opening a serial port (local dry-run)",
    )
    args = parser.parse_args()

    if args.list_ports:
        devices = _list_tty_devices()
        if devices:
            print("Available tty devices:", ", ".join(devices))
        else:
            print("No /dev/tty* serial devices found.")
        return

    if args.stdout:
        ser = StdoutSerial()
    else:
        ser = try_open_port(args.port, args.baudrate)

    try:
        if not args.no_basic:
            run_basic_test(ser)

        if args.interactive:
            run_interactive(ser)
        elif args.no_basic:
            run_interactive(ser)
    finally:
        ser.close()
        logger.info("Port closed.")


if __name__ == "__main__":
    main()
