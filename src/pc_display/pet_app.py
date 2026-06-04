"""
PC-side Desktop Pet Display Application.

Connects to the Atlas 200I DK A2 board over USB CDC/ACM serial,
receives display commands, and renders an interactive desktop pet
on the PC screen.

Usage:
    # run standalone (serial from Atlas board)
    python -m src.pc_display.pet_app --port COM3

    # run with demo mode (no hardware needed, cycles expressions)
    python -m src.pc_display.pet_app --demo

    # list available serial ports
    python -m src.pc_display.pet_app --list-ports
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pc_display.pet_renderer import PetRenderer
from src.pc_display.serial_reader import SerialReader

logger = logging.getLogger(__name__)

DEMO_EXPRESSIONS = [
    ("neutral", 2500),
    ("happy", 3000),
    ("thinking", 4000),
    ("surprised", 2000),
    ("excited", 3000),
    ("happy", 2000),
    ("sleepy", 3500),
    ("annoyed", 2500),
    ("neutral", 2000),
    ("sad", 3000),
    ("neutral", 2000),
]

DEMO_MESSAGES = [
    (None, 0),       # neutral
    ("Hello!", 0),   # happy
    ("Hmm, let me think about that...", 0),  # thinking
    ("Oh!", 0),      # surprised
    ("That's amazing!", 0),  # excited
    ("I like you!", 0),      # happy
    ("So sleepy...", 0),      # sleepy
    ("Hmph.", 0),    # annoyed
    (None, 0),       # neutral
    ("I miss you...", 0),    # sad
    (None, 0),       # neutral
]


def list_serial_ports() -> list[str]:
    try:
        import serial.tools.list_ports
    except ImportError:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def run_demo() -> None:
    """Run the pet in demo mode — cycles through expressions without hardware."""
    renderer = PetRenderer(scale=1.2, fps=30)
    renderer.set_expression("neutral")

    import threading

    def demo_loop() -> None:
        idx = 0
        while renderer._running:
            expr, dur = DEMO_EXPRESSIONS[idx % len(DEMO_EXPRESSIONS)]
            msg, _ = DEMO_MESSAGES[idx % len(DEMO_MESSAGES)]
            renderer.set_expression(expr)
            if msg:
                renderer.show_bubble(msg)
            idx += 1
            time.sleep(dur / 1000.0)

    thread = threading.Thread(target=demo_loop, daemon=True)
    thread.start()
    renderer.run()


def run_with_serial(
    port: str,
    baudrate: int = 115200,
    scale: float = 1.0,
) -> None:
    """Run the pet display connected to Atlas board over USB serial."""
    renderer = PetRenderer(scale=scale)

    reader = SerialReader(
        port=port,
        baudrate=baudrate,
        timeout=0.5,
        reconnect_delay=2.0,
    )
    reader.bind_renderer(renderer)
    reader.start()

    try:
        renderer.run()
    finally:
        reader.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Desktop Pet PC Display Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=str,
        default=os.environ.get("EMBED_PC_DISPLAY_PORT"),
        help="Serial port for USB connection to Atlas board (e.g. COM3, /dev/ttyACM0)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baudrate (default: 115200)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Pet display scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (no hardware needed)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_ports:
        ports = list_serial_ports()
        if ports:
            print("Available serial ports:")
            for p in ports:
                print(f"  {p}")
        else:
            print("No serial ports found (pyserial may not be installed).")
        return

    if args.demo:
        logger.info("Starting demo mode")
        run_demo()
        return

    if not args.port:
        parser.error(
            "No --port specified and EMBED_PC_DISPLAY_PORT not set.\n"
            "Use --port COM3 (Windows) or --port /dev/ttyACM0 (Linux), "
            "or use --demo for demo mode."
        )

    logger.info("Connecting to Atlas board on %s", args.port)
    print(
        f"Desktop pet starting...\n"
        f"  Port: {args.port}\n"
        f"  Baudrate: {args.baudrate}\n"
        f"  Press ESC or close the pet window to exit.\n"
    )
    run_with_serial(args.port, args.baudrate, args.scale)


if __name__ == "__main__":
    main()
