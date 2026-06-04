"""
PC-side display viewer — reads JSON display commands from stdin and renders them.

Usage:
  # View a dry-run test output in real time
  python scripts/debug/test_usb_display.py --stdout | python scripts/debug/display_viewer.py

  # Or replay from a saved file
  python scripts/debug/display_viewer.py < test_output.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

# ── expression → ASCII art ───────────────────────────────────────

EXPRESSION_FACES: dict[str, str] = {
    "happy":    "(＾▽＾)",
    "idle":     "(・_・)",
    "sad":      "(╥_╥)",
    "surprised":"(⊙o⊙)",
    "blink":    "(・_・ )~",
    "angry":    "(｀皿´)",
    "love":     "(´▽`ʃƪ)♡",
    "sleep":    "(－.－)Zzz",
    "wink":     "(^_−)☆",
}

EXPRESSION_COLORS: dict[str, str] = {
    "happy": "\033[93m",   # yellow
    "idle":  "\033[0m",    # default
    "sad":   "\033[94m",   # blue
    "surprised": "\033[95m",  # magenta
    "blink": "\033[0m",
    "angry": "\033[91m",   # red
    "love":  "\033[95m",   # magenta
    "sleep": "\033[96m",   # cyan
    "wink":  "\033[93m",
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def render_expression(cmd: dict) -> None:
    payload = cmd.get("payload", {})
    name = payload.get("expression", "idle")
    intensity = payload.get("intensity")
    style = payload.get("style")
    duration = payload.get("duration_ms")

    face = EXPRESSION_FACES.get(name, f"({name})")
    color = EXPRESSION_COLORS.get(name, "")

    info_parts = []
    if intensity is not None:
        info_parts.append(f"intensity={intensity}")
    if style:
        info_parts.append(f"style={style}")
    if duration:
        info_parts.append(f"duration={duration}ms")
    info = ", ".join(info_parts)

    width = max(50, len(face) + 6)
    print(f"{color}{'─' * width}{RESET}")
    print(f"{color}{BOLD}  {face}  [expression: {name}]{RESET}")
    if info:
        print(f"{color}  {DIM}{info}{RESET}{color}")
    print(f"{color}{'─' * width}{RESET}")


def render_display(cmd: dict) -> None:
    payload = cmd.get("payload", {})
    text = payload.get("text", "")
    kind = payload.get("kind", "speech")
    duration = payload.get("duration_ms")

    bubble_size = max(20, min(60, len(text) * 2 + 4))
    top = "╭" + "─" * (bubble_size - 2) + "╮"
    bottom = "╰" + "─" * (bubble_size - 2) + "╯"

    print(f"{top}")
    print(f"│ {text:<{bubble_size - 3}}│")
    info_line = f"  [{kind}]"
    if duration:
        info_line += f" {duration}ms"
    print(f"{bottom}  {DIM}{info_line}{RESET}")


def render_light(cmd: dict) -> None:
    payload = cmd.get("payload", {})
    state = payload.get("state", "off")
    color = payload.get("color", "")
    brightness = payload.get("brightness")
    pattern = payload.get("pattern")
    duration = payload.get("duration_ms")

    light_glyphs = {"on": "💡 ON", "off": "💡 OFF", "blink": "💡 BLINK", "pulse": "💡 PULSE"}
    glyph = light_glyphs.get(state, f"💡 {state.upper()}")

    info_parts = []
    if color:
        info_parts.append(f"color={color}")
    if brightness is not None:
        info_parts.append(f"brightness={brightness}")
    if pattern:
        info_parts.append(f"pattern={pattern}")
    if duration:
        info_parts.append(f"{duration}ms")
    info = ", ".join(info_parts)

    bar_len = min(30, brightness // 8 + 1) if brightness else 0
    bar = "█" * bar_len + "░" * (30 - bar_len) if bar_len else ""

    print(f"  {glyph}")
    if bar:
        print(f"  [{bar}]")
    if info:
        print(f"  {DIM}{info}{RESET}")


# ── main loop ────────────────────────────────────────────────────

def main() -> None:
    count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            print(f"{DIM}[skip: invalid JSON]{RESET}")
            continue

        cmd_type = cmd.get("type", "unknown")
        ts = cmd.get("ts", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "--:--:--"

        print(f"\n{DIM}[{time_str}] #{count} ── {cmd_type} ──{RESET}")

        if cmd_type == "expression":
            render_expression(cmd)
        elif cmd_type == "display":
            render_display(cmd)
        elif cmd_type == "light":
            render_light(cmd)
        else:
            print(f"  {BOLD}? unknown command type: {cmd_type}{RESET}")
            print(f"  {json.dumps(cmd, ensure_ascii=False)}")

        count += 1

    print(f"\n{DIM}── {count} commands rendered ──{RESET}")


if __name__ == "__main__":
    main()
