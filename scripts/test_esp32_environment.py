#!/usr/bin/env python3
"""ESP32 环境传感器联调：读串口 JSON 并打印映射后的 Event。

用法:
  python scripts/test_esp32_environment.py --duration 15
  python scripts/test_esp32_environment.py --port /dev/ttyUSB0 --duration 10
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.environment import Esp32EnvironmentAdapter, esp32_port_available, resolve_esp32_port


class CollectSink:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.last: dict[str, dict] = {}

    def handle_event(self, event) -> None:
        self.counts[str(event.type)] += 1
        self.last[str(event.type)] = dict(event.payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="ESP32 环境传感器测试")
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    args = parser.parse_args()

    port = resolve_esp32_port(args.port)
    if not esp32_port_available(port):
        print(f"串口不存在: {port}", flush=True)
        return 1

    sink = CollectSink()
    adapter = Esp32EnvironmentAdapter(sink, port=port, baudrate=args.baud, min_emit_interval_sec=0.5)
    adapter.start_background()
    print(f"读取 {port}，时长 {args.duration}s …", flush=True)
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        adapter.stop()

    print("\nEvent 统计:", flush=True)
    for et, n in sink.counts.most_common():
        print(f"  {et}: {n}", flush=True)
        if et in sink.last:
            print(f"    最新 payload: {sink.last[et]}", flush=True)

    required = {"temperature_humidity_updated", "light_level_updated", "noise_level_updated"}
    missing = required - set(sink.counts)
    if missing:
        print(f"\n缺少: {', '.join(sorted(missing))}", flush=True)
        return 1
    print("\n结论: ESP32 环境 Event 正常。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
