from __future__ import annotations

"""环境传感器串口行解析：ESP32 JSON、STM32 JSON、legacy 光照文本。"""

import json
import re
from dataclasses import dataclass

LIGHT_LINE_PATTERN = re.compile(r"Light:\s*([+-]?\d+(?:\.\d+)?)\s*lx", re.IGNORECASE)


@dataclass
class EnvironmentSensorReading:
    """单行解析结果；未出现的字段为 None（由适配器与上次读数合并）。"""

    temperature_c: float | None = None
    humidity_pct: float | None = None
    light_lux: float | None = None
    noise_db: float | None = None

    def has_any_field(self) -> bool:
        return any(
            value is not None
            for value in (self.temperature_c, self.humidity_pct, self.light_lux, self.noise_db)
        )

    def merge(self, other: EnvironmentSensorReading) -> EnvironmentSensorReading:
        return EnvironmentSensorReading(
            temperature_c=other.temperature_c if other.temperature_c is not None else self.temperature_c,
            humidity_pct=other.humidity_pct if other.humidity_pct is not None else self.humidity_pct,
            light_lux=other.light_lux if other.light_lux is not None else self.light_lux,
            noise_db=other.noise_db if other.noise_db is not None else self.noise_db,
        )


def parse_environment_sensor_line(line: str) -> EnvironmentSensorReading | None:
    """解析一行串口数据，支持 JSON 与 legacy `Light: 123.4 lx`。"""
    text = line.strip()
    if not text:
        return None

    if text.startswith("{"):
        return _parse_json_line(text)

    legacy = _parse_legacy_light_line(text)
    if legacy is not None:
        return legacy

    return None


def parse_esp32_sensor_line(line: str) -> EnvironmentSensorReading | None:
    """兼容旧名；要求温湿度+噪声齐全（用于单测断言完整 JSON）。"""
    reading = parse_environment_sensor_line(line)
    if reading is None:
        return None
    if (
        reading.temperature_c is None
        or reading.humidity_pct is None
        or reading.light_lux is None
        or reading.noise_db is None
    ):
        return None
    return reading


def _parse_json_line(line: str) -> EnvironmentSensorReading | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    reading = EnvironmentSensorReading()
    try:
        if "temperature" in obj or "temperature_c" in obj:
            reading.temperature_c = float(obj.get("temperature", obj.get("temperature_c")))
        if "humidity" in obj or "humidity_pct" in obj:
            reading.humidity_pct = float(obj.get("humidity", obj.get("humidity_pct")))
        if "lux" in obj or "light_lux" in obj:
            reading.light_lux = float(obj.get("lux", obj.get("light_lux", 0)))
        if "noise_db" in obj:
            reading.noise_db = float(obj["noise_db"])
    except (TypeError, ValueError):
        return None

    return reading if reading.has_any_field() else None


def _parse_legacy_light_line(line: str) -> EnvironmentSensorReading | None:
    match = LIGHT_LINE_PATTERN.search(line)
    if match is None:
        return None
    return EnvironmentSensorReading(light_lux=float(match.group(1)))
