from __future__ import annotations

"""环境读数 → Agent 标准 level 字段（可配置阈值，与 asr-test 默认对齐）。"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentThresholdConfig:
    low_light_lux: float = 120.0
    low_temperature_c: float = 18.0
    high_temperature_c: float = 30.0
    dry_humidity_pct: float = 30.0
    humid_humidity_pct: float = 70.0
    noisy_db: float = 65.0


def _env_float(name: str, explicit: float | None, default: float) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def resolve_environment_thresholds(
    *,
    low_light_lux: float | None = None,
    low_temperature_c: float | None = None,
    high_temperature_c: float | None = None,
    dry_humidity_pct: float | None = None,
    humid_humidity_pct: float | None = None,
    noisy_db: float | None = None,
) -> EnvironmentThresholdConfig:
    return EnvironmentThresholdConfig(
        low_light_lux=_env_float("EMBED_ENV_LOW_LIGHT_LUX", low_light_lux, 120.0),
        low_temperature_c=_env_float("EMBED_ENV_LOW_TEMPERATURE_C", low_temperature_c, 18.0),
        high_temperature_c=_env_float("EMBED_ENV_HIGH_TEMPERATURE_C", high_temperature_c, 30.0),
        dry_humidity_pct=_env_float("EMBED_ENV_DRY_HUMIDITY_PCT", dry_humidity_pct, 30.0),
        humid_humidity_pct=_env_float("EMBED_ENV_HUMID_HUMIDITY_PCT", humid_humidity_pct, 70.0),
        noisy_db=_env_float("EMBED_ENV_NOISY_DB", noisy_db, 65.0),
    )


def light_level_from_lux(
    lux: float,
    thresholds: EnvironmentThresholdConfig | None = None,
) -> tuple[str, bool]:
    cfg = thresholds or EnvironmentThresholdConfig()
    if lux < 30:
        return "dark", True
    if lux < cfg.low_light_lux:
        return "low", True
    if lux < cfg.low_light_lux * 2:
        return "normal", False
    return "bright", False


def noise_level_from_db(
    noise_db: float,
    thresholds: EnvironmentThresholdConfig | None = None,
) -> tuple[str, bool]:
    cfg = thresholds or EnvironmentThresholdConfig()
    if noise_db >= cfg.noisy_db:
        return "high", True
    if noise_db >= cfg.noisy_db * 0.7:
        return "normal", False
    return "low", False


def temperature_level_from_c(
    temp_c: float,
    thresholds: EnvironmentThresholdConfig | None = None,
) -> str:
    cfg = thresholds or EnvironmentThresholdConfig()
    if temp_c < cfg.low_temperature_c:
        return "low"
    if temp_c > cfg.high_temperature_c:
        return "high"
    return "normal"


def humidity_level_from_pct(
    humidity_pct: float,
    thresholds: EnvironmentThresholdConfig | None = None,
) -> str:
    cfg = thresholds or EnvironmentThresholdConfig()
    if humidity_pct < cfg.dry_humidity_pct:
        return "dry"
    if humidity_pct > cfg.humid_humidity_pct:
        return "humid"
    return "normal"
