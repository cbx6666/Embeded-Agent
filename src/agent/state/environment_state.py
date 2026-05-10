from dataclasses import dataclass


@dataclass
class EnvironmentState:
    """环境状态：保存光照、噪声、温湿度的标准化字段。"""

    light_lux: int | None = None
    light_level: str | None = None
    noise_db: int | None = None
    noise_level: str | None = None
    temperature_c: float | None = None
    temperature_level: str | None = None
    humidity_pct: float | None = None
    humidity_level: str | None = None
