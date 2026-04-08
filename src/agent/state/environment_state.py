from dataclasses import dataclass


@dataclass
class EnvironmentState:
    """环境状态：为光照、噪声、温湿度等输入预留接口。"""

    light: int | None = None
    noise: int | None = None
    temperature: float | None = None
    humidity: float | None = None
