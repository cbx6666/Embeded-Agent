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

    @property
    def light(self) -> int | None:
        return self.light_lux

    @light.setter
    def light(self, value: int | None) -> None:
        self.light_lux = value

    @property
    def noise(self) -> int | None:
        return self.noise_db

    @noise.setter
    def noise(self, value: int | None) -> None:
        self.noise_db = value

    @property
    def temperature(self) -> float | None:
        return self.temperature_c

    @temperature.setter
    def temperature(self, value: float | None) -> None:
        self.temperature_c = value

    @property
    def humidity(self) -> float | None:
        return self.humidity_pct

    @humidity.setter
    def humidity(self, value: float | None) -> None:
        self.humidity_pct = value
