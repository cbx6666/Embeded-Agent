from __future__ import annotations



"""ESP32 / STM32 USB 串口环境传感器 → 标准 environment Event。



支持上行格式：

- ESP32 JSON：temperature, humidity, lux, noise_db

- STM32 JSON：temperature_c, humidity_pct, light_lux, noise_db

- Legacy 文本：Light: 123.4 lx

"""



import logging

import os

import threading

import time

from typing import Any, Protocol



from src.adapters.environment.levels import (

    EnvironmentThresholdConfig,

    humidity_level_from_pct,

    light_level_from_lux,

    noise_level_from_db,

    resolve_environment_thresholds,

    temperature_level_from_c,

)

from src.adapters.environment.parser import (

    EnvironmentSensorReading,

    parse_environment_sensor_line,

    parse_esp32_sensor_line,

)

from src.adapters.environment.serial_reader import open_line_serial_reader

from src.adapters.perception_config import perception_interval_sec

from src.agent.event.event_builders import (

    make_light_level_event,

    make_noise_level_event,

    make_temperature_humidity_event,

)



logger = logging.getLogger(__name__)



DEFAULT_ESP32_PORT = "/dev/ttyUSB0"

DEFAULT_ESP32_BAUD = 115200

DEFAULT_SOURCE = "esp32_usb_v1"



# 兼容旧导出名

Esp32SensorReading = EnvironmentSensorReading





class EventEmitSink(Protocol):

    def handle_event(self, event: Any) -> Any:

        ...





def resolve_esp32_port(explicit: str | None = None) -> str:

    return (explicit or os.environ.get("EMBED_ESP32_SENSOR_PORT") or DEFAULT_ESP32_PORT).strip()





def _resolve_baudrate(explicit: int | None = None) -> int:

    if explicit is not None:

        return int(explicit)

    raw = os.environ.get("EMBED_ESP32_SENSOR_BAUD", "").strip()

    if raw:

        try:

            return int(raw)

        except ValueError:

            pass

    return DEFAULT_ESP32_BAUD





def esp32_port_available(port: str | None = None) -> bool:

    path = resolve_esp32_port(port)

    return os.path.exists(path)





class Esp32EnvironmentAdapter:

    """后台读串口 JSON/文本行，映射为 environment Event。"""



    def __init__(

        self,

        sink: EventEmitSink,

        *,

        port: str | None = None,

        baudrate: int | None = None,

        source: str = DEFAULT_SOURCE,

        min_emit_interval_sec: float | None = None,

        thresholds: EnvironmentThresholdConfig | None = None,

        serial_timeout_sec: float = 2.0,

    ) -> None:

        self._sink = sink

        self._port = resolve_esp32_port(port)

        self._baudrate = _resolve_baudrate(baudrate)

        self._source = source

        self._thresholds = thresholds or resolve_environment_thresholds()

        self._serial_timeout = float(serial_timeout_sec)

        self._min_emit_interval = (

            perception_interval_sec() if min_emit_interval_sec is None else float(min_emit_interval_sec)

        )

        self._stop = threading.Event()

        self._thread: threading.Thread | None = None

        self._last_emit_mono = 0.0

        self._merged_reading = EnvironmentSensorReading()



    def start_background(self) -> None:

        if self._thread is not None and self._thread.is_alive():

            return

        if not esp32_port_available(self._port):

            raise FileNotFoundError(f"ESP32 传感器串口不存在: {self._port}")

        self._stop.clear()

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="esp32-env")

        self._thread.start()



    def stop(self) -> None:

        self._stop.set()

        if self._thread is not None:

            self._thread.join(timeout=3.0)

            self._thread = None



    def _should_emit(self, reading: EnvironmentSensorReading) -> bool:

        now = time.monotonic()

        if now - self._last_emit_mono < self._min_emit_interval:

            return False

        prev = self._merged_reading

        if not prev.has_any_field():

            return True

        if (

            _delta(prev.temperature_c, reading.temperature_c) >= 0.2

            or _delta(prev.humidity_pct, reading.humidity_pct) >= 1.0

            or _delta(prev.light_lux, reading.light_lux) >= 5.0

            or _delta(prev.noise_db, reading.noise_db) >= 3.0

        ):

            return True

        return now - self._last_emit_mono >= 1.0



    def _emit_reading(self, reading: EnvironmentSensorReading) -> None:

        ts = int(time.time())

        cfg = self._thresholds

        events = []



        if reading.temperature_c is not None and reading.humidity_pct is not None:

            events.append(

                make_temperature_humidity_event(

                    temperature_c=reading.temperature_c,

                    humidity_pct=reading.humidity_pct,

                    source=f"{self._source}:temp_humidity",

                    temperature_level=temperature_level_from_c(reading.temperature_c, cfg),

                    humidity_level=humidity_level_from_pct(reading.humidity_pct, cfg),

                    timestamp=ts,

                )

            )



        if reading.light_lux is not None:

            light_level, is_low_light = light_level_from_lux(reading.light_lux, cfg)

            events.append(

                make_light_level_event(

                    light_lux=max(0, int(round(reading.light_lux))),

                    source=f"{self._source}:lux",

                    level=light_level,

                    is_low_light=is_low_light,

                    timestamp=ts,

                )

            )



        if reading.noise_db is not None:

            noise_level, is_noisy = noise_level_from_db(reading.noise_db, cfg)

            events.append(

                make_noise_level_event(

                    noise_db=max(0, int(round(reading.noise_db))),

                    source=f"{self._source}:noise",

                    level=noise_level,

                    is_noisy=is_noisy,

                    timestamp=ts,

                )

            )



        for event in events:

            try:

                self._sink.handle_event(event)

            except Exception as exc:

                logger.warning("environment event emit failed: %s", exc)



        self._last_emit_mono = time.monotonic()



    def _run_loop(self) -> None:

        ser = open_line_serial_reader(self._port, self._baudrate, self._serial_timeout)

        try:

            ser.open()

            logger.info("environment sensor on %s @ %d", self._port, self._baudrate)

        except OSError as exc:

            logger.error("无法打开环境传感器串口 %s: %s", self._port, exc)

            return

        try:

            while not self._stop.is_set():

                line = ser.readline(timeout=0.5)

                if not line:

                    continue

                partial = parse_environment_sensor_line(line)

                if partial is None:

                    continue

                self._merged_reading = self._merged_reading.merge(partial)

                if self._should_emit(self._merged_reading):

                    self._emit_reading(self._merged_reading)

        finally:

            ser.close()





def _delta(previous: float | None, current: float | None) -> float:

    if previous is None or current is None:

        return 0.0

    return abs(previous - current)


