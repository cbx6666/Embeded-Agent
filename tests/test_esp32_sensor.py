from __future__ import annotations

import json
import unittest

from src.adapters.environment.levels import (
    EnvironmentThresholdConfig,
    humidity_level_from_pct,
    light_level_from_lux,
    noise_level_from_db,
    temperature_level_from_c,
)
from src.adapters.environment.parser import (
    EnvironmentSensorReading,
    parse_environment_sensor_line,
    parse_esp32_sensor_line,
)
from src.agent.event.event_builders import (
    make_light_level_event,
    make_noise_level_event,
    make_temperature_humidity_event,
)
from src.agent.event.event_model import Event
from src.agent.event.router import EventRouter
from src.agent.state import AgentState
from src.agent.state.reducer import reduce_state


class Esp32ParseTest(unittest.TestCase):
    def test_parse_live_format(self) -> None:
        line = '{"temperature":25.90,"humidity":61.50,"lux":0.00,"noise_db":70.33}'
        reading = parse_esp32_sensor_line(line)
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertAlmostEqual(reading.temperature_c, 25.9)
        self.assertAlmostEqual(reading.humidity_pct, 61.5)
        self.assertAlmostEqual(reading.light_lux, 0.0)
        self.assertAlmostEqual(reading.noise_db, 70.33)

    def test_parse_stm32_field_names(self) -> None:
        line = json.dumps(
            {
                "temperature_c": 24.5,
                "humidity_pct": 42.0,
                "light_lux": 88.0,
                "noise_db": 55.0,
            }
        )
        reading = parse_esp32_sensor_line(line)
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertAlmostEqual(reading.temperature_c, 24.5)
        self.assertAlmostEqual(reading.humidity_pct, 42.0)
        self.assertAlmostEqual(reading.light_lux, 88.0)
        self.assertAlmostEqual(reading.noise_db, 55.0)

    def test_parse_legacy_light_line(self) -> None:
        reading = parse_environment_sensor_line("Light: 150.5 lx")
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertAlmostEqual(reading.light_lux, 150.5)
        self.assertIsNone(reading.temperature_c)

    def test_partial_json_only_light(self) -> None:
        reading = parse_environment_sensor_line('{"lux": 10.0}')
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertAlmostEqual(reading.light_lux, 10.0)
        self.assertIsNone(parse_esp32_sensor_line('{"lux": 10.0}'))

    def test_merge_partial_readings(self) -> None:
        merged = EnvironmentSensorReading()
        merged = merged.merge(parse_environment_sensor_line('{"temperature":26.0,"humidity":55.0}') or merged)
        merged = merged.merge(parse_environment_sensor_line("Light: 120.0 lx") or merged)
        self.assertAlmostEqual(merged.temperature_c, 26.0)
        self.assertAlmostEqual(merged.humidity_pct, 55.0)
        self.assertAlmostEqual(merged.light_lux, 120.0)

    def test_reject_invalid(self) -> None:
        self.assertIsNone(parse_esp32_sensor_line(""))
        self.assertIsNone(parse_esp32_sensor_line("not json"))


class EnvironmentLevelsTest(unittest.TestCase):
    def test_levels_use_configurable_thresholds(self) -> None:
        cfg = EnvironmentThresholdConfig(low_light_lux=100.0, noisy_db=60.0, dry_humidity_pct=35.0)
        self.assertEqual(light_level_from_lux(20.0, cfg)[0], "dark")
        self.assertEqual(light_level_from_lux(80.0, cfg)[0], "low")
        self.assertEqual(noise_level_from_db(62.0, cfg)[0], "high")
        self.assertEqual(humidity_level_from_pct(20.0, cfg), "dry")
        self.assertEqual(temperature_level_from_c(10.0, cfg), "low")


class Esp32ReducerTest(unittest.TestCase):
    def test_events_update_environment_state(self) -> None:
        state = AgentState()
        reading = parse_esp32_sensor_line(
            json.dumps(
                {
                    "temperature": 26.0,
                    "humidity": 55.0,
                    "lux": 120.0,
                    "noise_db": 50.0,
                }
            )
        )
        assert reading is not None
        light_level, is_low = light_level_from_lux(reading.light_lux)
        noise_level, is_noisy = noise_level_from_db(reading.noise_db)
        reduce_state(
            state,
            make_temperature_humidity_event(
                temperature_c=reading.temperature_c,
                humidity_pct=reading.humidity_pct,
                source="esp32_usb_v1",
            ),
        )
        reduce_state(
            state,
            make_light_level_event(
                light_lux=int(reading.light_lux),
                source="esp32_usb_v1",
                level=light_level,
                is_low_light=is_low,
            ),
        )
        reduce_state(
            state,
            make_noise_level_event(
                noise_db=int(reading.noise_db),
                source="esp32_usb_v1",
                level=noise_level,
                is_noisy=is_noisy,
            ),
        )
        self.assertEqual(state.environment.temperature_c, 26.0)
        self.assertEqual(state.environment.humidity_pct, 55.0)
        self.assertEqual(state.environment.light_lux, 120)
        self.assertEqual(state.environment.noise_db, 50)


class EnvironmentRoutingTest(unittest.TestCase):
    def test_environment_events_route_state_only(self) -> None:
        router = EventRouter()
        for event_type in (
            "light_level_updated",
            "temperature_humidity_updated",
            "noise_level_updated",
        ):
            decision = router.classify(Event(type=event_type, timestamp=0, payload={}))
            self.assertEqual(decision.kind, "state_only")
            self.assertFalse(decision.uses_llm)


if __name__ == "__main__":
    unittest.main()
