from __future__ import annotations

"""P1~P4 最小修复：defer 语义、准入冷却、reminder_last_ts、自检日志。"""

import json
import tempfile
import unittest
from pathlib import Path
from src.adapters.voice.arbitration.session_probe import VoiceSessionProbe
from src.agent.core import build_default_core
from src.agent.core.models import Event, Intent
from src.agent.guard.guard import Guard
from src.agent.policy_config import GuardPolicy
from src.agent.state.agent_state import AgentState
from tests.fakes.fake_llm_service import FakeLLMService
from tests.test_care_checks import _set_recent
from tests.test_periodic_and_scheduler import _state


class _ProbeResetMixin:
    def setUp(self) -> None:
        VoiceSessionProbe.global_probe().reset_for_tests()


class _CoreTestBase(_ProbeResetMixin, unittest.TestCase):
    def make_core(self, fake: FakeLLMService | None = None):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.fake = fake or FakeLLMService()
        log_lines: list[str] = []
        self.log_lines = log_lines

        class _Out:
            def execute(self, action):
                pass

            def show_text(self, line: str) -> None:
                log_lines.append(line)

        self.output = _Out()
        core = build_default_core(
            output=self.output,
            store_path=base / "state.json",
            profile_store_path=base / "profiles.json",
            memory_store_path=base / "memory.json",
            timer_background=False,
            llm_service=self.fake,
            memory_async=False,
        )
        self.addCleanup(core.shutdown)
        return core

    def _json_log(self, core) -> dict:
        for line in reversed(self.log_lines):
            if line.startswith("[自检-JSON] "):
                return json.loads(line[len("[自检-JSON] ") :])
        fields = getattr(core.last_decision_result, "log_fields", {}) or {}
        return dict(fields)


class BehaviorDistractionDeferTest(_CoreTestBase):
    def test_listening_defer_does_not_consume_admission_or_schedule(self) -> None:
        core = self.make_core()
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.state.interaction.dialogue_state = "listening"

        behavior = next(
            t for t in core.autonomous_scheduler.tasks if t.trigger == "behavior_distraction_check"
        )
        behavior.remaining_sec = 0.0
        behavior.due = True
        behavior.remaining_sec = 20.0
        behavior.due = False

        actions, _ = core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1000,
                payload={"trigger": "behavior_distraction_check", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(actions, [])
        log = self._json_log(core)
        self.assertTrue(log.get("deferred"))
        self.assertTrue(log.get("schedule_reverted"))
        self.assertFalse(log.get("admission_marked"))
        self.assertFalse(log.get("speak_action_generated"))
        self.assertNotIn("spoke", log)
        self.assertIsNone(core.state.cooldown.autonomous_check_last_ts.get("behavior_distraction_check"))
        self.assertTrue(behavior.due)
        self.assertAlmostEqual(behavior.remaining_sec, 0.0, places=3)


class WellnessDeferTest(_CoreTestBase):
    def _fatigue_state(self, core) -> None:
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.state.user.fatigue_level = "high"
        _set_recent(
            core.state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )

    def test_agent_thinking_defer_reverts_without_reminder_ts(self) -> None:
        fake = FakeLLMService()
        fake.set_response("wellness_care_check", {"reply": "歇会儿"})
        core = self.make_core(fake)
        self._fatigue_state(core)
        core.state.interaction.dialogue_state = "thinking"

        wellness = next(t for t in core.autonomous_scheduler.tasks if t.trigger == "wellness_care_check")
        wellness.remaining_sec = 0.0
        wellness.due = True
        wellness.remaining_sec = 30.0
        wellness.due = False

        actions, _ = core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1020,
                payload={"trigger": "wellness_care_check", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(actions, [])
        log = self._json_log(core)
        self.assertTrue(log.get("deferred"))
        self.assertTrue(log.get("schedule_reverted"))
        self.assertFalse(log.get("admission_marked"))
        self.assertNotIn("posture_reminder", core.state.cooldown.reminder_last_ts)
        self.assertNotIn("rest_reminder", core.state.cooldown.reminder_last_ts)
        self.assertTrue(wellness.due)

    def test_media_playing_defer_reverts(self) -> None:
        fake = FakeLLMService()
        fake.set_response("wellness_care_check", {"reply": "歇会儿"})
        core = self.make_core(fake)
        self._fatigue_state(core)
        probe = VoiceSessionProbe.global_probe()
        probe.bind_media_playing(lambda: True)

        wellness = next(t for t in core.autonomous_scheduler.tasks if t.trigger == "wellness_care_check")
        wellness.remaining_sec = 0.0
        wellness.due = True
        wellness.remaining_sec = 30.0
        wellness.due = False

        core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1020,
                payload={"trigger": "wellness_care_check", "source": "agent_autonomy"},
            )
        )
        log = self._json_log(core)
        self.assertTrue(log.get("deferred"))
        self.assertTrue(log.get("schedule_reverted"))
        self.assertIsNone(core.state.cooldown.autonomous_check_last_ts.get("wellness_care_check"))


class SensorDeferSemanticsTest(_CoreTestBase):
    def test_listening_defer_reverts_without_admission(self) -> None:
        core = self.make_core()
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.handle_event(
            Event(
                type="temperature_humidity_updated",
                timestamp=2,
                payload={"temperature_c": 25.0, "humidity_pct": 50},
            )
        )
        core.state.interaction.dialogue_state = "listening"

        sensor = next(t for t in core.autonomous_scheduler.tasks if t.trigger == "sensor_status_report")
        sensor.remaining_sec = 0.0
        sensor.due = True
        sensor.remaining_sec = 300.0
        sensor.due = False

        actions, _ = core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1000,
                payload={"trigger": "sensor_status_report", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(actions, [])
        log = self._json_log(core)
        self.assertEqual(log.get("check_outcome"), "deferred_revert")
        self.assertTrue(log.get("deferred"))
        self.assertTrue(log.get("schedule_reverted"))
        self.assertFalse(log.get("admission_marked"))
        self.assertTrue(sensor.due)


class EnvironmentNoOpVsDeferTest(_CoreTestBase):
    def _low_light_setup(self, core) -> None:
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.handle_event(
            Event(
                type="light_level_updated",
                timestamp=2,
                payload={"light_lux": 120, "light_level": "low"},
            )
        )

    def test_llm_no_op_marks_admitted(self) -> None:
        fake = FakeLLMService()
        fake.set_response("environment_care_check", {"intent": "no_op", "reply": ""})
        core = self.make_core(fake)
        self._low_light_setup(core)

        core.handle_event(
            Event(
                type="system_triggered",
                timestamp=2000,
                payload={"trigger": "environment_care_check", "source": "agent_autonomy"},
            )
        )
        log = self._json_log(core)
        self.assertFalse(log.get("deferred"))
        self.assertTrue(log.get("admission_marked"))
        self.assertIsNotNone(core.state.cooldown.autonomous_check_last_ts.get("environment_care_check"))

    def test_voice_session_defer_skips_admission_not_revert(self) -> None:
        fake = FakeLLMService()
        fake.set_response("environment_care_check", {"intent": "adjust_environment_feedback", "reply": "x"})
        core = self.make_core(fake)
        self._low_light_setup(core)
        core.state.interaction.dialogue_state = "listening"

        env = next(t for t in core.autonomous_scheduler.tasks if t.trigger == "environment_care_check")
        env.remaining_sec = 0.0
        env.due = True
        env.remaining_sec = 60.0
        env.due = False

        core.handle_event(
            Event(
                type="system_triggered",
                timestamp=2000,
                payload={"trigger": "environment_care_check", "source": "agent_autonomy"},
            )
        )
        log = self._json_log(core)
        self.assertTrue(log.get("deferred"))
        self.assertFalse(log.get("schedule_reverted"))
        self.assertFalse(log.get("admission_marked"))
        self.assertFalse(env.due)


class ReminderLastTsOnTtsFinishedTest(_CoreTestBase):
    def test_reminder_ts_only_after_tts_finished_with_voice_runtime(self) -> None:
        fake = FakeLLMService()
        fake.set_response("wellness_care_check", {"reply": "坐直一点"})
        core = self.make_core(fake)
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.state.user.posture = "slouching"
        _set_recent(
            core.state,
            "posture",
            [(t, "slouching", 0.9) for t in range(1000, 1061, 10)],
        )
        core.device_adapter.voice_runtime = object()

        core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1060,
                payload={"trigger": "wellness_care_check", "source": "agent_autonomy"},
            )
        )
        self.assertNotIn("posture_reminder", core.state.cooldown.reminder_last_ts)

        guard = Guard(GuardPolicy())
        allowed, _ = guard.filter(
            [Intent("suggest_rest", "test", payload={"reason": "posture_reminder"})],
            state=core.state,
            timestamp=1061,
        )
        self.assertTrue(allowed)

        core.handle_event(
            Event(
                type="tts_finished",
                timestamp=1100,
                payload={
                    "text": "坐直一点",
                    "kind": "notification",
                    "reason": "posture_reminder",
                },
            )
        )
        self.assertEqual(core.state.cooldown.reminder_last_ts.get("posture_reminder"), 1100)

        allowed_after, _ = guard.filter(
            [Intent("suggest_rest", "test", payload={"reason": "posture_reminder"})],
            state=core.state,
            timestamp=1150,
        )
        self.assertFalse(allowed_after)

        core.handle_event(
            Event(
                type="tts_finished",
                timestamp=1200,
                payload={
                    "text": "坐直一点",
                    "kind": "notification",
                    "reason": "posture_reminder",
                    "cancelled": True,
                },
            )
        )
        self.assertEqual(core.state.cooldown.reminder_last_ts.get("posture_reminder"), 1100)


class AutonomousCheckLogFieldsTest(_CoreTestBase):
    def test_speak_action_not_logged_as_spoke(self) -> None:
        fake = FakeLLMService()
        fake.set_response("wellness_care_check", {"reply": "累了就歇会儿"})
        core = self.make_core(fake)
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.state.user.fatigue_level = "high"
        _set_recent(
            core.state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )

        core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1020,
                payload={"trigger": "wellness_care_check", "source": "agent_autonomy"},
            )
        )
        log = self._json_log(core)
        self.assertTrue(log.get("speak_action_generated"))
        self.assertNotIn("spoke", log)

    def test_defer_logs_structured_fields(self) -> None:
        core = self.make_core()
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.state.interaction.dialogue_state = "listening"

        core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1000,
                payload={"trigger": "behavior_distraction_check", "source": "agent_autonomy"},
            )
        )
        log = self._json_log(core)
        self.assertTrue(log.get("deferred"))
        self.assertTrue(log.get("schedule_reverted"))
        self.assertFalse(log.get("speak_action_generated"))


if __name__ == "__main__":
    unittest.main()
