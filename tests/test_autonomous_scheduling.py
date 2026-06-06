from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.config.policy_config import AutonomousScheduleConfig
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.memory.memory_task import MemorySubmitResult
from src.agent.scheduling import AutonomousScheduler
from src.agent.state import AgentState
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore
from tests.fakes.fake_llm_service import FakeLLMService


class _DisabledMemoryWorker:
    def submit_event_memory(self, **kwargs):
        del kwargs
        return MemorySubmitResult(False, None, "disabled_for_test", 0)

    def submit_action_memory(self, **kwargs):
        del kwargs
        return MemorySubmitResult(False, None, "disabled_for_test", 0)

    def shutdown(self, timeout=None):
        del timeout


class AutonomousCheckPolicyIntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.llm = FakeLLMService()
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=self.llm,
            store=JsonStore(Path(self.temp_dir.name) / "runtime.json"),
            memory_worker=_DisabledMemoryWorker(),  # type: ignore[arg-type]
        )

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_focus_check_skips_without_active_focus(self) -> None:
        actions, _ = self.core.handle_event(_check("focus_health_check", timestamp=1000))

        self.assertEqual(actions, [])
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(
            self.core.last_decision_result.stage_metadata["autonomous_check"]["reason"],
            "focus_not_active",
        )

    def test_single_sustained_focus_abnormality_uses_rule(self) -> None:
        self._prepare_focus()
        self._record_signal("user_fatigue_updated", "fatigue_level", "high")

        actions, _ = self.core.handle_event(_check("focus_health_check", timestamp=1000))

        self.assertEqual(self.llm.calls, [])
        self.assertEqual([intent.type for intent in self.core.last_intents], ["suggest_rest"])
        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in actions))
        self.assertEqual(
            self.core.last_decision_result.stage_metadata["decision_source"],
            "autonomous_check_policy",
        )

    def test_multiple_abnormalities_use_llm_once_per_check_cooldown(self) -> None:
        self._prepare_focus()
        self._record_signal("user_fatigue_updated", "fatigue_level", "high")
        self._record_signal("user_attention_updated", "attention", "distracted")

        self.core.handle_event(_check("focus_health_check", timestamp=1000))
        first_call_count = len(self.llm.calls)
        self.core.handle_event(_check("focus_health_check", timestamp=1020))

        self.assertGreater(first_call_count, 0)
        self.assertEqual(len(self.llm.calls), first_call_count)
        self.assertEqual(
            self.core.last_decision_result.stage_metadata["autonomous_check"]["reason"],
            "autonomous_check_cooldown_active",
        )

    def test_environment_check_requires_work_context_and_sustained_trend(self) -> None:
        self.core.state.user.presence = "present"
        self._record_signal("noise_level_updated", "level", "high")

        skipped, _ = self.core.handle_event(_check("environment_check", timestamp=1000))
        self.core.state.focus.active = True
        allowed, _ = self.core.handle_event(_check("environment_check", timestamp=1001))

        self.assertEqual(skipped, [])
        self.assertTrue(any(action.type == "set_light_state" for action in allowed))
        self.assertEqual(self.llm.calls, [])

    def test_periodic_check_is_closed_by_default(self) -> None:
        actions, _ = self.core.handle_event(_check("periodic_check", timestamp=1000))

        self.assertEqual(actions, [])
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(
            self.core.last_decision_result.stage_metadata["autonomous_check"]["reason"],
            "periodic_check_disabled",
        )

    def test_untrusted_source_cannot_enter_llm(self) -> None:
        self._prepare_focus()
        self._record_signal("user_fatigue_updated", "fatigue_level", "high")

        event = _check("focus_health_check", timestamp=1000)
        event.payload["source"] = "manual"
        actions, _ = self.core.handle_event(event)

        self.assertEqual(actions, [])
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(
            self.core.last_decision_result.stage_metadata["autonomous_check"]["reason"],
            "untrusted_autonomous_source",
        )

    def _prepare_focus(self) -> None:
        self.core.state.focus.active = True
        self.core.state.focus.elapsed_sec = 600
        self.core.state.focus.remaining_sec = 900
        self.core.state.user.presence = "present"

    def _record_signal(self, event_type: str, key: str, value: str) -> None:
        for timestamp in (100, 101, 102):
            payload = {key: value, "confidence": 0.9}
            if event_type == "user_attention_updated":
                payload["behavior"] = "phone_use"
            self.core.handle_event(
                Event(type=event_type, timestamp=timestamp, payload=payload)
            )


class AutonomousSchedulerTestCase(unittest.TestCase):
    def test_scheduler_emits_only_when_configured_interval_is_due(self) -> None:
        state = AgentState()
        emitted: list[Event] = []
        scheduler = AutonomousScheduler(
            state_provider=lambda: state,
            event_sink=emitted.append,
            config=AutonomousScheduleConfig(
                event_source="test_scheduler",
                poll_interval_sec=0.01,
                intervals_sec={
                    "focus_health_check": 10,
                    "environment_check": 20,
                    "periodic_check": 1,
                },
                disabled_triggers=frozenset({"periodic_check"}),
            ),
        )

        self.assertEqual(scheduler.run_due(100), [])
        self.assertEqual(scheduler.run_due(109), [])
        self.assertEqual(
            [event.payload["trigger"] for event in scheduler.run_due(110)],
            ["focus_health_check"],
        )
        self.assertEqual(
            {event.payload["trigger"] for event in scheduler.run_due(120)},
            {"focus_health_check", "environment_check"},
        )
        self.assertNotIn("periodic_check", {event.payload["trigger"] for event in emitted})
        self.assertTrue(
            all(event.payload["source"] == "test_scheduler" for event in emitted)
        )

    def test_stop_prevents_future_emission(self) -> None:
        emitted: list[Event] = []
        scheduler = AutonomousScheduler(
            state_provider=AgentState,
            event_sink=emitted.append,
            config=AutonomousScheduleConfig(
                intervals_sec={"focus_health_check": 1},
                disabled_triggers=frozenset(),
            ),
        )
        scheduler.run_due(1)
        scheduler.stop()

        self.assertEqual(scheduler.run_due(10), [])
        self.assertEqual(emitted, [])


def _check(trigger: str, *, timestamp: int) -> Event:
    return Event(
        type="system_triggered",
        timestamp=timestamp,
        payload={"trigger": trigger, "source": "agent_autonomy"},
    )


if __name__ == "__main__":
    unittest.main()
