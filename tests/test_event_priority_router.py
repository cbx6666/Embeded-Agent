from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import get_args

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event, EventPriorityRouter
from src.agent.event.types import EventType
from src.agent.memory.memory_task import MemorySubmitResult
from src.agent.state import AgentState
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore
from tests.fakes.fake_llm_service import FakeLLMService


class _FailOnDecisionPipeline:
    def decide(self, **kwargs):
        del kwargs
        raise AssertionError("DecisionPipeline must not be called for state-only events")


class _CapturingMemoryWorker:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def submit_event_memory(self, **kwargs):
        self.events.append(kwargs["event"])
        return MemorySubmitResult(True, "feedback-task", "accepted", 1)

    def submit_action_memory(self, **kwargs):
        del kwargs
        return MemorySubmitResult(False, None, "no_actions", 0)

    def shutdown(self, timeout=None):
        del timeout


class EventPriorityRouterTestCase(unittest.TestCase):
    def test_every_event_type_has_an_explicit_route(self) -> None:
        router = EventPriorityRouter()
        for event_type in get_args(EventType):
            payload = {"trigger": "periodic_check"} if event_type == "system_triggered" else {}
            route = router.classify(Event(type=event_type, timestamp=1, payload=payload))
            self.assertFalse(route.reason.startswith("unclassified_event:"), event_type)

    def test_route_groups_are_orthogonal_to_llm_permission(self) -> None:
        router = EventPriorityRouter()

        semantic = router.classify(Event(type="user_text_input", timestamp=1, payload={"text": "hello"}))
        structured = router.classify(Event(type="focus_start_requested", timestamp=1))
        state_only = router.classify(Event(type="user_fatigue_updated", timestamp=1))
        telemetry = router.classify(Event(type="timer_ticked", timestamp=1))
        profile = router.classify(Event(type="user_profile_updated", timestamp=1))
        internal = router.classify(
            Event(type="system_triggered", timestamp=1, payload={"trigger": "focus_timer_started"})
        )

        self.assertEqual((semantic.priority, semantic.handling), ("P0", "orchestrator"))
        self.assertTrue(semantic.should_allow_llm)
        self.assertEqual((structured.priority, structured.handling), ("P0", "rule_intent_builder"))
        self.assertTrue(structured.should_enter_decision)
        self.assertFalse(structured.should_allow_llm)
        self.assertEqual((state_only.priority, state_only.handling), ("P2", "state_only"))
        self.assertEqual((telemetry.priority, telemetry.handling), ("P3", "state_only"))
        self.assertEqual((profile.priority, profile.handling), ("P4", "profile_handler"))
        self.assertEqual((internal.priority, internal.handling), ("P4", "internal_only"))

    def test_signal_trend_window_is_bounded_to_policy_limit(self) -> None:
        state = AgentState()
        service = RuntimeHistoryService()
        for timestamp in range(60):
            service.record_event(
                state,
                Event(
                    type="user_fatigue_updated",
                    timestamp=timestamp,
                    payload={
                        "fatigue_level": "moderate" if timestamp % 2 else "mild",
                        "confidence": 0.75,
                    },
                ),
            )

        trend = state.runtime_history.signal_trends["fatigue"]
        self.assertEqual(len(trend["recent_values"]), 50)
        self.assertEqual(sum(trend["value_counts"].values()), 50)
        self.assertEqual(trend["confidence_summary"]["count"], 50)


class EventRoutingCoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _core(self, *, decision_pipeline=None) -> AgentCore:
        return AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=FakeLLMService(reply_text="fallback reply"),
            store=JsonStore(self.root / "runtime.json"),
            decision_pipeline=decision_pipeline,
        )

    def test_timer_ticks_reduce_save_and_skip_decision_pipeline(self) -> None:
        core = self._core(decision_pipeline=_FailOnDecisionPipeline())
        try:
            core.state.focus.active = True
            core.state.focus.start_ts = 100
            core.state.focus.target_duration_sec = 600
            core.state.focus.remaining_sec = 600

            for offset in range(1, 11):
                actions, results = core.handle_event(
                    Event(
                        type="timer_ticked",
                        timestamp=100 + offset,
                        payload={"remaining_sec": 600 - offset, "timer": "focus"},
                    )
                )
                self.assertEqual(actions, [])
                self.assertEqual(results, [])

            self.assertEqual(core.state.focus.remaining_sec, 590)
            self.assertEqual(core.state.focus.elapsed_sec, 10)
            saved = AgentState.from_dict(core.store.load_state_dict())
            self.assertEqual(saved.focus.remaining_sec, 590)
            route_entry = core.last_runtime_trace.find("event_route", "classified")[0]
            self.assertEqual(route_entry.payload["event_route_priority"], "P3")
            self.assertTrue(route_entry.payload["decision_skipped_by_router"])
            self.assertTrue(core.last_runtime_trace.find("decision_pipeline", "skipped"))
        finally:
            core.shutdown()

    def test_fatigue_updates_preserve_bounded_trend_and_skip_decision(self) -> None:
        core = self._core(decision_pipeline=_FailOnDecisionPipeline())
        values = ["mild", "mild", "moderate", "moderate", "moderate"] * 2
        try:
            for index, value in enumerate(values, start=1):
                core.handle_event(
                    Event(
                        type="user_fatigue_updated",
                        timestamp=index,
                        payload={"fatigue_level": value, "confidence": 0.8},
                    )
                )

            trend = core.state.runtime_history.signal_trends["fatigue"]
            self.assertEqual(core.state.user.fatigue_level, "moderate")
            self.assertEqual(trend["current"], "moderate")
            self.assertEqual(trend["previous"], "moderate")
            self.assertEqual(trend["consecutive_same_count"], 3)
            self.assertEqual(len(trend["recent_values"]), 10)
            self.assertEqual(trend["value_counts"], {"mild": 4, "moderate": 6})
            self.assertEqual(trend["confidence_summary"]["average"], 0.8)
            saved = AgentState.from_dict(core.store.load_state_dict())
            self.assertEqual(
                saved.runtime_history.signal_trends["fatigue"]["current"],
                "moderate",
            )
        finally:
            core.shutdown()

    def test_low_frequency_check_context_contains_state_trends(self) -> None:
        core = self._core()
        try:
            core.state.focus.active = True
            core.state.focus.elapsed_sec = 600
            core.state.user.presence = "present"
            for timestamp in (10, 11, 12):
                core.handle_event(
                    Event(
                        type="user_fatigue_updated",
                        timestamp=timestamp,
                        payload={"fatigue_level": "high", "confidence": 0.9},
                    )
                )
            for timestamp in (13, 14, 15):
                core.handle_event(
                    Event(
                        type="user_attention_updated",
                        timestamp=timestamp,
                        payload={"attention": "distracted", "behavior": "phone_use", "confidence": 0.7},
                    )
                )
            core.handle_event(
                Event(
                    type="system_triggered",
                    timestamp=700,
                    payload={"trigger": "focus_health_check", "source": "agent_autonomy"},
                )
            )

            context = core.last_decision_result.stage_metadata["context"]
            trends = context["state"]["trends"]
            self.assertEqual(trends["fatigue"]["current"], "high")
            self.assertEqual(trends["attention"]["current"], "distracted")
            self.assertEqual(trends["fatigue"]["value_ratios"], {"high": 1.0})
            self.assertNotIn("recent_values", trends["fatigue"])
            self.assertEqual(
                context["personal_context"]["runtime_history"]["fatigue_summary"]["current"],
                "high",
            )
            route_entry = core.last_runtime_trace.find("event_route", "classified")[0]
            self.assertEqual(route_entry.payload["event_route_priority"], "P1")
        finally:
            core.shutdown()

    def test_semantic_event_uses_llm_and_structured_event_uses_rules(self) -> None:
        core = self._core()
        try:
            text_actions, _ = core.handle_event(
                Event(type="user_text_input", timestamp=20, payload={"text": "hello"})
            )
            self.assertIn("speak", {action.type for action in text_actions})
            self.assertTrue(core.last_decision_result.used_llm)

            focus_actions, _ = core.handle_event(
                Event(type="focus_start_requested", timestamp=21, payload={"duration_sec": 600})
            )
            self.assertIn("start_timer", {action.type for action in focus_actions})
            self.assertFalse(core.last_decision_result.used_llm)
            self.assertEqual(
                core.last_decision_result.stage_metadata["decision_source"],
                "rule_intent_builder",
            )
            route = core.last_decision_result.stage_metadata["event_route"]
            self.assertEqual(route["event_route_priority"], "P0")
            self.assertFalse(route["should_allow_llm"])
        finally:
            core.shutdown()

    def test_profile_event_is_skipped_with_dedicated_handler_trace(self) -> None:
        core = self._core(decision_pipeline=_FailOnDecisionPipeline())
        try:
            actions, _ = core.handle_event(
                Event(type="user_profile_updated", timestamp=30, payload={"source": "test"})
            )

            self.assertEqual(actions, [])
            route_entry = core.last_runtime_trace.find("event_route", "classified")[0]
            self.assertEqual(route_entry.payload["event_route_priority"], "P4")
            self.assertEqual(route_entry.payload["event_route_handling"], "profile_handler")
            self.assertTrue(route_entry.payload["requires_dedicated_handler"])
        finally:
            core.shutdown()

    def test_break_feedback_skips_decision_but_enters_async_memory(self) -> None:
        worker = _CapturingMemoryWorker()
        core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=FakeLLMService(),
            store=JsonStore(self.root / "feedback-runtime.json"),
            decision_pipeline=_FailOnDecisionPipeline(),
            memory_worker=worker,  # type: ignore[arg-type]
        )
        try:
            actions, _ = core.handle_event(
                Event(
                    type="break_suggestion_rejected",
                    timestamp=40,
                    payload={"source": "user_feedback"},
                )
            )

            self.assertEqual(actions, [])
            self.assertEqual([event.type for event in worker.events], ["break_suggestion_rejected"])
            route_entry = core.last_runtime_trace.find("event_route", "classified")[0]
            self.assertEqual(route_entry.payload["event_route_handling"], "feedback_signal")
            self.assertTrue(route_entry.payload["decision_skipped_by_router"])
        finally:
            core.shutdown()

    def test_voice_setting_event_uses_dedicated_profile_handler(self) -> None:
        core = self._core(decision_pipeline=_FailOnDecisionPipeline())
        try:
            actions, _ = core.handle_event(
                Event(
                    type="voice_volume_changed",
                    timestamp=50,
                    payload={"volume": 35},
                )
            )

            self.assertEqual(actions, [])
            profile = core.personal_context_builder.user_profile_service.profile_context(
                core.state.current_user_id
            )
            self.assertEqual(profile["preference"]["tts_volume"], 35.0)
            self.assertTrue(
                core.last_runtime_trace.find("dedicated_event_handler", "handled")
            )
        finally:
            core.shutdown()


if __name__ == "__main__":
    unittest.main()
