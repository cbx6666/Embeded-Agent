from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.memory.memory_background_worker import MemoryBackgroundWorker
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryRunResult
from src.agent.memory.memory_candidate import MemoryCandidate
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore
from src.storage.long_term_memory_store import LongTermMemoryStore
from tests.fakes.fake_llm_service import FakeLLMService


class _BlockingEventMemoryPipeline:
    def __init__(self, store: LongTermMemoryStore) -> None:
        self.store = store
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def process_event(self, user_id, event, state, llm_service):
        del state, llm_service
        self.started.set()
        self.release.wait(timeout=2)
        stored = self.store.upsert_candidate(
            user_id,
            MemoryCandidate(
                memory_type="behavior_pattern",
                content="用户更喜欢一小时专注。",
                confidence=0.8,
                evidence=[
                    {
                        "source_event_type": event.type,
                        "timestamp": event.timestamp,
                        "source": "dialogue",
                        "user_text": event.payload.get("text"),
                    }
                ],
            ),
            timestamp=event.timestamp,
        )
        self.finished.set()
        return LongTermMemoryRunResult(stored=[stored])

    def process_actions(self, user_id, actions, timestamp, **kwargs):
        del user_id, actions, timestamp, kwargs
        return LongTermMemoryRunResult()


class _FailingEventMemoryPipeline:
    def __init__(self, store: LongTermMemoryStore) -> None:
        self.store = store
        self.attempted = threading.Event()

    def process_event(self, user_id, event, state, llm_service):
        del user_id, event, state, llm_service
        self.attempted.set()
        raise RuntimeError("scripted event memory failure")

    def process_actions(self, user_id, actions, timestamp, **kwargs):
        del user_id, actions, timestamp, kwargs
        return LongTermMemoryRunResult()


class AgentCoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.llm = FakeLLMService(reply_text="fallback reply")
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=self.llm,
            store=JsonStore(self.root / "runtime_store.json"),
        )

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_focus_start_updates_state_and_realizes_timer_action(self) -> None:
        actions, results = self.core.handle_event(
            Event(type="focus_start_requested", timestamp=1000, payload={"duration_sec": 600, "source": "test"})
        )

        self.assertTrue(self.core.state.focus.active)
        self.assertEqual(self.core.state.focus.target_duration_sec, 600)
        self.assertIn("start_timer", {action.type for action in actions})
        self.assertTrue(all(result.success for result in results))

    def test_user_text_runs_llm_roles_and_records_messages(self) -> None:
        actions, _ = self.core.handle_event(
            Event(type="user_text_input", timestamp=2000, payload={"text": "hello", "source": "test"})
        )

        self.assertIn("situation_analyst", self.llm.calls)
        self.assertIn("intent_planner", self.llm.calls)
        self.assertIn("safety_critic", self.llm.calls)
        self.assertIn("response_writer", self.llm.calls)
        self.assertIn("speak", {action.type for action in actions})
        self.assertTrue(any(message["role"] == "user" for message in self.core.state.runtime_history.recent_messages))

    def test_trivial_greeting_skips_all_memory_llm_calls(self) -> None:
        self.core.handle_event(
            Event(type="user_text_input", timestamp=2001, payload={"text": "你好", "source": "test"})
        )

        self.assertNotIn("memory_observer", self.llm.calls)
        trace = self.core.last_runtime_trace
        self.assertIsNotNone(trace)
        event_entry = trace.find("memory_pipeline", "event_skipped")[0]
        action_entry = trace.find("memory_pipeline", "action_skipped")[0]
        self.assertEqual(event_entry.payload["memory_event_skip_reason"], "skipped_trivial_user_text")
        self.assertFalse(event_entry.payload["memory_event_pipeline_called"])
        self.assertFalse(event_entry.payload["memory_event_enqueued"])
        self.assertFalse(event_entry.payload["memory_event_sync_called"])
        self.assertIn(
            action_entry.payload["memory_action_skip_reason"],
            {"skipped_speak_display_only", "skipped_non_memory_actions"},
        )
        self.assertFalse(action_entry.payload["memory_action_enqueued"])
        self.assertFalse(action_entry.payload["memory_action_sync_called"])

    def test_explicit_preference_is_eventually_stored_without_same_turn_read(self) -> None:
        memory_store = LongTermMemoryStore(self.root / "async_memory.json")
        pipeline = _BlockingEventMemoryPipeline(memory_store)
        core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=FakeLLMService(reply_text="fallback reply"),
            store=JsonStore(self.root / "async_runtime.json"),
            long_term_memory_pipeline=pipeline,  # type: ignore[arg-type]
        )
        try:
            actions, _ = core.handle_event(
                Event(
                    type="user_text_input",
                    timestamp=2002,
                    payload={"text": "以后我更喜欢 1 小时专注，不喜欢 25 分钟", "source": "test"},
                )
            )

            self.assertIn("speak", {action.type for action in actions})
            self.assertTrue(pipeline.started.wait(timeout=1))
            self.assertFalse(pipeline.finished.is_set())
            self.assertEqual(memory_store.list("default"), [])

            trace = core.last_runtime_trace
            self.assertIsNotNone(trace)
            event_entry = trace.find("memory_pipeline", "event_enqueued")[0]
            context_entry = trace.find("personal_context", "built")[0]
            self.assertTrue(event_entry.payload["memory_event_gate_allowed"])
            self.assertTrue(event_entry.payload["memory_event_enqueued"])
            self.assertIsNone(event_entry.payload["memory_event_enqueue_error"])
            self.assertTrue(
                event_entry.payload["memory_event_submit_result"]["accepted"]
            )
            self.assertEqual(
                event_entry.payload["memory_event_submit_result"]["reason"],
                "enqueued",
            )
            self.assertFalse(event_entry.payload["memory_event_sync_called"])
            self.assertFalse(event_entry.payload["memory_event_pipeline_called_sync"])
            self.assertEqual(
                context_entry.payload["personal_context"]["behavior_patterns"],
                [],
            )

            pipeline.release.set()
            self.assertTrue(pipeline.finished.wait(timeout=2))
            next_context = core.personal_context_builder.build(
                user_id="default",
                state=core.state,
                event=Event(type="user_text_input", timestamp=2003, payload={"text": "继续"}),
            )
            self.assertTrue(next_context.behavior_patterns)
        finally:
            pipeline.release.set()
            core.shutdown()

    def test_event_memory_failure_does_not_break_current_response(self) -> None:
        pipeline = _FailingEventMemoryPipeline(LongTermMemoryStore(self.root / "failing_memory.json"))
        llm_service = FakeLLMService(reply_text="fallback reply")
        memory_worker = MemoryBackgroundWorker(
            pipeline,  # type: ignore[arg-type]
            llm_service,
            max_retries=0,
        )
        core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=llm_service,
            store=JsonStore(self.root / "failing_runtime.json"),
            long_term_memory_pipeline=pipeline,  # type: ignore[arg-type]
            memory_worker=memory_worker,
        )
        try:
            actions, _ = core.handle_event(
                Event(type="user_text_input", timestamp=2003, payload={"text": "以后请记住这个偏好"})
            )

            self.assertIn("speak", {action.type for action in actions})
            self.assertTrue(pipeline.attempted.wait(timeout=1))
            core.memory_worker.wait_for_idle(timeout=2)
            self.assertEqual(core.memory_worker.stats()["failed_count"], 1)
            trace = core.last_runtime_trace
            self.assertIsNotNone(trace)
            self.assertTrue(trace.find("memory_pipeline", "event_enqueued"))
        finally:
            core.shutdown()

    def test_sensor_event_skips_memory_pipeline(self) -> None:
        self.core.handle_event(
            Event(type="light_level_updated", timestamp=2003, payload={"light_lux": 100, "level": "normal"})
        )

        self.assertNotIn("memory_observer", self.llm.calls)
        trace = self.core.last_runtime_trace
        self.assertIsNotNone(trace)
        event_entry = trace.find("memory_pipeline", "event_skipped")[0]
        action_entry = trace.find("memory_pipeline", "action_skipped")[0]
        self.assertEqual(event_entry.payload["memory_event_skip_reason"], "skipped_sensor_event")
        self.assertFalse(event_entry.payload["memory_event_sync_called"])
        self.assertFalse(action_entry.payload["memory_action_enqueued"])

    def test_internal_action_result_event_skips_memory_pipeline(self) -> None:
        self.core.handle_event(
            Event(
                type="system_triggered",
                timestamp=2004,
                payload={"trigger": "agent_response_completed", "source": "agent_action_result"},
            )
        )

        self.assertNotIn("memory_observer", self.llm.calls)
        trace = self.core.last_runtime_trace
        self.assertIsNotNone(trace)
        event_entry = trace.find("memory_pipeline", "event_skipped")[0]
        action_entry = trace.find("memory_pipeline", "action_skipped")[0]
        self.assertEqual(event_entry.payload["memory_event_skip_reason"], "skipped_internal_event")
        self.assertEqual(action_entry.payload["memory_action_skip_reason"], "skipped_empty_actions")

    def test_focus_action_memory_is_enqueued_not_called_synchronously(self) -> None:
        self.core.handle_event(
            Event(type="focus_start_requested", timestamp=2005, payload={"duration_sec": 3600, "source": "test"})
        )

        trace = self.core.last_runtime_trace
        self.assertIsNotNone(trace)
        action_entry = trace.find("memory_pipeline", "action_enqueued")[0]
        self.assertTrue(action_entry.payload["memory_action_gate_allowed"])
        self.assertTrue(action_entry.payload["memory_action_enqueued"])
        self.assertTrue(
            action_entry.payload["memory_action_submit_result"]["accepted"]
        )
        self.assertEqual(
            action_entry.payload["memory_action_submit_result"]["reason"],
            "enqueued",
        )
        self.assertTrue(action_entry.payload["memory_async_submit_success"])
        self.assertFalse(action_entry.payload["memory_action_sync_called"])
        self.assertFalse(action_entry.payload["memory_action_pipeline_called_sync"])

    def test_continue_focus_action_restores_state_without_agent_loop(self) -> None:
        self.llm.responses.update(
            {
                "intent_planner": [
                    '{"intents":[{"type":"continue_focus","priority":90,"reason":"continue","payload":{"duration_minutes":20},"requires_llm":false}],"reasoning":"continue","risk_level":"low","interrupt_user":false}'
                ]
            }
        )

        actions, _ = self.core.handle_event(
            Event(type="user_text_input", timestamp=2100, payload={"text": "continue focus", "source": "test"})
        )

        self.assertIn("start_timer", {action.type for action in actions})
        self.assertTrue(self.core.state.focus.active)
        self.assertEqual(self.core.state.focus.target_duration_sec, 1200)

    def test_timer_finished_generates_completion_feedback(self) -> None:
        self.core.handle_event(
            Event(type="focus_start_requested", timestamp=1000, payload={"duration_sec": 1500, "source": "test"})
        )

        actions, _ = self.core.handle_event(
            Event(type="timer_finished", timestamp=2500, payload={"timer": "focus"})
        )

        self.assertFalse(self.core.state.focus.active)
        self.assertIn("stop_timer", {action.type for action in actions})
        self.assertIn("speak", {action.type for action in actions})

    def test_guard_cooldown_blocks_repeated_rest_notification(self) -> None:
        self.core.state.focus.active = True
        self.core.state.focus.elapsed_sec = 600
        self.core.state.focus.remaining_sec = 900
        self.core.state.user.presence = "present"
        self.core.state.user.fatigue_level = "high"

        first_actions, _ = self.core.handle_event(
            Event(type="system_triggered", timestamp=3000, payload={"trigger": "focus_health_check"})
        )
        second_actions, _ = self.core.handle_event(
            Event(type="system_triggered", timestamp=3020, payload={"trigger": "focus_health_check"})
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in first_actions))
        self.assertFalse(any(action.payload.get("reason") == "rest_reminder" for action in second_actions))


if __name__ == "__main__":
    unittest.main()
