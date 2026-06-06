from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.rule_intent_builder import RuleIntentBuilder
from src.agent.event import Event
from src.agent.memory.memory_task import MemorySubmitResult
from src.agent.state import AgentState
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore
from tests.fakes.fake_llm_service import FakeLLMService


class _FailIfCalledOrchestrator:
    def decide(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("structured decisions must not call the Orchestrator")


class _DisabledMemoryWorker:
    def submit_event_memory(self, **kwargs):
        del kwargs
        return MemorySubmitResult(False, None, "disabled_for_test", 0)

    def submit_action_memory(self, **kwargs):
        del kwargs
        return MemorySubmitResult(False, None, "disabled_for_test", 0)

    def shutdown(self, timeout=None):
        del timeout


class RuleIntentBuilderUnitTestCase(unittest.TestCase):
    def test_unsupported_event_returns_none(self) -> None:
        state = AgentState()
        plan = RuleIntentBuilder().build(
            event=Event(type="user_text_input", timestamp=1, payload={"text": "hello"}),
            previous_state=state,
            current_state=state,
        )

        self.assertIsNone(plan)


class RuleIntentBuilderIntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.llm = FakeLLMService(reply_text="fallback reply")
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=self.llm,
            store=JsonStore(self.root / "runtime.json"),
            decision_pipeline=DecisionPipeline(orchestrator=_FailIfCalledOrchestrator()),
            memory_worker=_DisabledMemoryWorker(),  # type: ignore[arg-type]
        )

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_focus_start_uses_rule_validator_guard_and_realizer(self) -> None:
        actions, _ = self.core.handle_event(
            Event(
                type="focus_start_requested",
                timestamp=100,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )

        self.assertEqual([intent.type for intent in self.core.last_intents], ["start_focus"])
        self.assertEqual({action.type for action in actions}, {"start_timer", "display"})
        self.assertEqual(
            next(action for action in actions if action.type == "start_timer").payload["duration_sec"],
            1500,
        )
        self._assert_structured_result("focus_start_requested")

    def test_duplicate_focus_start_is_no_op(self) -> None:
        self.core.handle_event(
            Event(type="focus_start_requested", timestamp=100, payload={"duration_sec": 600})
        )
        self.llm.calls.clear()

        actions, _ = self.core.handle_event(
            Event(type="focus_start_requested", timestamp=101, payload={"duration_sec": 900})
        )

        self.assertEqual([intent.type for intent in self.core.last_intents], ["no_op"])
        self.assertEqual(actions, [])
        self.assertEqual(self.core.state.focus.target_duration_sec, 600)
        self._assert_structured_result("focus_start_requested")

    def test_active_focus_stop_uses_rule_path(self) -> None:
        self.core.handle_event(
            Event(type="focus_start_requested", timestamp=100, payload={"duration_sec": 600})
        )
        self.llm.calls.clear()

        actions, _ = self.core.handle_event(
            Event(type="focus_stop_requested", timestamp=200, payload={"source": "test"})
        )

        self.assertEqual([intent.type for intent in self.core.last_intents], ["stop_focus"])
        self.assertEqual({action.type for action in actions}, {"stop_timer", "display"})
        self._assert_structured_result("focus_stop_requested")

    def test_inactive_focus_stop_is_no_op(self) -> None:
        actions, _ = self.core.handle_event(
            Event(type="focus_stop_requested", timestamp=200, payload={"source": "test"})
        )

        self.assertEqual([intent.type for intent in self.core.last_intents], ["no_op"])
        self.assertEqual(actions, [])
        self._assert_structured_result("focus_stop_requested")

    def test_active_timer_finished_uses_deterministic_completion_copy(self) -> None:
        self.core.handle_event(
            Event(type="focus_start_requested", timestamp=100, payload={"duration_sec": 600})
        )
        self.llm.calls.clear()

        actions, _ = self.core.handle_event(
            Event(type="timer_finished", timestamp=700, payload={"timer": "focus"})
        )

        self.assertEqual([intent.type for intent in self.core.last_intents], ["complete_focus"])
        self.assertEqual({action.type for action in actions}, {"stop_timer", "speak", "display"})
        visible_texts = {
            action.payload.get("text")
            for action in actions
            if action.type in {"speak", "display"}
        }
        self.assertEqual(visible_texts, {"这轮专注完成了。"})
        self._assert_structured_result("timer_finished")

    def test_stale_timer_finished_is_no_op(self) -> None:
        actions, _ = self.core.handle_event(
            Event(type="timer_finished", timestamp=700, payload={"timer": "focus"})
        )

        self.assertEqual([intent.type for intent in self.core.last_intents], ["no_op"])
        self.assertEqual(actions, [])
        self._assert_structured_result("timer_finished")

    def _assert_structured_result(self, event_type: str) -> None:
        result = self.core.last_decision_result
        self.assertIsNotNone(result)
        self.assertFalse(result.used_llm)
        self.assertIsNone(result.situation)
        self.assertIsNone(result.safety_review)
        self.assertEqual(result.response.speak_text, "")
        self.assertEqual(result.stage_metadata["decision_source"], "rule_intent_builder")
        self.assertTrue(result.stage_metadata["structured_decision"])
        self.assertEqual(result.stage_metadata["rule_event_type"], event_type)
        self.assertTrue(result.stage_metadata["validator"]["ok"])
        self.assertIn("guard", result.stage_metadata)
        self.assertIn("action_realizer", result.stage_metadata)
        self.assertEqual(self.llm.calls, [])

        trace = self.core.last_runtime_trace
        self.assertIsNotNone(trace)
        self.assertTrue(trace.find("rule_intent_builder", "plan_built"))
        self.assertTrue(trace.find("validator", "intent_plan"))
        self.assertTrue(trace.find("guard", "filtered"))
        self.assertTrue(trace.find("action_realizer", "realized"))
        self.assertFalse(trace.find("llm_output"))


class SemanticDecisionRegressionTestCase(unittest.TestCase):
    def test_user_text_still_uses_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            llm = FakeLLMService(reply_text="fallback reply")
            core = AgentCore(
                output=ConsoleOutput(silent=True),
                timer_service=TimerService(background=False),
                runtime_history_service=RuntimeHistoryService(),
                llm_service=llm,
                store=JsonStore(Path(temp_dir) / "runtime.json"),
                memory_worker=_DisabledMemoryWorker(),  # type: ignore[arg-type]
            )
            try:
                actions, _ = core.handle_event(
                    Event(type="user_text_input", timestamp=1, payload={"text": "hello"})
                )

                self.assertTrue(core.last_decision_result.used_llm)
                self.assertIn("situation_analyst", llm.calls)
                self.assertIn("intent_planner", llm.calls)
                self.assertIn("safety_critic", llm.calls)
                self.assertIn("response_writer", llm.calls)
                self.assertIn("speak", {action.type for action in actions})
                self.assertNotEqual(
                    core.last_decision_result.stage_metadata.get("decision_source"),
                    "rule_intent_builder",
                )
            finally:
                core.shutdown()


if __name__ == "__main__":
    unittest.main()
