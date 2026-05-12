from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.event import Event
from src.agent.llm_agent.agent_context import AgentContextBuilder
from src.agent.memory.llm_memory_manager import LLMMemoryManager, MemoryContextBuilder, MemoryValidator
from src.agent.memory.schemas import MemoryCandidate
from src.agent.memory.memory_pipeline import MemoryPipeline
from src.agent.memory.memory_store import MemoryStore
from src.agent.memory.profile_snapshot_builder import ProfileSnapshotBuilder
from src.agent.state import AgentState
from src.services.llm_service import LLMService


class ScriptedLLM(LLMService):
    def __init__(self, responses: dict[str, list[str] | str] | None = None) -> None:
        self.responses = dict(responses or {})
        self.calls: list[str] = []
        self.reply_calls = 0

    def complete_json(self, role: str, prompt: str) -> str:  # type: ignore[override]
        self.calls.append(role)
        value = self.responses.get(role)
        if isinstance(value, list):
            if value:
                return value.pop(0)
        if isinstance(value, str):
            return value
        return super()._mock_complete_json(role, prompt)

    def generate_reply(self, text: str, state=None) -> str:  # type: ignore[override]
        self.reply_calls += 1
        return f"fallback reply: {text[:20]}"


class LLMCenteredArchitectureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_multi_intent_complex_natural_language(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": _json(
                    summary="User wants to continue focus but reduce interruption.",
                    user_intent="continue focus with fewer reminders",
                    should_respond=True,
                ),
                "intent_planner": json.dumps(
                    {
                        "intents": [
                            {"type": "continue_focus", "priority": 90, "reason": "continue", "payload": {"duration_minutes": 20}},
                            {"type": "reduce_reminder_frequency", "priority": 70, "reason": "less interruption", "payload": {}},
                            {"type": "answer_user", "priority": 60, "reason": "ack", "payload": {}, "requires_llm": True},
                        ],
                        "reasoning": "Respect the user request.",
                        "risk_level": "low",
                        "interrupt_user": False,
                    }
                ),
                "safety_critic": _safety("approve"),
                "response_writer": _response("Got it. I will keep reminders lighter."),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=10, payload={"text": "I am tired but want 20 more minutes; stop reminding me so much."}),
            llm_service=llm,
        )

        self.assertEqual({"continue_focus", "reduce_reminder_frequency", "answer_user"}, {intent.type for intent in result.intents})
        self.assertIn("start_timer", {action.type for action in result.actions})
        self.assertIn("speak", {action.type for action in result.actions})
        self.assertEqual(llm.calls[:4], ["situation_analyst", "intent_planner", "safety_critic", "response_writer"])

    def test_llm_output_illegal_schema_is_rejected(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": _json(summary="test", should_respond=True),
                "intent_planner": json.dumps(
                    {
                        "intents": [{"type": "hack_system", "priority": 100, "reason": "bad", "payload": {}}],
                        "reasoning": "bad",
                        "risk_level": "low",
                    }
                ),
                "safety_critic": _safety("approve"),
                "response_writer": _response("This should not become an action."),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=20, payload={"text": "do something unsafe"}),
            llm_service=llm,
        )

        self.assertEqual(result.actions, [])
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertIn("validation:", str(result.fallback_reason))

    def test_safety_critic_revise(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": _json(summary="User is busy.", should_respond=True),
                "intent_planner": json.dumps(
                    {
                        "intents": [{"type": "suggest_rest", "priority": 90, "reason": "tired", "payload": {}}],
                        "reasoning": "Suggest rest.",
                        "risk_level": "low",
                    }
                ),
                "safety_critic": json.dumps(
                    {
                        "decision": "revise",
                        "reason": "Avoid interrupting; acknowledge quietly.",
                        "revised_plan": {
                            "intents": [{"type": "display_update", "priority": 40, "reason": "quiet revision", "payload": {}}],
                            "reasoning": "Quiet display only.",
                            "risk_level": "low",
                        },
                    }
                ),
                "response_writer": _response("I will keep this quiet."),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=30, payload={"text": "not now"}),
            llm_service=llm,
        )

        self.assertEqual([intent.type for intent in result.intents], ["display_update"])
        self.assertEqual(result.safety_review.decision, "revise")  # type: ignore[union-attr]
        self.assertEqual({action.type for action in result.actions}, {"display"})

    def test_memory_extraction_and_consolidation(self) -> None:
        store = MemoryStore(self.root / "memory.json")
        candidate = {
            "memory_type": "explicit_preference",
            "content": "User prefers fewer rest reminders.",
            "confidence": 0.82,
            "evidence": [{"event": "user_text_input"}],
        }
        llm = ScriptedLLM(
            {
                "memory_observer": json.dumps({"worth_remembering": True, "reason": "preference"}),
                "memory_extractor": json.dumps({"candidates": [candidate]}),
                "memory_critic": json.dumps({"approved_indexes": [0], "rejected_reasons": []}),
                "memory_consolidator": json.dumps({"candidates": [candidate]}),
            }
        )
        context = MemoryContextBuilder().build(
            user_id="u1",
            event=Event(type="user_text_input", timestamp=40, payload={"text": "Please remind me less."}),
            state=AgentState(current_user_id="u1"),
        )

        result = LLMMemoryManager(store).process(context, llm)

        self.assertEqual(len(result.stored), 1)
        self.assertEqual(store.list("u1")[0].content, "User prefers fewer rest reminders.")

        result2 = LLMMemoryManager(store).process(context, llm)
        self.assertEqual(len(store.list("u1")), 1)
        self.assertGreaterEqual(len(result2.stored[0].evidence), 1)

    def test_memory_validator_rejects_memory_without_evidence(self) -> None:
        validator = MemoryValidator()
        candidate = MemoryCandidate(
            memory_type="explicit_preference",
            content="User prefers quiet reminders.",
            confidence=0.8,
            evidence=[],
        )

        self.assertEqual(validator.validate(candidate), "memory evidence is required")

    def test_profile_snapshot_and_context_retrieval(self) -> None:
        store = MemoryStore(self.root / "memory.json")
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="explicit_preference",
                content="User prefers quiet visual feedback.",
                confidence=0.9,
            ),
            timestamp=50,
        )
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="interaction_style",
                content="Use concise responses during focus.",
                confidence=0.8,
            ),
            timestamp=51,
        )
        state = AgentState(current_user_id="u1")
        snapshot = ProfileSnapshotBuilder(store).build(user_id="u1", state=state)
        context = AgentContextBuilder().build(
            previous_state=AgentState(current_user_id="u1"),
            current_state=state,
            event=Event(type="user_text_input", timestamp=52, payload={"text": "quiet please"}),
            profile_snapshot=snapshot,
        )

        self.assertEqual(len(snapshot.explicit_preferences), 1)
        self.assertTrue(any("quiet" in item["content"] for item in context.relevant_memories))

    def test_llm_fallback(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": "{not-json",
                "intent_planner": "{not-json",
                "safety_critic": _safety("approve"),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=60, payload={"text": "hello"}),
            llm_service=llm,
        )

        self.assertIn("llm_fallback", str(result.fallback_reason))
        self.assertEqual([intent.type for intent in result.intents], ["answer_user"])
        self.assertIn("speak", {action.type for action in result.actions})

    def test_guard_rejection(self) -> None:
        state = AgentState()
        state.user.presence = "away"
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="system_triggered", timestamp=70, payload={"trigger": "focus_health_check"}),
            profile_snapshot=None,
        )

        decision = DeterministicGuard().filter(
            IntentPlan(intents=[AgentIntent(type="suggest_rest", reason="rest")]),
            context,
        )

        self.assertEqual(decision.plan.intents[0].type, "no_op")
        self.assertEqual(decision.blocked_intents[0].type, "suggest_rest")

    def test_action_realization(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="focus_start_requested", timestamp=80, payload={"duration_sec": 600}),
            profile_snapshot=None,
        )
        actions = ActionRealizer().realize(
            IntentPlan(
                intents=[
                    AgentIntent(type="start_focus", priority=80),
                    AgentIntent(type="set_tts_volume", priority=70, payload={"volume": 25}),
                ]
            ),
            response=type("Draft", (), {"speak_text": "", "display_text": ""})(),
            context=context,
        )

        self.assertEqual(actions[0].type, "start_timer")
        self.assertEqual(actions[0].payload["duration_sec"], 600)
        self.assertIn("set_tts_volume", {action.type for action in actions})

    def test_memory_pipeline_builds_profile_snapshot(self) -> None:
        pipeline = MemoryPipeline(MemoryStore(self.root / "memory.json"))
        pipeline.store.upsert_candidate(
            "u1",
            MemoryCandidate(memory_type="active_constraint", content="Do not speak during silent mode.", confidence=0.9),
            timestamp=90,
        )

        snapshot = pipeline.build_profile_snapshot("u1", AgentState(current_user_id="u1"))

        self.assertEqual(snapshot.active_constraints[0]["content"], "Do not speak during silent mode.")


def _json(**values: object) -> str:
    payload = {
        "summary": values.get("summary", "summary"),
        "user_intent": values.get("user_intent", "intent"),
        "current_state": values.get("current_state", "state"),
        "risks": values.get("risks", []),
        "uncertainties": values.get("uncertainties", []),
        "should_respond": values.get("should_respond", False),
        "risk_level": values.get("risk_level", "low"),
    }
    return json.dumps(payload)


def _safety(decision: str) -> str:
    return json.dumps({"decision": decision, "reason": "ok", "revised_plan": None})


def _response(text: str) -> str:
    return json.dumps({"speak_text": text, "display_text": text, "tone": "calm"})


if __name__ == "__main__":
    unittest.main()
