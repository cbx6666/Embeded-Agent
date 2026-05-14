from __future__ import annotations

import json
import unittest

from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.agent_context_builder import AgentContextBuilder
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.event import Event
from src.agent.llm_agent.schemas import ResponseDraft
from src.agent.state import AgentState
from tests.fakes.fake_llm_service import FakeLLMService


class BoundaryAndLLMFailureScenarioTestCase(unittest.TestCase):
    def test_malformed_json_extra_control_fields_and_empty_response_fallback(self) -> None:
        llm = FakeLLMService(
            {
                "situation_analyst": "{not-json",
                "intent_planner": json.dumps(
                    {
                        "actions": [{"type": "speak"}],
                        "intents": [{"type": "answer_user", "priority": 50, "payload": {}}],
                    }
                ),
                "safety_critic": json.dumps({"decision": "approve", "reason": "ok", "revised_plan": None}),
                "response_writer": json.dumps({"speak_text": "", "display_text": "", "tone": "calm"}),
            },
            reply_text="fallback stayed stable",
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=10, payload={"text": "hello"}),
            llm_service=llm,
        )

        self.assertEqual([intent.type for intent in result.intents], ["answer_user"])
        self.assertEqual([action.payload.get("text") for action in result.actions if action.type == "speak"], ["fallback stayed stable"])
        self.assertIn("llm_fallback", str(result.fallback_reason))
        self.assertTrue(any(event["stage"] == "prompt" for event in result.stage_metadata["trace"]["events"]))

    def test_invalid_intent_type_is_rejected_before_guard_and_action(self) -> None:
        llm = FakeLLMService(
            {
                "situation_analyst": _situation(),
                "intent_planner": json.dumps(
                    {
                        "intents": [{"type": "invent_device_command", "priority": 99, "payload": {"state_patch": {"x": 1}}}],
                        "reasoning": "bad",
                        "risk_level": "low",
                    }
                ),
                "safety_critic": json.dumps({"decision": "approve", "reason": "ok", "revised_plan": None}),
                "response_writer": json.dumps({"speak_text": "should not execute", "display_text": "should not execute"}),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=20, payload={"text": "please"}),
            llm_service=llm,
        )

        self.assertEqual(result.actions, [])
        self.assertEqual([intent.type for intent in result.intents], ["no_op"])
        self.assertIn("unregistered type", result.fallback_reason or "")

    def test_guard_cooldown_presence_and_action_realizer_are_deterministic(self) -> None:
        state = AgentState()
        state.user.presence = "away"
        state.cooldown.reminder_last_ts["rest_reminder"] = 100
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="system_triggered", timestamp=120, payload={"trigger": "focus_health_check"}),
            personal_context=None,
        )

        guarded = DeterministicGuard(reminder_cooldown_sec=300).filter(
            IntentPlan(intents=[AgentIntent(type="suggest_rest", priority=80, reason="rest")]),
            context,
        )
        self.assertEqual(guarded.plan.intents[0].type, "no_op")
        self.assertIn("presence safety blocked", guarded.findings[0].reason)

        local_context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=130, payload={"text": "volume"}),
            personal_context=None,
        )
        plan = IntentPlan(intents=[AgentIntent(type="set_tts_volume", priority=10, payload={"volume": 999})])
        first = ActionRealizer().realize(plan, response=ResponseDraft(), context=local_context)
        second = ActionRealizer().realize(plan, response=ResponseDraft(), context=local_context)

        self.assertEqual([action.payload for action in first], [action.payload for action in second])
        self.assertEqual(first[0].payload["volume"], 100)


def _situation() -> str:
    return json.dumps(
        {
            "summary": "test",
            "user_intent": "test",
            "current_state": "state",
            "risks": [],
            "uncertainties": [],
            "should_respond": True,
            "risk_level": "low",
        }
    )


if __name__ == "__main__":
    unittest.main()
