from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.decision.intent import AgentIntent
from src.agent.decision.intent_guard import guard_intents
from src.agent.decision.llm_intent_planner import plan_intents_with_llm
from src.agent.decision.planner import build_candidate_intents, plan_intents
from src.agent.decision.policy import decide_actions_with_intents
from src.agent.state import AgentState
from src.agent.event import Event
from src.services.llm_service import LLMService


class SpyLLMService(LLMService):
    def __init__(self) -> None:
        self.reply_call_count = 0
        self.choose_call_count = 0
        self.choose_response: str | None = None
        self.last_prompt = ""
        self.last_allowed_intents: list[str] = []

    def generate_reply(self, text: str, state) -> str:  # type: ignore[override]
        self.reply_call_count += 1
        return "这是来自 LLM 的回复。"

    def choose_intents(self, prompt: str, allowed_intent_types: list[str]) -> str:  # type: ignore[override]
        self.choose_call_count += 1
        self.last_prompt = prompt
        self.last_allowed_intents = list(allowed_intent_types)
        if self.choose_response is not None:
            return self.choose_response
        return super().choose_intents(prompt, allowed_intent_types)


class LLMIntentPlannerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.spy_llm = SpyLLMService()
        self.state = AgentState()

    def test_empty_candidates_do_not_call_llm(self) -> None:
        event = Event(type="user_text_input", timestamp=1000, payload={"text": "你好"})
        intents = plan_intents_with_llm(
            event=event,
            state=self.state,
            candidate_intents=[],
            llm_service=self.spy_llm,
        )

        self.assertEqual(intents, [])
        self.assertEqual(self.spy_llm.choose_call_count, 0)

    def test_status_query_uses_rule_intent_without_llm(self) -> None:
        previous_state = AgentState()
        current_state = AgentState()
        event = Event(type="user_text_input", timestamp=1001, payload={"text": "当前状态", "source": "test"})

        intents = plan_intents(previous_state, current_state, event, llm_service=self.spy_llm)

        self.assertEqual(self.spy_llm.choose_call_count, 0)
        self.assertEqual(intents[0].reason, "status_query")

    def test_open_user_text_can_use_llm_and_return_answer_user(self) -> None:
        previous_state = AgentState()
        current_state = AgentState()
        event = Event(type="user_text_input", timestamp=1002, payload={"text": "我有点累，怎么办", "source": "test"})

        intents = plan_intents(previous_state, current_state, event, llm_service=self.spy_llm)

        self.assertEqual(self.spy_llm.choose_call_count, 1)
        self.assertEqual(intents[0].type, "answer_user")
        self.assertTrue(intents[0].payload.get("llm_selected"))

    def test_speech_and_text_share_same_llm_intent_logic(self) -> None:
        previous_state = AgentState()
        current_state = AgentState()
        text_event = Event(type="user_text_input", timestamp=1003, payload={"text": "为什么我总是分心", "source": "test"})
        speech_event = Event(type="speech_recognized", timestamp=1004, payload={"text": "为什么我总是分心", "source": "asr"})

        text_intents = plan_intents(previous_state, current_state, text_event, llm_service=self.spy_llm)
        speech_intents = plan_intents(previous_state, current_state, speech_event, llm_service=self.spy_llm)

        self.assertEqual(self.spy_llm.choose_call_count, 2)
        self.assertEqual(text_intents[0].type, "answer_user")
        self.assertEqual(speech_intents[0].type, "answer_user")

    def test_invalid_json_falls_back_to_candidates(self) -> None:
        event = Event(type="user_text_input", timestamp=1005, payload={"text": "你好"})
        candidates = [AgentIntent(type="answer_user", priority=70, reason="user_dialogue", requires_llm=True)]
        self.spy_llm.choose_response = "{not-json"

        intents = plan_intents_with_llm(
            event=event,
            state=self.state,
            candidate_intents=candidates,
            llm_service=self.spy_llm,
        )

        self.assertEqual(intents, candidates)

    def test_illegal_intent_type_falls_back_to_candidates(self) -> None:
        event = Event(type="user_text_input", timestamp=1006, payload={"text": "你好"})
        candidates = [AgentIntent(type="answer_user", priority=70, reason="user_dialogue", requires_llm=True)]
        self.spy_llm.choose_response = (
            '{"intents":[{"type":"hack_system","priority":10,"reason":"非法","payload":{},"requires_llm":true}]}'
        )

        intents = plan_intents_with_llm(
            event=event,
            state=self.state,
            candidate_intents=candidates,
            llm_service=self.spy_llm,
        )

        self.assertEqual(intents, candidates)

    def test_silent_mode_guard_allows_intent_but_realizer_will_handle_speak_boundary(self) -> None:
        state = AgentState()
        state.interaction.mode = "silent"
        event = Event(type="system_triggered", timestamp=1007, payload={"trigger": "focus_health_check"})
        intents = [AgentIntent(type="suggest_rest", priority=80, reason="rest_reminder")]

        guarded = guard_intents(intents, state=state, event=event, fallback_intents=intents)

        self.assertEqual(guarded[0].type, "suggest_rest")

    def test_away_filters_proactive_intents(self) -> None:
        state = AgentState()
        state.user.presence = "away"
        event = Event(type="system_triggered", timestamp=1008, payload={"trigger": "periodic_check"})
        intents = [
            AgentIntent(type="suggest_rest", priority=80, reason="rest_reminder"),
            AgentIntent(type="remind_distraction", priority=80, reason="distraction_reminder"),
            AgentIntent(type="adjust_environment_feedback", priority=20, reason="environment_warning"),
        ]

        guarded = guard_intents(intents, state=state, event=event, fallback_intents=None)

        self.assertEqual(len(guarded), 1)
        self.assertEqual(guarded[0].type, "no_op")

    def test_autonomous_system_trigger_disallows_requires_llm(self) -> None:
        state = AgentState()
        event = Event(type="system_triggered", timestamp=1009, payload={"trigger": "focus_health_check"})
        intents = [AgentIntent(type="answer_user", priority=10, reason="llm", requires_llm=True)]

        guarded = guard_intents(intents, state=state, event=event, fallback_intents=None)

        self.assertEqual(guarded[0].type, "no_op")

    def test_policy_decide_actions_with_intents_returns_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "runtime_store.json"
            state = AgentState()
            llm_service = SpyLLMService()
            _, actions = decide_actions_with_intents(
                previous_state=state,
                current_state=state,
                event=Event(type="user_text_input", timestamp=1010, payload={"text": "你好呀", "source": "test"}),
                llm_service=llm_service,
            )

            self.assertTrue(any(action.type in {"speak", "display"} for action in actions))
            self.assertEqual(llm_service.choose_call_count, 1)


if __name__ == "__main__":
    unittest.main()
