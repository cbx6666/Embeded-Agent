from __future__ import annotations

import json
import unittest

from src.agent.config.policy_config import DecisionPolicyConfig
from src.agent.decision.agent_context_builder import AgentContext, AgentContextBuilder
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.event import Event
from src.agent.llm_agent.fast_dialogue import build_fast_dialogue_prompt
from src.agent.state import AgentState
from src.agent.state.focus_state import FocusState
from tests.fakes.fake_llm_service import FakeLLMService


class LlmFastModeTestCase(unittest.TestCase):
    def test_fast_mode_uses_single_fast_dialogue_call(self) -> None:
        llm = FakeLLMService()
        pipeline = DecisionPipeline(decision_policy=DecisionPolicyConfig(llm_mode="fast"))

        result = pipeline.decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="speech_recognized", timestamp=1, payload={"text": "你好"}),
            llm_service=llm,
        )

        self.assertEqual(llm.calls, ["fast_dialogue"])
        self.assertEqual(llm.reply_calls, 0)
        self.assertEqual(result.intents[0].type, "answer_user")
        self.assertIn("speak", {action.type for action in result.actions})

    def test_fast_prompt_includes_focus_state(self) -> None:
        state = AgentState(
            focus=FocusState(active=True, elapsed_sec=600, remaining_sec=900, target_duration_sec=1500),
        )
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="speech_recognized", timestamp=3, payload={"text": "我还剩多久"}),
            personal_context=None,
        )
        prompt = build_fast_dialogue_prompt(context)

        self.assertIn("专注：进行中", prompt)
        self.assertIn("剩余 15 分钟", prompt)
        self.assertIn("我还剩多久", prompt)
        self.assertIn("结构化上下文 JSON", prompt)

    def test_fast_mode_answers_with_state_context(self) -> None:
        llm = FakeLLMService()
        state = AgentState(
            focus=FocusState(active=True, elapsed_sec=600, remaining_sec=120, target_duration_sec=1500),
        )
        pipeline = DecisionPipeline(decision_policy=DecisionPolicyConfig(llm_mode="fast"))
        result = pipeline.decide(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="speech_recognized", timestamp=4, payload={"text": "还有多久"}),
            llm_service=llm,
        )

        speak_actions = [action for action in result.actions if action.type == "speak"]
        self.assertTrue(speak_actions)
        self.assertIn("120", speak_actions[0].payload.get("text", ""))

    def test_full_mode_still_runs_four_roles(self) -> None:
        llm = FakeLLMService()
        pipeline = DecisionPipeline(decision_policy=DecisionPolicyConfig(llm_mode="full"))

        pipeline.decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=2, payload={"text": "你好"}),
            llm_service=llm,
        )

        self.assertEqual(
            llm.calls[:4],
            ["situation_analyst", "intent_planner", "safety_critic", "response_writer"],
        )


if __name__ == "__main__":
    unittest.main()
