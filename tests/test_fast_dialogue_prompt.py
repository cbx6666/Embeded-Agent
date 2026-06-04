from __future__ import annotations

import unittest

from src.agent.decision.agent_context_builder import AgentContextBuilder
from src.agent.event import Event
from src.agent.llm_agent.fast_dialogue import build_fast_dialogue_prompt
from src.agent.state import AgentState
from src.agent.state.focus_state import FocusState


class FastDialoguePromptTestCase(unittest.TestCase):
    def test_prompt_contains_state_and_user_message(self) -> None:
        state = AgentState(
            focus=FocusState(active=True, elapsed_sec=600, remaining_sec=900, target_duration_sec=1500),
        )
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="speech_recognized", timestamp=1, payload={"text": "我还剩多久"}),
            personal_context=None,
        )
        prompt = build_fast_dialogue_prompt(context)

        self.assertIn("专注：进行中", prompt)
        self.assertIn("剩余 15 分钟", prompt)
        self.assertIn("我还剩多久", prompt)


if __name__ == "__main__":
    unittest.main()
