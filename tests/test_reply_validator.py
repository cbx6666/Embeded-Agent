from __future__ import annotations

import unittest

from src.agent.llm.prompt_builder import build_wellness_prompt
from src.agent.llm.reply_validator import normalize_reply, validate_tts_reply


class ReplyValidatorTest(unittest.TestCase):
    def test_empty_reply_invalid(self) -> None:
        valid, reason = validate_tts_reply("")
        self.assertFalse(valid)
        self.assertEqual(reason, "empty")

    def test_incomplete_reply_invalid(self) -> None:
        for text in (
            "有点累，要不要。",
            "如果累了的话。",
            "注意坐姿，放松。",
            "你现在的话，可以。",
        ):
            valid, reason = validate_tts_reply(text)
            self.assertFalse(valid, msg=text)
            self.assertTrue(reason)

    def test_complete_short_reply_valid(self) -> None:
        valid, reason = validate_tts_reply("稍微往后靠一点，别让脖子太累。")
        self.assertTrue(valid)
        self.assertEqual(reason, "")
        self.assertTrue(validate_tts_reply("歇会儿")[0])
        self.assertTrue(validate_tts_reply("光线有点暗")[0])

    def test_normalize_strips_whitespace(self) -> None:
        self.assertEqual(normalize_reply("  你好。  "), "你好。")

    def test_wellness_prompt_includes_tts_quality(self) -> None:
        prompt = build_wellness_prompt(
            wellness_summary={"should_care": True},
            selected_intent="suggest_rest",
            care_focus="fatigue",
            user_context={},
        )
        self.assertIn("完整句子", prompt)
        self.assertIn("语音合成", prompt)


if __name__ == "__main__":
    unittest.main()
