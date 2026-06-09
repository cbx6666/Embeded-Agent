from __future__ import annotations

"""wellness_care_check / environment_care_check 触发与职责边界测试。

覆盖本轮需求第十一节：
- 疲劳高连续 20s -> suggest_rest
- 负面情绪连续 30s -> offer_emotion_care
- 疲劳高 + 低光 同时出现 -> wellness 出 suggest_rest（不被环境抢走）
- environment_care_check 只能 environment_warning / no_op，不能 rest_reminder
- sensor_status_report 只播环境，不出 wellness 提醒
- wellness 能拿到 user_context / memories
- speaking 状态下 wellness 提醒不被静默丢弃
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.core.models import Event, Intent
from src.agent.decision.behavior_distraction_handler import BehaviorDistractionHandler
from src.agent.decision.environment_care_handler import EnvironmentCareHandler
from src.agent.decision.wellness_care_handler import WellnessCareHandler
from src.agent.action.realizer import ActionRealizer
from src.agent.policy_config import WellnessCareCheckPolicy
from src.agent.state.agent_state import AgentState
from src.agent.context.memory_usage_hints import build_memory_usage_hints
from src.agent.llm.prompt_builder import build_wellness_prompt
from src.agent.memory.memory_model import GROUP_KEY_BY_TYPE
from src.agent.prompt_io import read_prompt
from src.agent.state.summary_builder import (
    build_environment_care_summary,
    build_wellness_care_summary,
)


class _FakeClient:
    """最小 LLM client：按 role 返回固定 dict，并记录 prompt。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, role: str, prompt: str, *, temperature=None) -> dict:
        self.prompts.append((role, prompt))
        return dict(self.payload)


class _FailingClient:
    def complete_json(self, role: str, prompt: str, *, temperature=None) -> dict:
        del role, prompt, temperature
        raise RuntimeError("llm unavailable")


def _set_recent(state: AgentState, signal: str, samples: list[tuple[int, str, float]]) -> None:
    """直接写入某信号的窗口时间序列 [(ts, value, confidence), ...]。"""

    state.runtime_history.signal_trends[signal] = {
        "current": samples[-1][1] if samples else None,
        "recent_values": [
            {"timestamp": ts, "value": value, "confidence": conf} for ts, value, conf in samples
        ],
    }


def _present_state() -> AgentState:
    state = AgentState()
    state.user.presence = "present"
    return state


class WellnessSummaryTriggerTest(unittest.TestCase):
    def test_fatigue_high_sustained_20s_triggers_rest(self) -> None:
        state = _present_state()
        state.user.fatigue_level = "high"
        state.user.fatigue_confidence = 0.9
        _set_recent(
            state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )
        summary = build_wellness_care_summary(state, check_time=1020)
        self.assertTrue(summary["should_care"])
        self.assertGreaterEqual(summary["fatigue"]["sustained_high_sec"], 20)
        self.assertEqual(summary["recommended_care_focus"], "fatigue")

    def test_negative_emotion_streak_30s_triggers_emotion(self) -> None:
        state = _present_state()
        state.user.emotion = "sad"
        state.user.emotion_confidence = 0.8
        _set_recent(
            state,
            "emotion",
            [(1000, "sad", 0.8), (1015, "sad", 0.8), (1030, "sad", 0.8)],
        )
        summary = build_wellness_care_summary(state, check_time=1030)
        self.assertTrue(summary["should_care"])
        self.assertGreaterEqual(summary["emotion"]["negative_streak_sec"], 30)
        self.assertEqual(summary["recommended_care_focus"], "emotion")

    def test_fatigue_wins_over_low_light(self) -> None:
        # 疲劳高 + 低光：wellness 关注疲劳，环境不参与（两条独立链路）。
        state = _present_state()
        state.user.fatigue_level = "high"
        state.environment.light_lux = 120  # 低光异常
        state.environment.light_level = "low"
        _set_recent(
            state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )
        summary = build_wellness_care_summary(state, check_time=1020)
        self.assertEqual(summary["recommended_care_focus"], "fatigue")
        # wellness 汇总里完全不含环境字段。
        self.assertNotIn("light", summary)
        self.assertNotIn("environment_triggers", summary)

    def test_no_signal_no_care(self) -> None:
        summary = build_wellness_care_summary(_present_state(), check_time=1000)
        self.assertFalse(summary["should_care"])
        self.assertEqual(summary["recommended_care_focus"], "none")

    def test_summary_carries_memories(self) -> None:
        memories = {"preferences": [{"content": "喜欢听音乐放松"}]}
        summary = build_wellness_care_summary(
            _present_state(), memories=memories, check_time=1000
        )
        self.assertEqual(summary["memories"], memories)

    def test_posture_requires_current_bad_and_stricter_window(self) -> None:
        state = _present_state()
        state.user.posture = "sitting"
        # 窗口里曾有 bad 姿态，但当前已恢复正常 -> 不触发。
        _set_recent(
            state,
            "posture",
            [(1000, "slouching", 0.9), (1030, "slouching", 0.9), (1060, "sitting", 0.9)],
        )
        summary = build_wellness_care_summary(state, check_time=1060)
        self.assertFalse(summary["posture"]["trigger_candidate"])

    def test_posture_triggers_when_current_bad_and_sustained(self) -> None:
        state = _present_state()
        state.user.posture = "slouching"
        samples = [(ts, "slouching", 0.9) for ts in range(1000, 1061, 10)]
        _set_recent(state, "posture", samples)
        summary = build_wellness_care_summary(state, check_time=1060)
        self.assertTrue(summary["posture"]["trigger_candidate"])
        self.assertGreaterEqual(summary["posture"]["sustained_bad_posture_sec"], 60)

    def test_rotates_after_posture_broadcast(self) -> None:
        state = _present_state()
        state.user.fatigue_level = "high"
        state.user.posture = "slouching"
        _set_recent(
            state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )
        _set_recent(
            state,
            "posture",
            [(1000, "slouching", 0.9), (1030, "slouching", 0.9), (1060, "slouching", 0.9)],
        )
        state.runtime_history.reminder_records = [
            {
                "reason": "posture_reminder",
                "timestamp": 900,
                "text": "坐姿提醒",
            }
        ]
        summary = build_wellness_care_summary(state, check_time=1060)
        self.assertEqual(summary["last_wellness_focus"], "posture")
        self.assertEqual(summary["recommended_care_focus"], "fatigue")

    def test_rotates_after_fatigue_broadcast(self) -> None:
        state = _present_state()
        state.user.fatigue_level = "high"
        state.user.emotion = "sad"
        _set_recent(
            state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )
        _set_recent(
            state,
            "emotion",
            [(1000, "sad", 0.8), (1015, "sad", 0.8), (1030, "sad", 0.8)],
        )
        state.runtime_history.reminder_records = [
            {"reason": "rest_reminder", "timestamp": 900, "text": "休息提醒"}
        ]
        summary = build_wellness_care_summary(state, check_time=1030)
        self.assertEqual(summary["last_wellness_focus"], "fatigue")
        self.assertEqual(summary["recommended_care_focus"], "emotion")

    def test_rotates_after_emotion_broadcast(self) -> None:
        state = _present_state()
        state.user.fatigue_level = "high"
        state.user.emotion = "sad"
        _set_recent(
            state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )
        _set_recent(
            state,
            "emotion",
            [(1000, "sad", 0.8), (1015, "sad", 0.8), (1030, "sad", 0.8)],
        )
        state.runtime_history.reminder_records = [
            {"reason": "emotion_reminder", "timestamp": 900, "text": "情绪关怀"}
        ]
        summary = build_wellness_care_summary(state, check_time=1030)
        self.assertEqual(summary["last_wellness_focus"], "emotion")
        self.assertEqual(summary["recommended_care_focus"], "fatigue")


class _ProbeResetMixin:
    def setUp(self) -> None:
        from src.adapters.voice.arbitration.session_probe import VoiceSessionProbe

        VoiceSessionProbe.global_probe().reset_for_tests()


class WellnessHandlerTest(_ProbeResetMixin, unittest.TestCase):
    def _fatigue_state(self) -> AgentState:
        state = _present_state()
        state.user.fatigue_level = "high"
        _set_recent(
            state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )
        return state

    def test_strong_trigger_produces_suggest_rest(self) -> None:
        handler = WellnessCareHandler()
        client = _FakeClient({"reply": "累了就歇会儿吧"})
        result = handler.decide(
            state=self._fatigue_state(),
            event=Event(type="system_triggered", timestamp=1020, payload={}),
            llm_client=client,
            user_context={"memories": {"x": 1}},
        )
        self.assertEqual(result.intents[0].type, "suggest_rest")
        reasons = {a.payload.get("reason") for a in result.actions}
        self.assertIn("rest_reminder", reasons)
        self.assertEqual(result.log_fields["final_action_reason"], "rest_reminder")

    def test_llm_empty_reply_no_hardcoded_fallback(self) -> None:
        handler = WellnessCareHandler()
        client = _FakeClient({"reply": ""})
        result = handler.decide(
            state=self._fatigue_state(),
            event=Event(type="system_triggered", timestamp=1020, payload={}),
            llm_client=client,
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertEqual(result.actions, [])
        self.assertTrue(result.log_fields.get("fallback_suppressed"))
        self.assertEqual(
            result.log_fields.get("final_action_reason"),
            "llm_failed_no_hardcoded_fallback",
        )

    def test_speaking_does_not_silently_drop_wellness(self) -> None:
        state = self._fatigue_state()
        state.interaction.dialogue_state = "speaking"
        handler = WellnessCareHandler()
        client = _FakeClient({"reply": "歇会儿"})
        result = handler.decide(
            state=state,
            event=Event(type="system_triggered", timestamp=1020, payload={}),
            llm_client=client,
            user_context={},
        )
        # speaking 不应导致静默丢弃：仍产出 suggest_rest 与 speak 动作（交由 TTS 串行）。
        self.assertEqual(result.intents[0].type, "suggest_rest")
        self.assertTrue(any(a.type == "speak" for a in result.actions))

    def test_user_away_no_op(self) -> None:
        state = self._fatigue_state()
        state.user.presence = "away"
        handler = WellnessCareHandler()
        result = handler.decide(
            state=state,
            event=Event(type="system_triggered", timestamp=1020, payload={}),
            llm_client=_FakeClient({"reply": "x"}),
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertEqual(result.log_fields["final_action_reason"], "user_away")


class WellnessPersonalizationTest(unittest.TestCase):
    def _memory_hints_with_profile(self) -> dict:
        merged = {
            GROUP_KEY_BY_TYPE["hobby"]: [
                {"content": "用户喜欢听抒情歌", "confidence": 0.9, "tags": ["music"], "evidence": "抒情歌"},
                {"content": "用户喜欢听相声", "confidence": 0.85, "tags": ["comedy"], "evidence": "相声"},
            ],
            GROUP_KEY_BY_TYPE["work_style"]: [
                {"content": "用户喜欢写代码时短休息", "confidence": 0.8, "tags": ["code"], "evidence": "写代码"}
            ],
            GROUP_KEY_BY_TYPE["preference"]: [
                {"content": "用户希望助手更严格监督", "confidence": 0.85, "tags": [], "evidence": "严格"}
            ],
        }
        hints = build_memory_usage_hints(
            profile={"hobbies": ["篮球"]},
            preferences={},
            memories=merged,
            context_type="wellness_care",
            current_state={"posture": "leaning", "fatigue_level": "none", "emotion": "neutral"},
        )
        return hints

    def test_memory_hints_return_personalization_candidates(self) -> None:
        hints = self._memory_hints_with_profile()
        cands = hints.get("personalization_candidates")
        self.assertIsInstance(cands, list)
        self.assertGreaterEqual(len(cands), 2)
        categories = {c.get("category") for c in cands}
        self.assertGreaterEqual(len(categories), 2)

    def test_posture_prompt_includes_personalization_candidates(self) -> None:
        hints = self._memory_hints_with_profile()
        summary = {
            "should_care": True,
            "recommended_care_focus": "posture",
            "posture": {"current_posture": "leaning"},
            "focus_summary": {"active": False},
        }
        wellness_reply_context = {
            "trigger_focus": "posture",
            "trigger_summary": "体态偏不良（leaning）",
            "personalization_candidates": hints["personalization_candidates"],
            "recent_reminder_texts": [],
            "recent_personalization_used": [],
            "memory_usage_instruction": hints.get("memory_usage_instruction"),
        }
        prompt = build_wellness_prompt(
            wellness_summary=summary,
            selected_intent="suggest_rest",
            care_focus="posture",
            user_context={"memory_usage_hints": hints},
            wellness_reply_context=wellness_reply_context,
            media_ask_allowed=True,
        )
        self.assertIn("personalization_candidates", prompt)
        self.assertIn("用户希望助手更严格监督", prompt)

    def test_prompt_requires_natural_non_template_wording(self) -> None:
        prompt_text = read_prompt(
            Path(__file__).resolve().parents[1] / "src/agent/prompts/wellness_care_check.md"
        )
        self.assertIn("不代表", prompt_text)
        self.assertIn("禁止模板化", prompt_text)
        self.assertIn("连续多轮都用同一类记忆", prompt_text)

    def test_recent_reminder_texts_enter_prompt(self) -> None:
        summary = {
            "should_care": True,
            "recommended_care_focus": "fatigue",
            "fatigue": {"current_level": "high"},
            "focus_summary": {"active": False},
        }
        wellness_reply_context = {
            "trigger_focus": "fatigue",
            "trigger_summary": "疲劳程度 high",
            "personalization_candidates": [],
            "recent_reminder_texts": ["要不要听点抒情歌放松一下"],
            "recent_personalization_used": ["rest"],
            "memory_usage_instruction": "自然融入，不必每轮点名。",
        }
        prompt = build_wellness_prompt(
            wellness_summary=summary,
            selected_intent="suggest_rest",
            care_focus="fatigue",
            user_context={},
            wellness_reply_context=wellness_reply_context,
        )
        self.assertIn("recent_reminder_texts", prompt)
        self.assertIn("抒情歌", prompt)
        base = read_prompt(
            Path(__file__).resolve().parents[1] / "src/agent/prompts/wellness_care_check.md"
        )
        self.assertIn("抒情歌", base)
        self.assertIn("换说法", base)
        self.assertIn("recent_reminder_texts", base)


class EnvironmentHandlerTest(_ProbeResetMixin, unittest.TestCase):
    def _low_light_state(self) -> AgentState:
        state = _present_state()
        state.environment.light_lux = 120
        state.environment.light_level = "low"
        return state

    def test_environment_empty_reply_no_fallback(self) -> None:
        handler = EnvironmentCareHandler()
        client = _FakeClient({"intent": "adjust_environment_feedback", "reply": ""})
        result = handler.decide(
            state=self._low_light_state(),
            event=Event(type="system_triggered", timestamp=2000, payload={}),
            llm_client=client,
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertEqual(result.actions, [])
        self.assertTrue(result.log_fields.get("fallback_suppressed"))

    def test_environment_care_emits_environment_warning(self) -> None:
        handler = EnvironmentCareHandler()
        client = _FakeClient({"intent": "adjust_environment_feedback", "reply": "光线有点暗"})
        result = handler.decide(
            state=self._low_light_state(),
            event=Event(type="system_triggered", timestamp=2000, payload={}),
            llm_client=client,
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "adjust_environment_feedback")
        reasons = {a.payload.get("reason") for a in result.actions}
        self.assertEqual(reasons, {"environment_warning"})

    def test_environment_care_cannot_emit_rest_reminder(self) -> None:
        # LLM 越权返回 suggest_rest 时，被夹断为 no_op，绝不产生休息提醒。
        handler = EnvironmentCareHandler()
        client = _FakeClient({"intent": "suggest_rest", "reply": "去睡觉"})
        result = handler.decide(
            state=self._low_light_state(),
            event=Event(type="system_triggered", timestamp=2000, payload={}),
            llm_client=client,
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertEqual(result.actions, [])

    def test_environment_care_speaking_deferred(self) -> None:
        from src.adapters.voice.arbitration.session_probe import VoiceSessionProbe

        state = self._low_light_state()
        state.interaction.dialogue_state = "speaking"
        VoiceSessionProbe.global_probe().set_user_speak_active(True)
        handler = EnvironmentCareHandler()
        result = handler.decide(
            state=state,
            event=Event(type="system_triggered", timestamp=2000, payload={}),
            llm_client=_FakeClient({"intent": "adjust_environment_feedback", "reply": "x"}),
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertEqual(result.log_fields["final_action_reason"], "tts_speaking_deferred")

    def test_environment_summary_excludes_wellness(self) -> None:
        summary = build_environment_care_summary(self._low_light_state(), check_time=2000)
        for key in ("fatigue", "emotion", "posture"):
            self.assertNotIn(key, summary)
        self.assertTrue(summary["should_consider_care"])


class FallbackSuppressionTest(_ProbeResetMixin, unittest.TestCase):
    @patch("src.agent.decision.behavior_distraction_handler.build_behavior_distraction_summary")
    def test_behavior_llm_failure_no_fallback(self, mock_summary) -> None:
        mock_summary.return_value = {
            "trigger_candidate": True,
            "trigger_detail": {"reason": "phone_use", "phone_use_ratio": 0.5},
            "window_phone_use_events": 2,
            "window_yolo_phone_events": 1,
        }
        handler = BehaviorDistractionHandler()
        result = handler.decide(
            state=_present_state(),
            event=Event(type="system_triggered", timestamp=1020, payload={}),
            llm_client=_FailingClient(),
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertEqual(result.actions, [])
        self.assertTrue(result.log_fields.get("fallback_suppressed"))

    def test_realizer_empty_answer_user_no_actions(self) -> None:
        actions = ActionRealizer().realize([Intent("answer_user", "test")], reply_text="")
        self.assertEqual(actions, [])

    @patch("src.agent.decision.behavior_distraction_handler.build_behavior_distraction_summary")
    def test_behavior_invalid_reply_no_tts(self, mock_summary) -> None:
        mock_summary.return_value = {
            "trigger_candidate": True,
            "trigger_detail": {"reason": "phone_use", "phone_use_ratio": 0.5},
            "window_phone_use_events": 2,
            "window_yolo_phone_events": 1,
        }
        handler = BehaviorDistractionHandler()
        result = handler.decide(
            state=_present_state(),
            event=Event(type="system_triggered", timestamp=1020, payload={}),
            llm_client=_FakeClient({"reply": "有点累，要不要。"}),
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertEqual(result.actions, [])
        self.assertTrue(result.log_fields.get("invalid_llm_reply"))

    def test_wellness_invalid_reply_suppressed(self) -> None:
        handler = WellnessCareHandler()
        state = _present_state()
        state.user.fatigue_level = "high"
        _set_recent(
            state,
            "fatigue",
            [(1000, "high", 0.9), (1010, "high", 0.9), (1020, "high", 0.9)],
        )
        result = handler.decide(
            state=state,
            event=Event(type="system_triggered", timestamp=1020, payload={}),
            llm_client=_FakeClient({"reply": "如果累了的话。"}),
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertTrue(result.log_fields.get("invalid_llm_reply"))

    def test_environment_invalid_reply_suppressed(self) -> None:
        handler = EnvironmentCareHandler()
        state = _present_state()
        state.environment.light_lux = 120
        state.environment.light_level = "low"
        result = handler.decide(
            state=state,
            event=Event(type="system_triggered", timestamp=2000, payload={}),
            llm_client=_FakeClient(
                {"intent": "adjust_environment_feedback", "reply": "注意坐姿，放松。"}
            ),
            user_context={},
        )
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertTrue(result.log_fields.get("invalid_llm_reply"))


if __name__ == "__main__":
    unittest.main()
