from __future__ import annotations

"""memory_usage_hints 行为测试（覆盖需求第十二节）。

均基于真实数据结构（profile / preference / memory by_type 分组），
不依赖任何固定爱好关键词（如“篮球”），任意 hobby/preference 都应同样工作。
"""

import unittest

from src.agent.context.memory_usage_hints import build_memory_usage_hints
from src.agent.memory.memory_model import GROUP_KEY_BY_TYPE
from tests.fakes.fake_llm_service import FakeLLMService


def _mem(mtype: str, content: str, *, confidence: float = 0.8, tags=None) -> dict:
    """构造一条 retrieve by_type 风格的记忆条目（按 plural 分组键归组）。"""

    return {
        GROUP_KEY_BY_TYPE[mtype]: [
            {"content": content, "confidence": confidence, "tags": list(tags or []), "evidence": content}
        ]
    }


def _merge(*memories: dict) -> dict:
    merged: dict = {}
    for m in memories:
        for key, items in m.items():
            merged.setdefault(key, []).extend(items)
    return merged


class SpeechFocusTest(unittest.TestCase):
    def test_vision_fatigue_does_not_drive_speech_focus(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories={},
            context_type="speech",
            current_state={"fatigue_level": "high", "emotion": "neutral"},
            user_query="我更喜欢科比",
        )
        self.assertEqual(hints["focus"], "general")

    def test_user_utterance_fatigue_drives_speech_focus(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories={},
            context_type="speech",
            current_state={"fatigue_level": "none"},
            user_query="今天有点累",
        )
        self.assertEqual(hints["focus"], "fatigue")


class ProfileHobbyTest(unittest.TestCase):
    def test_arbitrary_profile_hobby_enters_candidates(self) -> None:
        # 用一个不在任何硬编码列表里的爱好，验证通用性。
        for hobby in ("京剧", "羽毛球", "做饭"):
            hints = build_memory_usage_hints(
                profile={"hobbies": [hobby]},
                preferences={},
                memories={},
                context_type="wellness_care",
                current_state={"fatigue_level": "high"},
            )
            labels = [c["label"] for c in hints["suggestion_candidates"]]
            self.assertIn(hobby, labels, f"hobby {hobby} 应进入候选")
            self.assertEqual(hints["focus"], "fatigue")


class PersonalizationCandidatesTest(unittest.TestCase):
    def _rich_memories(self) -> dict:
        return _merge(
            _mem("hobby", "用户喜欢听抒情歌", tags=["music"]),
            _mem("hobby", "用户喜欢听相声", tags=["comedy"]),
            _mem("hobby", "用户喜欢打篮球", tags=["ball"]),
            _mem("work_style", "用户喜欢短时间专注学习", tags=["study"]),
            _mem("preference", "用户希望助手更严格地监督坐姿", confidence=0.85),
            _mem("interaction_style", "用户希望温柔提醒", confidence=0.8),
        )

    def test_wellness_returns_personalization_candidates(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=self._rich_memories(),
            context_type="wellness_care",
            current_state={"fatigue_level": "high"},
        )
        cands = hints.get("personalization_candidates")
        self.assertIsInstance(cands, list)
        self.assertGreaterEqual(len(cands), 3)
        self.assertIsNone(hints.get("recommended_content"))
        self.assertTrue(str(hints.get("memory_usage_instruction") or "").strip())

    def test_posture_focus_includes_personalization_candidates(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=self._rich_memories(),
            context_type="wellness_care",
            current_state={"posture": "leaning"},
        )
        self.assertEqual(hints["focus"], "posture")
        cands = hints.get("personalization_candidates") or []
        self.assertGreaterEqual(len(cands), 2)
        labels = {c["label"] for c in cands}
        self.assertTrue(
            "用户喜欢打篮球" in labels or "用户喜欢短时间专注学习" in labels,
            f"姿态场景也应拿到广义画像候选，实际：{labels}",
        )

    def test_candidates_span_multiple_categories(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=self._rich_memories(),
            context_type="wellness_care",
            current_state={"fatigue_level": "high"},
        )
        categories = {c.get("category") for c in hints.get("personalization_candidates") or []}
        self.assertGreaterEqual(len(categories), 2, f"候选应跨多类，实际类别：{categories}")
        relaxing = [c for c in hints["personalization_candidates"] if c.get("category") == "relaxing_content"]
        self.assertLessEqual(len(relaxing), 1, "放松娱乐类应去重，不应全是音乐/相声")


class FocusSessionGuardTest(unittest.TestCase):
    def test_wellness_includes_focus_habits_for_llm_when_not_focusing(self) -> None:
        memories = _merge(
            _mem("hobby", "用户喜欢听笑话", tags=["joke"]),
            _mem("habit", "用户倾向于设定30分钟的专注时长", tags=["focus", "time"]),
            _mem("work_style", "用户喜欢以30分钟为单位进行专注工作", tags=["focus"]),
        )
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=memories,
            context_type="wellness_care",
            current_state={"fatigue_level": "high", "focus_active": False},
            rotation_seed=0,
        )
        labels = [c["label"] for c in hints.get("personalization_candidates") or []]
        self.assertIn("用户喜欢听笑话", labels)
        self.assertIn("用户倾向于设定30分钟的专注时长", labels)
        self.assertIn("用户喜欢以30分钟为单位进行专注工作", labels)

    def test_wellness_keeps_focus_habits_while_focusing(self) -> None:
        memories = _mem("habit", "用户倾向于设定30分钟的专注时长", tags=["focus"])
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=memories,
            context_type="wellness_care",
            current_state={"fatigue_level": "high", "focus_active": True},
        )
        labels = [c["label"] for c in hints.get("personalization_candidates") or []]
        self.assertIn("用户倾向于设定30分钟的专注时长", labels)


class MemoryHobbyPreferenceTest(unittest.TestCase):
    def test_memory_hobby_enters_candidates_on_fatigue(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=_mem("hobby", "用户喜欢散步", tags=["walk"]),
            context_type="wellness_care",
            current_state={"fatigue_level": "moderate"},
        )
        labels = [c["label"] for c in hints["suggestion_candidates"]]
        self.assertIn("用户喜欢散步", labels)

    def test_memory_preference_enters_candidates(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=_mem("preference", "喜欢轻松的内容"),
            context_type="wellness_care",
            current_state={"fatigue_level": "high"},
        )
        labels = [c["label"] for c in hints["suggestion_candidates"]]
        self.assertIn("喜欢轻松的内容", labels)


class CareStrategyPriorityTest(unittest.TestCase):
    def test_care_strategy_outranks_plain_fact(self) -> None:
        memories = _merge(
            _mem("fact", "用户是大学生", confidence=0.9),
            _mem("care_strategy", "累了可以陪用户聊两句轻松话题", confidence=0.7),
        )
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=memories,
            context_type="wellness_care",
            current_state={"fatigue_level": "high"},
        )
        cands = hints["suggestion_candidates"]
        # fact 不是建议候选类型，care_strategy 必须出现且排在最前。
        self.assertTrue(cands)
        self.assertEqual(cands[0]["category"], "care_strategy")
        self.assertNotIn("用户是大学生", [c["label"] for c in cands])


class DislikeAvoidTest(unittest.TestCase):
    def test_dislike_enters_avoid_and_shortens_tone(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=_mem("dislike", "不喜欢被频繁打断", confidence=0.9),
            context_type="wellness_care",
            current_state={"fatigue_level": "high"},
        )
        avoid_labels = [a["label"] for a in hints["avoid_patterns"]]
        self.assertIn("不喜欢被频繁打断", avoid_labels)
        self.assertIn(hints["preferred_tone"], {"short", "gentle_short"})

    def test_disliked_topics_from_profile_enters_avoid(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={"disliked_topics": ["说教式提醒"]},
            memories={},
            context_type="speech",
            current_state={"emotion": "sad"},
        )
        self.assertIn("说教式提醒", [a["label"] for a in hints["avoid_patterns"]])


class EnvironmentScopeTest(unittest.TestCase):
    def test_environment_care_ignores_non_environment_hobby(self) -> None:
        hints = build_memory_usage_hints(
            profile={"hobbies": ["篮球", "音乐"]},
            preferences={},
            memories=_mem("hobby", "喜欢打球"),
            context_type="environment_care",
            current_state={"noise_level": "high"},
        )
        self.assertEqual(hints["suggestion_candidates"], [])
        self.assertEqual(hints["personalization_level"], "none")
        self.assertEqual(hints["focus"], "noise")


class BehaviorDistractionScopeTest(unittest.TestCase):
    def test_distraction_uses_work_style_not_entertainment(self) -> None:
        memories = _merge(
            _mem("work_style", "喜欢沉浸式长时间专注", tags=["focus"]),
            _mem("hobby", "喜欢听音乐", tags=["music"]),
        )
        hints = build_memory_usage_hints(
            profile={"hobbies": ["音乐"]},
            preferences={},
            memories=memories,
            context_type="behavior_distraction",
            current_state={"behavior": "phone_use"},
        )
        labels = [c["label"] for c in hints["suggestion_candidates"]]
        self.assertIn("喜欢沉浸式长时间专注", labels)
        # 不应把娱乐/音乐爱好作为分心提醒候选。
        self.assertNotIn("喜欢听音乐", labels)
        self.assertNotIn("音乐", labels)
        self.assertEqual(hints["focus"], "distraction")


class RecentlyUsedAngleTest(unittest.TestCase):
    def test_recommended_angle_avoids_recent(self) -> None:
        runtime = {"recent_reminders": [{"reason": "distraction_reminder", "timestamp": 1}]}
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories={},
            runtime=runtime,
            context_type="behavior_distraction",
            current_state={"behavior": "phone_use"},
        )
        self.assertIn("refocus", hints["recently_used_angles"])
        self.assertNotEqual(hints["recommended_angle"], "refocus")


class NoMemoryTest(unittest.TestCase):
    def test_no_memory_gives_none_and_no_fabrication(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories={},
            context_type="wellness_care",
            current_state={"fatigue_level": "high"},
        )
        self.assertEqual(hints["suggestion_candidates"], [])
        self.assertEqual(hints["personalization_level"], "none")
        # 无候选时仍给出场景默认方向，但不编造偏好。
        self.assertEqual(hints["recommended_angle"], "rest")

    def test_strong_when_high_confidence_candidate(self) -> None:
        hints = build_memory_usage_hints(
            profile={},
            preferences={},
            memories=_mem("care_strategy", "累了可以听点轻松的", confidence=0.85),
            context_type="wellness_care",
            current_state={"fatigue_level": "high"},
        )
        self.assertEqual(hints["personalization_level"], "strong")


class CoreInjectionTest(unittest.TestCase):
    """端到端验证：AgentCore._user_context 把 memory_usage_hints 注入 LLM prompt。"""

    def _core(self, fake):
        import tempfile
        from pathlib import Path

        from src.agent.core import build_default_core

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        core = build_default_core(
            store_path=base / "state.json",
            profile_store_path=base / "profiles.json",
            memory_store_path=base / "memory.json",
            timer_background=False,
            llm_service=fake,
            memory_async=False,
        )
        self.addCleanup(core.shutdown)
        return core

    def test_speech_prompt_contains_memory_usage_hints(self) -> None:
        from src.agent.core.models import Event
        from src.agent.memory.memory_model import make_memory_item

        fake = FakeLLMService()
        fake.set_response("speech_recognized", {"intent": "answer_user", "reply": "嗯。"})
        core = self._core(fake)
        user_id = core.state.current_user_id
        with core.memory._lock:  # type: ignore[attr-defined]
            core.memory._store.setdefault(user_id, []).append(  # type: ignore[attr-defined]
                make_memory_item(
                    user_id=user_id,
                    type="interaction_style",
                    content="喜欢像朋友一样自然提醒",
                    evidence="像朋友一样",
                    confidence=0.85,
                    tags=["friendly"],
                    timestamp=1,
                )
            )
        core.handle_event(
            Event(type="speech_recognized", timestamp=1000, payload={"text": "我有点累"})
        )
        prompts = [p for (role, p) in fake.prompts if role == "speech_recognized"]
        self.assertTrue(prompts)
        self.assertIn("memory_usage_hints", prompts[-1])
        self.assertIn("recommended_angle", prompts[-1])


if __name__ == "__main__":
    unittest.main()
