from __future__ import annotations

"""LLM 异步记忆抽取与结构化检索的行为测试（覆盖需求第九节）。"""

import tempfile
import unittest
from pathlib import Path

from src.agent.core import build_default_core
from src.agent.event.event_model import Event
from src.agent.memory.memory_extractor import MemoryExtractor
from src.agent.memory.memory_model import make_memory_item
from src.agent.memory.memory_service import MemoryService
from src.agent.policy_config import MemoryPolicy
from tests.fakes.fake_llm_service import FakeLLMService


class _FakeLLMClient:
    """模拟 LLMClient.complete_json(role, prompt) -> dict。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._responses: dict[str, dict] = {}

    def set(self, role: str, payload: dict) -> None:
        self._responses[role] = payload

    def complete_json(self, role: str, prompt: str) -> dict:
        self.calls.append((role, prompt))
        return self._responses.get(role, {"memory_items": []})


def _memory_service(tmp: Path, client: _FakeLLMClient, *, async_write: bool = False) -> MemoryService:
    policy = MemoryPolicy(store_path=str(tmp / "mem.json"), async_write=async_write)
    return MemoryService(policy, extractor=MemoryExtractor(client))


class ExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_extracts_hobby_and_care_strategy(self) -> None:
        client = _FakeLLMClient()
        client.set(
            "memory_extract",
            {
                "memory_items": [
                    {
                        "type": "habit",
                        "content": "用户累的时候喜欢听音乐放松",
                        "evidence": "我累的时候喜欢听点音乐",
                        "confidence": 0.85,
                        "tags": ["music", "fatigue", "relax"],
                    },
                    {
                        "type": "care_strategy",
                        "content": "疲惫时可建议用户听一小段轻松音乐",
                        "evidence": "我累的时候喜欢听点音乐",
                        "confidence": 0.8,
                        "tags": ["music", "fatigue", "care"],
                    },
                ]
            },
        )
        service = _memory_service(self.tmp, client)
        service.submit_speech_memory("default", "我累的时候喜欢听点音乐", 1000)

        types = {item["type"] for item in service.all_memories("default")}
        self.assertIn("habit", types)
        self.assertIn("care_strategy", types)

    def test_extracts_dislike_preference(self) -> None:
        client = _FakeLLMClient()
        client.set(
            "memory_extract",
            {
                "memory_items": [
                    {
                        "type": "dislike",
                        "content": "用户不喜欢频繁提醒",
                        "evidence": "我不喜欢频繁提醒",
                        "confidence": 0.9,
                        "tags": ["reminder"],
                    }
                ]
            },
        )
        service = _memory_service(self.tmp, client)
        service.submit_speech_memory("default", "我不喜欢频繁提醒", 1000)
        types = {item["type"] for item in service.all_memories("default")}
        self.assertIn("dislike", types)

    def test_trivial_text_skips_llm(self) -> None:
        client = _FakeLLMClient()
        service = _memory_service(self.tmp, client)
        for trivial in ("你好", "谢谢", "嗯"):
            service.submit_speech_memory("default", trivial, 1000)
        self.assertEqual(client.calls, [])
        self.assertEqual(service.all_memories("default"), [])

    def test_low_confidence_not_kept(self) -> None:
        client = _FakeLLMClient()
        client.set(
            "memory_extract",
            {"memory_items": [{"type": "fact", "content": "x", "confidence": 0.1}]},
        )
        service = _memory_service(self.tmp, client)
        service.submit_speech_memory("default", "随便说点什么吧", 1000)
        self.assertEqual(service.all_memories("default"), [])

    def test_invalid_type_filtered(self) -> None:
        client = _FakeLLMClient()
        client.set(
            "memory_extract",
            {"memory_items": [{"type": "nonsense", "content": "x", "confidence": 0.9}]},
        )
        service = _memory_service(self.tmp, client)
        service.submit_speech_memory("default", "随便说点什么吧", 1000)
        self.assertEqual(service.all_memories("default"), [])

    def test_duplicate_memory_merged(self) -> None:
        client = _FakeLLMClient()
        client.set(
            "memory_extract",
            {
                "memory_items": [
                    {
                        "type": "hobby",
                        "content": "用户喜欢音乐",
                        "evidence": "我喜欢音乐",
                        "confidence": 0.7,
                        "tags": ["music"],
                    }
                ]
            },
        )
        service = _memory_service(self.tmp, client)
        service.submit_speech_memory("default", "我喜欢音乐", 1000)
        service.submit_speech_memory("default", "我真的很喜欢音乐", 2000)
        hobbies = [m for m in service.all_memories("default") if m["type"] == "hobby"]
        self.assertEqual(len(hobbies), 1)


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.client = _FakeLLMClient()
        self.service = _memory_service(self.tmp, self.client)
        self._seed()

    def _seed(self) -> None:
        seeds = [
            ("care_strategy", "疲惫时可建议听音乐", ["music", "fatigue", "care"]),
            ("hobby", "用户喜欢音乐", ["music"]),
            ("habit", "用户经常熬夜", ["late_night"]),
            ("emotion_pattern", "考试前容易焦虑", ["exam", "anxiety"]),
            ("interaction_style", "喜欢像朋友一样自然提醒", ["friendly"]),
            ("preference", "喜欢简短回答", ["concise"]),
            ("work_style", "喜欢先做难题", ["hard_first"]),
            ("dislike", "不喜欢频繁提醒", ["reminder"]),
        ]
        with self.service._lock:  # type: ignore[attr-defined]
            bucket = self.service._store.setdefault("default", [])  # type: ignore[attr-defined]
            for i, (mtype, content, tags) in enumerate(seeds):
                bucket.append(
                    make_memory_item(
                        user_id="default",
                        type=mtype,
                        content=content,
                        evidence=content,
                        confidence=0.8,
                        tags=tags,
                        timestamp=1000 + i,
                    )
                )
            self.service._save()  # type: ignore[attr-defined]

    def test_wellness_fatigue_prioritizes_care_and_hobby(self) -> None:
        ctx = self.service.retrieve_user_context(
            "default", query="high tired", context_type="wellness_care", top_k=4
        )
        top_types = {item["type"] for item in ctx["top"]}
        self.assertTrue({"care_strategy", "hobby", "habit", "emotion_pattern"} & top_types)
        self.assertIn("care_strategy", top_types)

    def test_wellness_emotion_prioritizes_emotion_pattern(self) -> None:
        ctx = self.service.retrieve_user_context(
            "default", query="anxious stressed", context_type="wellness_care", top_k=4
        )
        top_types = {item["type"] for item in ctx["top"]}
        self.assertTrue({"emotion_pattern", "care_strategy"} & top_types)

    def test_speech_prioritizes_interaction_style(self) -> None:
        ctx = self.service.retrieve_user_context(
            "default", query="", context_type="speech", top_k=4
        )
        top_types = {item["type"] for item in ctx["top"]}
        self.assertTrue({"interaction_style", "preference", "work_style"} & top_types)

    def test_by_type_grouping_uses_plural_keys(self) -> None:
        ctx = self.service.retrieve_user_context(
            "default", query="", context_type="wellness_care", top_k=8
        )
        self.assertIn("care_strategies", ctx["by_type"])
        self.assertIn("hobbies", ctx["by_type"])

    def test_retrieval_is_type_diverse_not_single_type(self) -> None:
        # 注入多条同类型（hobby）记忆，验证 top_k 不会被同一类型占满。
        with self.service._lock:  # type: ignore[attr-defined]
            bucket = self.service._store["default"]  # type: ignore[attr-defined]
            for i in range(5):
                bucket.append(
                    make_memory_item(
                        user_id="default",
                        type="hobby",
                        content=f"额外爱好{i}",
                        evidence="x",
                        confidence=0.9,
                        tags=[f"extra{i}"],
                        timestamp=2000 + i,
                    )
                )
            self.service._save()  # type: ignore[attr-defined]
        ctx = self.service.retrieve_user_context(
            "default", query="", context_type="wellness_care", top_k=4
        )
        top_types = [item["type"] for item in ctx["top"]]
        # 不应 4 条全是 hobby；至少跨越多种类型。
        self.assertLessEqual(top_types.count("hobby"), 2)
        self.assertGreaterEqual(len(set(top_types)), 3)

    def test_recently_used_memory_rotates_out(self) -> None:
        # 第一次检索用过的高分记忆，短期内再次检索应被轮换降权，给其它记忆机会。
        first = self.service.retrieve_user_context(
            "default", query="", context_type="wellness_care", top_k=2
        )
        second = self.service.retrieve_user_context(
            "default", query="", context_type="wellness_care", top_k=2
        )
        first_contents = {item["content"] for item in first["top"]}
        second_contents = {item["content"] for item in second["top"]}
        # 两次检索不应完全相同（轮换生效）。
        self.assertNotEqual(first_contents, second_contents)


class CoreIntegrationTest(unittest.TestCase):
    def make_core(self, fake: FakeLLMService, *, memory_async: bool = False):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        core = build_default_core(
            store_path=base / "state.json",
            profile_store_path=base / "profiles.json",
            memory_store_path=base / "memory.json",
            timer_background=False,
            llm_service=fake,
            memory_async=memory_async,
        )
        self.addCleanup(core.shutdown)
        return core

    def test_speech_prompt_contains_interaction_style(self) -> None:
        fake = FakeLLMService()
        fake.set_response("speech_recognized", {"intent": "answer_user", "reply": "嗯。"})
        core = self.make_core(fake)
        user_id = core.state.current_user_id
        # 预置一条 interaction_style 记忆。
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
            Event(type="speech_recognized", timestamp=1000, payload={"text": "今天有点累"})
        )
        speech_prompts = [p for (role, p) in fake.prompts if role == "speech_recognized"]
        self.assertTrue(speech_prompts)
        self.assertIn("喜欢像朋友一样自然提醒", speech_prompts[-1])

    def test_wellness_prompt_contains_structured_memories(self) -> None:
        fake = FakeLLMService()
        core = self.make_core(fake)
        user_id = core.state.current_user_id
        with core.memory._lock:  # type: ignore[attr-defined]
            core.memory._store.setdefault(user_id, []).append(  # type: ignore[attr-defined]
                make_memory_item(
                    user_id=user_id,
                    type="hobby",
                    content="用户喜欢打篮球",
                    evidence="喜欢打篮球",
                    confidence=0.85,
                    tags=["basketball", "sport"],
                    timestamp=1,
                )
            )
        core.handle_event(Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"}))
        # 强负面情绪即时触发 wellness_care（无需连续窗口），进入 LLM。
        core.handle_event(
            Event(
                type="user_emotion_updated",
                timestamp=2,
                payload={"emotion": "angry", "confidence": 0.9},
            )
        )
        core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1000,
                payload={"trigger": "wellness_care_check", "source": "agent_autonomy"},
            )
        )
        wellness_prompts = [p for (role, p) in fake.prompts if role == "wellness_care_check"]
        self.assertTrue(wellness_prompts)
        self.assertIn("用户喜欢打篮球", wellness_prompts[-1])

    def test_memory_llm_failure_does_not_break_handle_event(self) -> None:
        class _BoomService(FakeLLMService):
            def complete_json(self, role: str, prompt: str) -> str:
                if role == "memory_extract":
                    raise RuntimeError("boom")
                return super().complete_json(role, prompt)

        fake = _BoomService()
        fake.set_response("speech_recognized", {"intent": "answer_user", "reply": "好。"})
        core = self.make_core(fake, memory_async=True)
        actions, _ = core.handle_event(
            Event(type="speech_recognized", timestamp=1000, payload={"text": "我喜欢先做难题"})
        )
        # 主链路正常完成。
        self.assertEqual(core.last_decision_result.source, "speech_llm")
        self.assertTrue(core.memory.wait_for_idle(timeout=5.0))
        self.assertEqual(core.memory.all_memories(core.state.current_user_id), [])

    def test_speech_submit_is_non_blocking(self) -> None:
        import threading
        import time

        release = threading.Event()

        class _SlowService(FakeLLMService):
            def complete_json(self, role: str, prompt: str) -> str:
                if role == "memory_extract":
                    release.wait(timeout=5.0)
                return super().complete_json(role, prompt)

        fake = _SlowService()
        fake.set_response("speech_recognized", {"intent": "answer_user", "reply": "好。"})
        core = self.make_core(fake, memory_async=True)
        start = time.monotonic()
        core.handle_event(
            Event(type="speech_recognized", timestamp=1000, payload={"text": "我喜欢先做难题"})
        )
        elapsed = time.monotonic() - start
        release.set()
        # 即使 memory LLM 阻塞 5s，主链路也应快速返回（远小于阻塞时长）。
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
