from __future__ import annotations

"""重构后 agent 层的行为测试。

围绕 LLM 决策入口（speech_recognized / wellness_care_check / behavior_distraction_check /
environment_care_check / sensor_status_report）、规则事件、仅更新状态的事件、动作闭集、
设备执行边界、调度器。
"""

import tempfile
import unittest
from pathlib import Path
from typing import get_args

from src.agent.action.action_model import Action
from src.agent.action.types import ACTION_TYPE_SET
from src.agent.core import build_default_core
from src.agent.event.event_model import Event
from src.agent.event.router import EventRouter
from src.agent.event.types import EventType
from src.agent.scheduler.autonomous_scheduler import AutonomousScheduler
from src.agent.state.agent_state import AgentState
from tests.fakes.fake_llm_service import FakeLLMService

_DELETED_EVENT_TYPES = [
    "user_text_input",
    "user_switched",
    "user_profile_updated",
    "user_preference_update_requested",
    "break_suggestion_accepted",
    "break_suggestion_rejected",
    "voice_volume_changed",
    "voice_timbre_changed",
    "voice_speed_changed",
    "display_sensor_updated",
]

_DELETED_ACTION_TYPES = [
    "set_tts_voice",
    "set_tts_speed",
    "render_pet_expression",
    "set_light_state",
    "start_voice_capture",
    "stop_voice_capture",
]

_OLD_TRIGGERS = [
    "focus_health_check",
    "environment_check",
    "periodic_check",
    "user_idle_check",
    "wellness_check",
]


class CaptureOutput:
    """记录被执行的输出动作（speak / display / set_tts_volume）。"""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> None:
        self.actions.append(action)


class _CoreTestBase(unittest.TestCase):
    def make_core(self, fake: FakeLLMService | None = None, *, memory_async: bool = False):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.output = CaptureOutput()
        self.fake = fake or FakeLLMService()
        core = build_default_core(
            output=self.output,
            store_path=base / "state.json",
            profile_store_path=base / "profiles.json",
            memory_store_path=base / "memory.json",
            timer_background=False,
            llm_service=self.fake,
            memory_async=memory_async,
        )
        self.addCleanup(core.shutdown)
        self.addCleanup(self._tmp.cleanup)
        return core


class ProtocolClosureTest(unittest.TestCase):
    def test_event_types_match_required_set(self) -> None:
        types = set(get_args(EventType))
        for deleted in _DELETED_EVENT_TYPES:
            self.assertNotIn(deleted, types, f"已删除事件仍存在: {deleted}")
        self.assertIn("speech_recognized", types)
        self.assertIn("system_triggered", types)
        self.assertEqual(len(types), 20)

    def test_action_types_match_required_set(self) -> None:
        self.assertEqual(
            ACTION_TYPE_SET,
            frozenset(
                {
                    "speak",
                    "display",
                    "start_timer",
                    "stop_timer",
                    "set_tts_volume",
                    "play_media",
                    "stop_media",
                    "pause_media",
                    "resume_media",
                    "next_media",
                }
            ),
        )
        for deleted in _DELETED_ACTION_TYPES:
            self.assertNotIn(deleted, ACTION_TYPE_SET, f"已删除动作仍存在: {deleted}")


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = EventRouter()

    def test_speech_goes_to_speech_llm(self) -> None:
        decision = self.router.classify(
            Event(type="speech_recognized", timestamp=1, payload={"text": "你好"})
        )
        self.assertEqual(decision.kind, "speech_llm")
        self.assertTrue(decision.uses_llm)

    def test_behavior_distraction_check_goes_to_behavior_distraction(self) -> None:
        decision = self.router.classify(
            Event(
                type="system_triggered",
                timestamp=1,
                payload={"trigger": "behavior_distraction_check", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(decision.kind, "behavior_distraction")
        self.assertTrue(decision.uses_llm)

    def test_removed_periodic_trigger_is_not_an_llm_entry(self) -> None:
        decision = self.router.classify(
            Event(
                type="system_triggered",
                timestamp=1,
                payload={"trigger": "periodic_state_check", "source": "agent_autonomy"},
            )
        )
        self.assertNotEqual(decision.kind, "periodic_state")
        self.assertFalse(decision.uses_llm)

    def test_old_triggers_do_not_enter_llm(self) -> None:
        for trigger in _OLD_TRIGGERS:
            decision = self.router.classify(
                Event(
                    type="system_triggered",
                    timestamp=1,
                    payload={"trigger": trigger, "source": "agent_autonomy"},
                )
            )
            self.assertEqual(decision.kind, "state_only", f"{trigger} 不应进入 LLM")
            self.assertFalse(decision.uses_llm)

    def test_structured_control_uses_rule(self) -> None:
        for event_type in ("focus_start_requested", "focus_stop_requested", "timer_finished"):
            decision = self.router.classify(Event(type=event_type, timestamp=1, payload={}))
            self.assertEqual(decision.kind, "rule")
            self.assertFalse(decision.uses_llm)

    def test_perception_events_state_only(self) -> None:
        for event_type in (
            "user_fatigue_updated",
            "user_emotion_updated",
            "user_presence_updated",
            "voice_wake_detected",
            "tts_started",
            "timer_ticked",
        ):
            decision = self.router.classify(Event(type=event_type, timestamp=1, payload={}))
            self.assertEqual(decision.kind, "state_only")
            self.assertFalse(decision.uses_llm)


class SpeechFlowTest(_CoreTestBase):
    def test_speech_enters_speech_llm_handler(self) -> None:
        fake = FakeLLMService()
        fake.set_response("speech_recognized", {"intent": "answer_user", "reply": "你好呀。"})
        core = self.make_core(fake)
        actions, _results = core.handle_event(
            Event(type="speech_recognized", timestamp=1000, payload={"text": "你好"})
        )
        self.assertEqual(core.last_decision_result.source, "speech_llm")
        self.assertEqual(fake.calls, ["speech_recognized"])
        self.assertEqual({a.type for a in actions}, {"speak", "display"})

    def test_speech_start_focus_starts_timer(self) -> None:
        fake = FakeLLMService()
        fake.set_response(
            "speech_recognized",
            {"intent": "start_focus", "reply": "好的，开始专注。", "duration_sec": 1500},
        )
        core = self.make_core(fake)
        actions, _ = core.handle_event(
            Event(type="speech_recognized", timestamp=1000, payload={"text": "开始专注"})
        )
        self.assertIn("start_timer", {a.type for a in actions})


class PerceptionFlowTest(_CoreTestBase):
    def test_fatigue_updates_state_without_llm(self) -> None:
        core = self.make_core()
        core.handle_event(
            Event(
                type="user_fatigue_updated",
                timestamp=1000,
                payload={"fatigue_level": "high", "confidence": 0.9},
            )
        )
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(core.last_decision_result.source, "state_only")
        self.assertEqual(core.state.user.fatigue_level, "high")

    def test_emotion_updates_state_without_llm(self) -> None:
        core = self.make_core()
        core.handle_event(
            Event(
                type="user_emotion_updated",
                timestamp=1000,
                payload={"emotion": "stressed", "confidence": 0.8},
            )
        )
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(core.state.user.emotion, "stressed")


class AutonomyFallbackTest(_CoreTestBase):
    def test_unknown_or_removed_trigger_does_not_call_llm(self) -> None:
        core = self.make_core()
        for trigger in ("wellness_check", "periodic_state_check", "user_idle_check"):
            core.handle_event(
                Event(
                    type="system_triggered",
                    timestamp=1000,
                    payload={"trigger": trigger, "source": "agent_autonomy"},
                )
            )
            self.assertEqual(core.last_decision_result.source, "state_only")
        self.assertEqual(self.fake.calls, [])


class RuleFlowTest(_CoreTestBase):
    def test_focus_events_and_timer_finished_use_rule(self) -> None:
        core = self.make_core()
        core.handle_event(Event(type="focus_start_requested", timestamp=1000, payload={"duration_sec": 1500}))
        self.assertEqual(core.last_decision_result.source, "rule")
        self.assertFalse(core.last_decision_result.used_llm)
        # focus_start / stop 仍是纯规则，不调用 LLM。
        self.assertEqual(self.fake.calls, [])

        core.handle_event(Event(type="timer_finished", timestamp=2600, payload={"remaining_sec": 0}))
        # 计时语义仍由规则确定（停表/完成），但专注结束会用 LLM 生成一句个性化轮换关怀文案。
        self.assertEqual(core.last_decision_result.source, "rule")
        self.assertEqual(self.fake.calls, ["focus_complete_care"])

    def test_focus_complete_personalizes_and_rotates_interest(self) -> None:
        # LLM 回写 prompt 里本轮 recommended_content 的兴趣标签，验证个性化与轮换。
        def responder(prompt: str) -> str:
            import json as _json
            import re

            labels = re.findall(r'"label"\s*:\s*"(用户喜欢[^"]+)"', prompt)
            label = labels[0] if labels else ""
            return _json.dumps({"reply": f"专注结束啦，{label}"}, ensure_ascii=False)

        fake = FakeLLMService()
        fake.responses["focus_complete_care"] = responder
        core = self.make_core(fake=fake)
        core.memory._store["default"] = []  # type: ignore[attr-defined]
        from src.agent.memory.memory_model import MemoryItem

        for content, tags in (("用户喜欢打篮球", ["ball"]), ("用户喜欢听笑话", ["joke"])):
            core.memory._store["default"].append(  # type: ignore[attr-defined]
                MemoryItem(user_id="default", type="hobby", content=content, evidence=content, confidence=0.85, tags=tags)
            )

        replies: list[str] = []
        for ts in (2600, 5200):
            self.output.actions.clear()
            core.handle_event(Event(type="focus_start_requested", timestamp=ts - 100, payload={"duration_sec": 60}))
            core.handle_event(Event(type="timer_finished", timestamp=ts, payload={"remaining_sec": 0}))
            spoken = [a.payload.get("text", "") for a in self.output.actions if a.type == "speak"][-1]
            replies.append(spoken)

        # 两次专注结束均走 LLM 个性化；care_rotation_index 随轮次递增。
        self.assertTrue(all("专注结束啦" in r and "用户喜欢" in r for r in replies))
        self.assertEqual(core.state.interaction.care_rotation_index, 2)


class DeviceAdapterTest(_CoreTestBase):
    def test_unsupported_action_returns_unsupported(self) -> None:
        core = self.make_core()
        for deleted in _DELETED_ACTION_TYPES:
            result = core.device_adapter.execute(Action(type=deleted, payload={}), timestamp=1)
            self.assertFalse(result.success, f"{deleted} 不应假成功")
            self.assertEqual(result.reason, "unsupported_action")

    def test_supported_action_executes(self) -> None:
        core = self.make_core()
        result = core.device_adapter.execute(Action(type="speak", payload={"text": "hi"}), timestamp=1)
        self.assertTrue(result.success)
        self.assertEqual(self.output.actions[-1].type, "speak")


class SchedulerTest(unittest.TestCase):
    def test_emits_behavior_then_wellness(self) -> None:
        now = {"t": 1000}
        emitted: list[Event] = []
        scheduler = AutonomousScheduler(
            state_provider=AgentState,
            event_sink=emitted.append,
            time_fn=lambda: now["t"],
        )
        # 起始后未到间隔，不发。
        self.assertEqual(scheduler.run_due(), [])
        now["t"] += 19
        self.assertEqual(scheduler.run_due(), [])
        now["t"] += 1  # 累计 20 秒：分心检查（20s）到点。
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].payload["trigger"], "behavior_distraction_check")

        # 继续推进，分心(20s)/疲劳情绪关怀(30s)/环境关怀(60s) 都会按优先级先后触发。
        triggers: set[str] = set()
        for _ in range(80):
            now["t"] += 1
            for ev in scheduler.run_due():
                triggers.add(ev.payload["trigger"])
        self.assertIn("behavior_distraction_check", triggers)
        self.assertIn("wellness_care_check", triggers)
        self.assertIn("environment_care_check", triggers)
        for event in emitted:
            self.assertEqual(event.type, "system_triggered")
            self.assertEqual(event.payload["source"], "agent_autonomy")


class MemoryAsyncTest(_CoreTestBase):
    def test_speech_memory_write_is_async_and_non_blocking(self) -> None:
        fake = FakeLLMService()
        fake.set_response("speech_recognized", {"intent": "answer_user", "reply": "记住啦。"})
        fake.set_response(
            "memory_extract",
            {
                "memory_items": [
                    {
                        "type": "dislike",
                        "content": "用户不喜欢频繁提醒",
                        "evidence": "用户说：以后请不要频繁提醒我",
                        "confidence": 0.88,
                        "tags": ["reminder", "frequency"],
                    }
                ]
            },
        )
        core = self.make_core(fake, memory_async=True)
        user_id = core.state.current_user_id

        # 含偏好的语音应被异步抽取，handle_event 立即返回，不等待 memory LLM。
        core.handle_event(
            Event(type="speech_recognized", timestamp=1000, payload={"text": "以后请不要频繁提醒我"})
        )
        self.assertTrue(core.memory.wait_for_idle(timeout=5.0))
        stored = core.memory.all_memories(user_id)
        self.assertTrue(any("不喜欢频繁提醒" in m.get("content", "") for m in stored))

    def test_trivial_speech_not_remembered(self) -> None:
        core = self.make_core(memory_async=False)
        user_id = core.state.current_user_id
        core.handle_event(Event(type="speech_recognized", timestamp=1000, payload={"text": "你好"}))
        self.assertEqual(core.memory.all_memories(user_id), [])


class RuntimePersistThrottleTest(_CoreTestBase):
    def test_high_frequency_perception_throttles_save(self) -> None:
        core = self.make_core()
        save_count = 0
        original_save = core.store.save_state

        def counting_save(state, **kwargs):
            nonlocal save_count
            save_count += 1
            return original_save(state, **kwargs)

        core.store.save_state = counting_save  # type: ignore[method-assign]
        after_make = save_count

        for i in range(10):
            core.handle_event(
                Event(
                    type="user_fatigue_updated",
                    timestamp=1000 + i,
                    payload={"fatigue_level": "high", "confidence": 0.9},
                )
            )

        self.assertEqual(save_count - after_make, 1)
        core.shutdown()


class FocusTimerResumeTest(unittest.TestCase):
    def test_remaining_recomputed_from_start_ts_not_stale_json(self) -> None:
        import time

        core = build_default_core(timer_background=False)
        try:
            now = int(time.time())
            core.state.focus.active = True
            core.state.focus.start_ts = now - 180
            core.state.focus.target_duration_sec = 600
            core.state.focus.remaining_sec = 999  # 故意写错的持久化值
            with core._lock:
                remaining = core._compute_focus_remaining_locked()
            self.assertGreaterEqual(remaining, 415)
            self.assertLessEqual(remaining, 425)
        finally:
            core.shutdown()

    def test_resume_focus_timer_starts_background_timer(self) -> None:
        core = build_default_core(timer_background=True)
        try:
            now = int(__import__("time").time())
            core.state.focus.active = True
            core.state.focus.start_ts = now - 30
            core.state.focus.target_duration_sec = 420
            core.state.focus.remaining_sec = 420
            resumed = core.resume_focus_timer_if_needed()
            self.assertIsNotNone(resumed)
            self.assertGreater(resumed, 380)
            self.assertTrue(core.timer_service.is_active())
        finally:
            core.shutdown()

    def test_cold_start_clears_persisted_focus_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store_path = base / "state.json"
            from src.storage.json_store import JsonStore

            persisted = AgentState()
            persisted.focus.active = True
            persisted.focus.start_ts = 1_700_000_000
            persisted.focus.target_duration_sec = 420
            persisted.focus.remaining_sec = 127
            persisted.interaction.in_conversation = True
            persisted.interaction.dialogue_state = "thinking"
            persisted.interaction.mode = "focus"
            JsonStore(store_path).save_state(persisted)

            core = build_default_core(
                store_path=store_path,
                profile_store_path=base / "profiles.json",
                memory_store_path=base / "memory.json",
                timer_background=False,
                llm_service=FakeLLMService(),
            )
            try:
                self.assertFalse(core.state.focus.active)
                self.assertIsNone(core.state.focus.start_ts)
                self.assertEqual(core.state.interaction.dialogue_state, "idle")
                self.assertEqual(core.state.interaction.mode, "normal")
                self.assertFalse(core.state.interaction.in_conversation)
                self.assertIsNone(core.resume_focus_timer_if_needed())
            finally:
                core.shutdown()


if __name__ == "__main__":
    unittest.main()
