import inspect
import unittest

from src.agent import reducer
from src.agent.event import Event
from src.agent.reducer import REDUCERS, reduce_state
from src.agent.state import AgentState


class ReducerTestCase(unittest.TestCase):
    def test_user_text_and_speech_update_interaction_state(self) -> None:
        """文本和语音识别事件只更新对话状态。"""
        state = AgentState()

        reduce_state(state, Event(type="user_text_input", timestamp=10, payload={"text": "你好"}))
        self.assertTrue(state.interaction.in_conversation)
        self.assertEqual(state.interaction.dialogue_state, "thinking")
        self.assertEqual(state.interaction.last_user_time, 10)

        reduce_state(state, Event(type="speech_recognized", timestamp=20, payload={"text": ""}))
        self.assertFalse(state.interaction.in_conversation)
        self.assertEqual(state.interaction.dialogue_state, "idle")
        self.assertEqual(state.interaction.last_user_time, 20)

    def test_focus_start_stop_and_session_record(self) -> None:
        """专注开始和手动停止会更新 focus 状态并归档 session。"""
        state = AgentState()

        reduce_state(
            state,
            Event(type="focus_start_requested", timestamp=100, payload={"duration_sec": 1500, "source": "test"}),
        )
        self.assertTrue(state.focus.active)
        self.assertEqual(state.interaction.mode, "focus")
        self.assertEqual(state.focus.start_ts, 100)
        self.assertEqual(state.focus.remaining_sec, 1500)
        self.assertEqual(state.focus.triggered_by, "test")

        reduce_state(state, Event(type="focus_stop_requested", timestamp=220, payload={}))
        self.assertFalse(state.focus.active)
        self.assertEqual(state.interaction.mode, "normal")
        self.assertEqual(state.runtime_history.focus_session_count, 1)
        self.assertEqual(state.runtime_history.focus_sessions[-1]["actual_duration_sec"], 120)
        self.assertEqual(state.runtime_history.focus_sessions[-1]["reason"], "manual_stop")

    def test_timer_ticked_updates_elapsed_and_remaining(self) -> None:
        """timer tick 只更新专注计时字段。"""
        state = AgentState()
        reduce_state(state, Event(type="focus_start_requested", timestamp=100, payload={"duration_sec": 1500}))

        reduce_state(state, Event(type="timer_ticked", timestamp=160, payload={"remaining_sec": 1440}))

        self.assertEqual(state.focus.elapsed_sec, 60)
        self.assertEqual(state.focus.remaining_sec, 1440)

    def test_timer_finished_completes_focus(self) -> None:
        """timer finished 会结束专注并写入 timer_complete session。"""
        state = AgentState()
        reduce_state(state, Event(type="focus_start_requested", timestamp=0, payload={"duration_sec": 60}))

        reduce_state(state, Event(type="timer_finished", timestamp=60, payload={}))

        self.assertFalse(state.focus.active)
        self.assertEqual(state.focus.last_focus_end_ts, 60)
        self.assertEqual(state.runtime_history.focus_sessions[-1]["reason"], "timer_complete")

    def test_system_timer_events_restore_and_stop_focus(self) -> None:
        state = AgentState()

        reduce_state(
            state,
            Event(
                type="system_triggered",
                timestamp=100,
                payload={
                    "trigger": "focus_timer_started",
                    "source": "agent_action_result",
                    "source_event_type": "user_text_input",
                    "duration_sec": 1200,
                },
            ),
        )
        self.assertTrue(state.focus.active)
        self.assertEqual(state.focus.target_duration_sec, 1200)
        self.assertEqual(state.focus.remaining_sec, 1200)

        reduce_state(
            state,
            Event(
                type="system_triggered",
                timestamp=220,
                payload={"trigger": "focus_timer_stopped", "source": "agent_action_result"},
            ),
        )
        self.assertFalse(state.focus.active)
        self.assertEqual(state.runtime_history.focus_sessions[-1]["reason"], "timer_stopped")

    def test_timer_tick_past_target_completes_focus(self) -> None:
        state = AgentState()
        reduce_state(state, Event(type="focus_start_requested", timestamp=10, payload={"duration_sec": 60}))

        reduce_state(state, Event(type="timer_ticked", timestamp=75, payload={"remaining_sec": 0}))

        self.assertFalse(state.focus.active)
        self.assertEqual(state.focus.last_focus_end_ts, 75)
        self.assertEqual(state.runtime_history.focus_sessions[-1]["reason"], "timer_complete")

    def test_user_state_events_update_fields_and_confidence(self) -> None:
        """用户感知事件更新对应用户状态块。"""
        state = AgentState()

        reduce_state(state, Event(type="user_presence_updated", timestamp=1, payload={"presence": "present", "confidence": "0.8"}))
        self.assertEqual(state.user.presence, "present")
        self.assertEqual(state.user.presence_confidence, 0.8)

        reduce_state(
            state,
            Event(
                type="user_attention_updated",
                timestamp=2,
                payload={"attention": "focused", "behavior": "working", "confidence": "0.7"},
            ),
        )
        self.assertEqual(state.user.attention, "focused")
        self.assertEqual(state.user.behavior, "working")
        self.assertEqual(state.user.attention_confidence, 0.7)
        self.assertEqual(state.user.behavior_confidence, 0.7)

        reduce_state(state, Event(type="user_emotion_updated", timestamp=3, payload={"emotion": "happy", "confidence": "0.9"}))
        self.assertEqual(state.user.emotion, "happy")
        self.assertEqual(state.user.emotion_confidence, 0.9)

        reduce_state(
            state,
            Event(type="user_fatigue_updated", timestamp=4, payload={"fatigue_level": "high", "confidence": "0.6"}),
        )
        self.assertEqual(state.user.fatigue_level, "high")
        self.assertEqual(state.user.fatigue_confidence, 0.6)

    def test_environment_events_update_environment_state(self) -> None:
        """环境事件更新环境状态块，不触发任何决策。"""
        state = AgentState()

        reduce_state(state, Event(type="light_level_updated", timestamp=1, payload={"light_lux": "120", "level": "low"}))
        self.assertEqual(state.environment.light_lux, 120)
        self.assertEqual(state.environment.light_level, "low")

        reduce_state(
            state,
            Event(
                type="temperature_humidity_updated",
                timestamp=2,
                payload={
                    "temperature_c": "28.5",
                    "humidity_pct": "55.2",
                    "temperature_level": "high",
                    "humidity_level": "normal",
                },
            ),
        )
        self.assertEqual(state.environment.temperature_c, 28.5)
        self.assertEqual(state.environment.humidity_pct, 55.2)
        self.assertEqual(state.environment.temperature_level, "high")
        self.assertEqual(state.environment.humidity_level, "normal")

        reduce_state(state, Event(type="noise_level_updated", timestamp=3, payload={"noise_db": "67", "level": "noisy"}))
        self.assertEqual(state.environment.noise_db, 67)
        self.assertEqual(state.environment.noise_level, "noisy")

    def test_voice_and_tts_events_update_dialogue_state(self) -> None:
        """语音输入和 TTS 事件只更新 dialogue_state。"""
        state = AgentState()

        reduce_state(state, Event(type="voice_input_started", timestamp=1, payload={}))
        self.assertEqual(state.interaction.dialogue_state, "listening")
        reduce_state(state, Event(type="voice_input_stopped", timestamp=2, payload={}))
        self.assertEqual(state.interaction.dialogue_state, "thinking")
        reduce_state(state, Event(type="tts_started", timestamp=3, payload={}))
        self.assertEqual(state.interaction.dialogue_state, "speaking")
        reduce_state(state, Event(type="tts_finished", timestamp=4, payload={}))
        self.assertEqual(state.interaction.dialogue_state, "idle")

    def test_unknown_event_is_safely_ignored(self) -> None:
        """未注册事件不抛异常，并返回原 state 对象。"""
        state = AgentState()
        before = state.to_dict()

        returned = reduce_state(state, Event(type="unknown_event", timestamp=1, payload={"x": 1}))  # type: ignore[arg-type]

        self.assertIs(returned, state)
        self.assertEqual(state.to_dict(), before)

    def test_invalid_numeric_payload_falls_back_safely(self) -> None:
        """非法数值 payload 不应让 reducer 崩溃。"""
        state = AgentState()

        reduce_state(state, Event(type="light_level_updated", timestamp=1, payload={"light_lux": "bad"}))
        reduce_state(state, Event(type="user_presence_updated", timestamp=2, payload={"confidence": "bad"}))
        reduce_state(state, Event(type="focus_start_requested", timestamp=3, payload={"duration_sec": "bad"}))

        self.assertIsNone(state.environment.light_lux)
        self.assertIsNone(state.user.presence_confidence)
        self.assertEqual(state.focus.target_duration_sec, 0)

    def test_reducer_uses_registry_and_does_not_import_decision_or_action(self) -> None:
        """reducer 保持状态层边界，不依赖 Intent/Action。"""
        source = inspect.getsource(reducer)

        self.assertIn("REDUCERS", source)
        self.assertIn("focus_start_requested", REDUCERS)
        self.assertIn("system_triggered", REDUCERS)
        self.assertIn("user_posture_updated", REDUCERS)
        self.assertIn("user_activity_updated", REDUCERS)
        self.assertNotIn("src.agent.action", source)
        self.assertNotIn("src.agent.decision.intent", source)


if __name__ == "__main__":
    unittest.main()
