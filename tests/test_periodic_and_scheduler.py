from __future__ import annotations

"""自检触发算法、调度防打断、sensor_status_report 的行为测试。"""

import tempfile
import unittest
from pathlib import Path

from src.agent.core import build_default_core
from src.agent.core.models import Event
from src.agent.decision.sensor_status_handler import SensorStatusHandler
from src.agent.event.router import EventRouter
from src.agent.policy_config import (
    BEHAVIOR_DISTRACTION_PRIORITY,
    ENVIRONMENT_CARE_PRIORITY,
    SENSOR_REPORT_PRIORITY,
    SPEECH_PRIORITY,
    WELLNESS_CARE_PRIORITY,
    ScheduledTaskPolicy,
    SchedulePolicy,
)
from src.agent.scheduler.autonomous_scheduler import AutonomousScheduler, ScheduledTask
from src.agent.state.agent_state import AgentState
from src.agent.state.summary_builder import (
    build_behavior_distraction_summary,
    build_sensor_status_summary,
)
from tests.fakes.fake_llm_service import FakeLLMService


def _state(
    *,
    fatigue="none",
    fatigue_conf=0.0,
    emotion="neutral",
    emotion_conf=0.0,
    attention="idle",
    attention_conf=0.0,
    posture="unknown",
    presence="present",
    temperature_c=None,
    humidity_pct=None,
    noise_db=None,
    light_lux=None,
    dialogue_state="idle",
    last_agent_response_time=None,
) -> AgentState:
    state = AgentState()
    state.user.presence = presence
    state.user.fatigue_level = fatigue
    state.user.fatigue_confidence = fatigue_conf
    state.user.emotion = emotion
    state.user.emotion_confidence = emotion_conf
    state.user.attention = attention
    state.user.attention_confidence = attention_conf
    state.user.posture = posture
    state.environment.temperature_c = temperature_c
    state.environment.humidity_pct = humidity_pct
    state.environment.noise_db = noise_db
    state.environment.light_lux = light_lux
    state.interaction.dialogue_state = dialogue_state
    state.interaction.last_agent_response_time = last_agent_response_time
    return state


class SensorSummaryTest(unittest.TestCase):
    def test_temperature_abnormal_item(self) -> None:
        summary = build_sensor_status_summary(_state(temperature_c=31))
        types = {item["type"] for item in summary["abnormal_items"]}
        self.assertIn("temperature", types)

    def test_noise_abnormal_item(self) -> None:
        summary = build_sensor_status_summary(_state(noise_db=72))
        types = {item["type"] for item in summary["abnormal_items"]}
        self.assertIn("noise", types)

    def test_light_abnormal_item(self) -> None:
        summary = build_sensor_status_summary(_state(light_lux=120))
        types = {item["type"] for item in summary["abnormal_items"]}
        self.assertIn("light", types)

    def test_sensor_summary_uses_concrete_values(self) -> None:
        summary = build_sensor_status_summary(
            _state(temperature_c=31.2, humidity_pct=68, noise_db=72, light_lux=120)
        )
        self.assertEqual(summary["temperature_c"], 31.2)
        self.assertEqual(summary["noise_db"], 72)
        self.assertEqual(summary["light_lux"], 120)
        env_types = {item["type"] for item in summary["abnormal_items"]}
        self.assertIn("temperature", env_types)
        self.assertIn("noise", env_types)


class RouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = EventRouter()

    def test_sensor_status_report_routes_to_sensor(self) -> None:
        decision = self.router.classify(
            Event(
                type="system_triggered",
                timestamp=1,
                payload={"trigger": "sensor_status_report", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(decision.kind, "sensor_status")
        self.assertFalse(decision.uses_llm)

    def test_behavior_distraction_routes_to_behavior_distraction(self) -> None:
        decision = self.router.classify(
            Event(
                type="system_triggered",
                timestamp=1,
                payload={"trigger": "behavior_distraction_check", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(decision.kind, "behavior_distraction")
        self.assertTrue(decision.uses_llm)

    def test_removed_periodic_trigger_falls_back_to_state_only(self) -> None:
        # periodic_state_check 已废弃删除：不再被识别为 LLM 入口。
        decision = self.router.classify(
            Event(
                type="system_triggered",
                timestamp=1,
                payload={"trigger": "periodic_state_check", "source": "agent_autonomy"},
            )
        )
        self.assertNotEqual(decision.kind, "periodic_state")
        self.assertFalse(decision.uses_llm)


class SchedulerTest(unittest.TestCase):
    def _scheduler(self, *, tasks=None, busy_ref=None, clock=None):
        clock = clock if clock is not None else {"t": 1000}
        busy_ref = busy_ref if busy_ref is not None else {"p": None}
        emitted: list[Event] = []
        policy = SchedulePolicy(tasks=tuple(tasks)) if tasks else SchedulePolicy()
        scheduler = AutonomousScheduler(
            state_provider=AgentState,
            event_sink=emitted.append,
            config=policy,
            time_fn=lambda: clock["t"],
            busy_priority_provider=lambda: busy_ref["p"],
        )
        return scheduler, emitted, clock, busy_ref

    def _task(self, scheduler: AutonomousScheduler, name: str) -> ScheduledTask:
        return next(t for t in scheduler.tasks if t.name == name)

    def test_priority_constants_ordering(self) -> None:
        self.assertLess(SPEECH_PRIORITY, BEHAVIOR_DISTRACTION_PRIORITY)
        self.assertLess(BEHAVIOR_DISTRACTION_PRIORITY, WELLNESS_CARE_PRIORITY)
        self.assertLess(WELLNESS_CARE_PRIORITY, ENVIRONMENT_CARE_PRIORITY)
        self.assertLess(ENVIRONMENT_CARE_PRIORITY, SENSOR_REPORT_PRIORITY)

    def test_tasks_have_remaining_and_due_state(self) -> None:
        scheduler, _emitted, _clock, _busy = self._scheduler()
        behavior = self._task(scheduler, "behavior_distraction_check")
        self.assertEqual(behavior.interval_sec, 20)
        self.assertEqual(behavior.priority, BEHAVIOR_DISTRACTION_PRIORITY)
        wellness = self._task(scheduler, "wellness_care_check")
        self.assertEqual(wellness.interval_sec, 30)
        self.assertEqual(wellness.priority, WELLNESS_CARE_PRIORITY)
        self.assertTrue(hasattr(wellness, "remaining_sec"))
        self.assertTrue(hasattr(wellness, "due"))
        environment = self._task(scheduler, "environment_care_check")
        self.assertEqual(environment.interval_sec, 60)
        self.assertEqual(environment.priority, ENVIRONMENT_CARE_PRIORITY)
        sensor = self._task(scheduler, "sensor_status_report")
        self.assertEqual(sensor.interval_sec, 300)
        self.assertEqual(sensor.priority, SENSOR_REPORT_PRIORITY)

    def test_behavior_distraction_emits_after_20s(self) -> None:
        scheduler, emitted, clock, _busy = self._scheduler()
        self.assertEqual(scheduler.run_due(), [])  # baseline
        clock["t"] += 19
        self.assertEqual(scheduler.run_due(), [])
        clock["t"] += 1
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].payload["trigger"], "behavior_distraction_check")

    def test_wellness_emits_after_30s(self) -> None:
        tasks = [ScheduledTaskPolicy("wellness_care_check", "wellness_care_check", 30, WELLNESS_CARE_PRIORITY)]
        scheduler, emitted, clock, _busy = self._scheduler(tasks=tasks)
        self.assertEqual(scheduler.run_due(), [])  # baseline
        clock["t"] += 29
        self.assertEqual(scheduler.run_due(), [])
        clock["t"] += 1
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].payload["trigger"], "wellness_care_check")

    def test_sensor_emits_every_300s(self) -> None:
        tasks = [ScheduledTaskPolicy("sensor_status_report", "sensor_status_report", 300, SENSOR_REPORT_PRIORITY)]
        scheduler, _emitted, clock, _busy = self._scheduler(tasks=tasks)
        self.assertEqual(scheduler.run_due(), [])  # baseline
        clock["t"] += 299
        self.assertEqual(scheduler.run_due(), [])
        clock["t"] += 1
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].payload["trigger"], "sensor_status_report")

    def test_priority_order_one_emit_per_tick(self) -> None:
        scheduler, _emitted, clock, _busy = self._scheduler()
        scheduler.run_due()  # baseline
        clock["t"] += 300  # behavior(20), wellness(30), environment(60), sensor(300) all due
        # 同一时刻只发一个，且严格按优先级：分心 > 疲劳/情绪关怀 > 环境关怀 > 环境详细播报。
        first = scheduler.run_due()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].payload["trigger"], "behavior_distraction_check")
        second = scheduler.run_due()
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].payload["trigger"], "wellness_care_check")
        third = scheduler.run_due()
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0].payload["trigger"], "environment_care_check")
        fourth = scheduler.run_due()
        self.assertEqual(len(fourth), 1)
        self.assertEqual(fourth[0].payload["trigger"], "sensor_status_report")

    def test_revert_emission_restores_due_for_immediate_retry(self) -> None:
        tasks = [ScheduledTaskPolicy("sensor_status_report", "sensor_status_report", 300, SENSOR_REPORT_PRIORITY)]
        scheduler, _emitted, clock, _busy = self._scheduler(tasks=tasks)
        scheduler.run_due()  # baseline
        clock["t"] += 300
        scheduler.run_due()  # emit + reset to 300
        sensor = self._task(scheduler, "sensor_status_report")
        self.assertFalse(sensor.due)
        self.assertAlmostEqual(sensor.remaining_sec, 300.0, places=3)

        self.assertTrue(scheduler.revert_emission("sensor_status_report"))
        self.assertTrue(sensor.due)
        self.assertAlmostEqual(sensor.remaining_sec, 0.0, places=3)

        clock["t"] += 1
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].payload["trigger"], "sensor_status_report")

    def test_revert_emission_honors_retry_after_sec(self) -> None:
        tasks = [ScheduledTaskPolicy("sensor_status_report", "sensor_status_report", 300, SENSOR_REPORT_PRIORITY)]
        scheduler, _emitted, clock, _busy = self._scheduler(tasks=tasks)
        scheduler.run_due()
        clock["t"] += 300
        scheduler.run_due()
        sensor = self._task(scheduler, "sensor_status_report")

        self.assertTrue(scheduler.revert_emission("sensor_status_report", retry_after_sec=30.0))
        self.assertFalse(sensor.due)
        self.assertAlmostEqual(sensor.remaining_sec, 30.0, places=3)

        clock["t"] += 29
        self.assertEqual(scheduler.run_due(), [])
        clock["t"] += 1
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)

    def test_speech_preempts_behavior_then_resumes(self) -> None:
        # 用显式 30s 周期聚焦「抢占冻结、不重置」逻辑。
        tasks = [
            ScheduledTaskPolicy(
                "behavior_distraction_check", "behavior_distraction_check", 30, BEHAVIOR_DISTRACTION_PRIORITY
            ),
            ScheduledTaskPolicy("wellness_care_check", "wellness_care_check", 30, WELLNESS_CARE_PRIORITY),
            ScheduledTaskPolicy("sensor_status_report", "sensor_status_report", 300, SENSOR_REPORT_PRIORITY),
        ]
        scheduler, _emitted, clock, busy = self._scheduler(tasks=tasks)
        scheduler.run_due()  # baseline at 1000
        clock["t"] += 25
        scheduler.run_due()
        behavior = self._task(scheduler, "behavior_distraction_check")
        self.assertAlmostEqual(behavior.remaining_sec, 5.0, places=3)

        # 语音抢占（优先级 0），倒计时冻结。
        busy["p"] = SPEECH_PRIORITY
        clock["t"] += 3
        scheduler.run_due()
        clock["t"] += 12
        scheduler.run_due()
        self.assertAlmostEqual(behavior.remaining_sec, 5.0, places=3)

        # 语音结束后从剩 5 秒继续，而不是重等 30 秒。
        busy["p"] = None
        clock["t"] += 3
        scheduler.run_due()
        self.assertAlmostEqual(behavior.remaining_sec, 2.0, places=3)
        clock["t"] += 2
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].payload["trigger"], "behavior_distraction_check")

    def test_sensor_preempted_not_reset(self) -> None:
        scheduler, _emitted, clock, busy = self._scheduler()
        scheduler.run_due()  # baseline
        sensor = self._task(scheduler, "sensor_status_report")
        sensor.remaining_sec = 30.0  # 直接设到剩 30 秒，聚焦抢占行为

        # behavior（优先级 1）运行时，sensor（3）冻结。
        busy["p"] = BEHAVIOR_DISTRACTION_PRIORITY
        clock["t"] += 10
        scheduler.run_due()
        clock["t"] += 15
        scheduler.run_due()
        self.assertAlmostEqual(sensor.remaining_sec, 30.0, places=3)

        # 解除后从剩 30 秒继续。
        busy["p"] = None
        clock["t"] += 5
        scheduler.run_due()
        self.assertAlmostEqual(sensor.remaining_sec, 25.0, places=3)

    def test_due_task_deferred_while_speech_busy(self) -> None:
        # 任务已到点（due）但语音正在处理时，进入 pending_due，不发出；语音结束后立即执行。
        scheduler, _emitted, clock, busy = self._scheduler()
        scheduler.run_due()  # baseline
        behavior = self._task(scheduler, "behavior_distraction_check")
        behavior.due = True
        behavior.remaining_sec = 0.0

        busy["p"] = SPEECH_PRIORITY
        self.assertEqual(scheduler.run_due(), [])  # 被语音延后
        self.assertTrue(behavior.due)  # 仍保持 due，不重置周期

        busy["p"] = None
        produced = scheduler.run_due()
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].payload["trigger"], "behavior_distraction_check")


class BehaviorDistractionSummaryTest(unittest.TestCase):
    def _state_with_phone_use(self, *, yolo: bool = True, ts: int = 1000) -> AgentState:
        state = AgentState()
        state.user.presence = "present"
        state.user.attention = "distracted"
        state.user.behavior = "phone_use"
        state.user.behavior_confidence = 0.85
        state.user.attention_confidence = 0.85
        state.runtime_history.attention_records = [
            {
                "timestamp": ts - 10,
                "attention": "distracted",
                "behavior": "phone_use",
                "confidence": 0.85,
                "yolo_phone_detected": yolo,
            },
            {
                "timestamp": ts - 5,
                "attention": "distracted",
                "behavior": "phone_use",
                "confidence": 0.85,
                "yolo_phone_detected": yolo,
            },
        ]
        return state

    def test_summary_triggers_only_when_yolo_detects_phone(self) -> None:
        # YOLO 真正检出手机才触发。
        ok = build_behavior_distraction_summary(self._state_with_phone_use(yolo=True), check_time=1000)
        self.assertTrue(ok["trigger_candidate"])
        # 硬性要求：行为分类为 phone_use 但 YOLO 没框到手机时**不**触发，避免误报。
        face_only = build_behavior_distraction_summary(self._state_with_phone_use(yolo=False), check_time=1000)
        self.assertFalse(face_only["trigger_candidate"])


class _CoreTestBase(unittest.TestCase):
    def make_core(self, fake: FakeLLMService):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
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


class SensorReportFlowTest(_CoreTestBase):
    def test_sensor_report_speaks_concrete_status(self) -> None:
        fake = FakeLLMService()
        core = self.make_core(fake)
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.handle_event(
            Event(
                type="temperature_humidity_updated",
                timestamp=2,
                payload={"temperature_c": 31.2, "humidity_pct": 68, "temperature_level": "high"},
            )
        )
        actions, _ = core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1000,
                payload={"trigger": "sensor_status_report", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(core.last_decision_result.source, "sensor_status_report")
        self.assertEqual({a.type for a in actions}, {"speak", "display"})
        # 环境播报为确定性：不调用 LLM，但口播文本必须包含准确数值。
        self.assertNotIn("sensor_status_report", fake.calls)
        spoken = core.last_decision_result.reply_text
        self.assertIn("31.2", spoken)
        self.assertIn("68", spoken)

    def test_deferred_sensor_reverts_scheduler_cycle(self) -> None:
        fake = FakeLLMService()
        core = self.make_core(fake)
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "present"})
        )
        core.handle_event(
            Event(
                type="temperature_humidity_updated",
                timestamp=2,
                payload={"temperature_c": 25.0, "humidity_pct": 50},
            )
        )
        core.state.interaction.dialogue_state = "listening"

        sensor = next(
            t for t in core.autonomous_scheduler.tasks if t.trigger == "sensor_status_report"
        )
        sensor.remaining_sec = 0.0
        sensor.due = True
        sensor.remaining_sec = 300.0
        sensor.due = False

        actions, _ = core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1000,
                payload={"trigger": "sensor_status_report", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(actions, [])
        self.assertTrue(sensor.due)
        self.assertAlmostEqual(sensor.remaining_sec, 0.0, places=3)

    def test_sensor_report_no_op_when_away(self) -> None:
        fake = FakeLLMService()
        fake.set_response(
            "sensor_status_report", {"intent": "report_status", "reply": "环境正常。"}
        )
        core = self.make_core(fake)
        core.handle_event(
            Event(type="user_presence_updated", timestamp=1, payload={"presence": "away"})
        )
        actions, _ = core.handle_event(
            Event(
                type="system_triggered",
                timestamp=1000,
                payload={"trigger": "sensor_status_report", "source": "agent_autonomy"},
            )
        )
        self.assertEqual(actions, [])
        # 用户不在场时不应调用 LLM。
        self.assertNotIn("sensor_status_report", fake.calls)


class SensorHandlerUnitTest(unittest.TestCase):
    def _client(self, payload):
        class _C:
            def __init__(self, p):
                self.p = p

            def complete_json(self, role, prompt):
                return self.p

        return _C(payload)

    def test_speaking_produces_speak_for_tts_queue(self) -> None:
        handler = SensorStatusHandler()
        state = _state(dialogue_state="speaking", temperature_c=25.0, humidity_pct=50)
        result = handler.decide(
            state=state,
            event=Event(type="system_triggered", timestamp=1000, payload={}),
            llm_client=self._client({"intent": "report_status", "reply": "x"}),
        )
        self.assertEqual(result.intents[0].type, "report_sensor_status")
        self.assertTrue(any(action.type == "speak" for action in result.actions))

    def test_recent_speak_does_not_block_report(self) -> None:
        handler = SensorStatusHandler()
        state = _state(last_agent_response_time=970, temperature_c=25.0, humidity_pct=50)
        result = handler.decide(
            state=state,
            event=Event(type="system_triggered", timestamp=1000, payload={}),
            llm_client=self._client({"intent": "report_status", "reply": "x"}),
        )
        self.assertEqual(result.intents[0].type, "report_sensor_status")
        self.assertTrue(any(action.type == "speak" for action in result.actions))


class ActivityPriorityTest(_CoreTestBase):
    def test_state_only_events_do_not_set_busy_priority(self) -> None:
        fake = FakeLLMService()
        core = self.make_core(fake)
        core.handle_event(
            Event(
                type="user_fatigue_updated",
                timestamp=1,
                payload={"fatigue_level": "high", "confidence": 0.9},
            )
        )
        # 高频 state_only 事件处理完后不占用 LLM 槽，调度器不会被冻结。
        self.assertIsNone(core.current_activity_priority())


if __name__ == "__main__":
    unittest.main()
