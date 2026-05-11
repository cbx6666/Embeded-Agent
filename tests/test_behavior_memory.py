from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.agent.action import display, speak
from src.agent.decision.planner import plan_intents
from src.agent.event import Event
from src.agent.memory.behavior.behavior_stats import BehaviorStats
from src.agent.memory.behavior.behavior_updater import BehaviorUpdater
from src.agent.memory.extractor.behavior_extractor import BehaviorSignal
from src.agent.memory.extractor.insight_extractor import InsightExtractor
from src.agent.memory.memory_candidate import MemoryCandidate
from src.agent.memory.memory_pipeline import MemoryPipeline
from src.agent.memory.policy.memory_policy import MemoryPolicy
from src.agent.memory.policy.personalization_policy import PersonalizationPolicy
from src.agent.state import AgentState
from src.agent.state.user_profile_state import UserProfileInsight
from src.services.user_profile_service import UserProfileService
from src.storage.behavior_store import BehaviorStore
from src.storage.profile_store import ProfileStore


class BehaviorMemoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_behavior_updater_tracks_long_term_stats(self) -> None:
        stats = BehaviorStats()
        updater = BehaviorUpdater()

        updater.update(stats, BehaviorSignal(type="focus_started", timestamp=1, payload={"hour": 22}))
        updater.update(stats, BehaviorSignal(type="focus_completed", timestamp=2, payload={"duration_sec": 1200}))
        updater.update(stats, BehaviorSignal(type="fatigue_detected", timestamp=3, payload={"focus_elapsed_sec": 600}))
        updater.update(stats, BehaviorSignal(type="distraction_detected", timestamp=4))
        updater.update(stats, BehaviorSignal(type="break_suggestion_shown", timestamp=5, payload={"content_type": "音乐"}))
        updater.update(stats, BehaviorSignal(type="break_suggestion_accepted", timestamp=6, payload={"content_type": "音乐"}))

        self.assertEqual(stats.focus_start_by_hour["22"], 1)
        self.assertEqual(stats.average_focus_duration_sec, 1200)
        self.assertEqual(stats.fatigue_event_count, 1)
        self.assertEqual(stats.distraction_event_count, 1)
        self.assertEqual(stats.music_break_suggestion_count, 1)
        self.assertEqual(stats.music_break_accept_rate, 1.0)

    def test_insight_extractor_creates_candidates_from_stats(self) -> None:
        stats = BehaviorStats(
            focus_start_by_hour={"21": 2, "22": 1},
            accepted_music_breaks=4,
            rejected_music_breaks=0,
            fatigue_after_focus_count=3,
            fatigue_after_focus_duration_total_sec=1800,
        )

        candidates = InsightExtractor().extract(stats)
        contents = {candidate.content for candidate in candidates}

        self.assertIn("用户倾向于夜间学习", contents)
        self.assertIn("用户疲劳时更容易接受音乐休息", contents)
        self.assertIn("用户较短专注后也容易疲劳", contents)

    def test_memory_policy_blocks_low_evidence_and_decays_old_insights(self) -> None:
        policy = MemoryPolicy(evidence_threshold=3, confidence_threshold=0.6, decay_after_days=1)
        candidate = MemoryCandidate(
            insight_type="study_time_pattern",
            content="用户倾向于夜间学习",
            confidence=0.9,
            evidence_count=1,
            source="test",
            explanation="测试低证据",
        )

        decision = policy.evaluate(candidate, existing_insights=[])
        old_insight = UserProfileInsight(
            insight_type="study_time_pattern",
            content="用户倾向于夜间学习",
            confidence=0.8,
            evidence_count=5,
            updated_at=0,
        )
        decayed, changed = policy.decay_insights([old_insight], now=3 * 24 * 3600)

        self.assertFalse(decision.allow_write)
        self.assertTrue(changed)
        self.assertLess(decayed[0].confidence, old_insight.confidence)

    def test_memory_pipeline_updates_stats_and_profile_insights(self) -> None:
        service = UserProfileService(ProfileStore(self.root / "profiles.json"))
        pipeline = MemoryPipeline(BehaviorStore(self.root / "behavior.json"), service)
        state = AgentState(current_user_id="xiaoli")

        for index in range(3):
            timestamp = _timestamp_at_hour(22, day=index + 1)
            state.focus.active = True
            state.focus.start_ts = timestamp
            pipeline.process_event(
                "xiaoli",
                Event(
                    type="focus_start_requested",
                    timestamp=timestamp,
                    payload={"duration_sec": 1500, "source": "test"},
                ),
                state,
            )

        stats = pipeline.get_stats("xiaoli")
        insights = service.get_user("xiaoli").insights

        self.assertEqual(stats.focus_start_by_hour["22"], 3)
        self.assertTrue(any(item.content == "用户倾向于夜间学习" for item in insights))
        self.assertTrue((self.root / "behavior.json").exists())

    def test_memory_pipeline_counts_one_break_suggestion_per_action_batch(self) -> None:
        service = UserProfileService(ProfileStore(self.root / "profiles.json"))
        pipeline = MemoryPipeline(BehaviorStore(self.root / "behavior.json"), service)

        pipeline.process_actions(
            "xiaoli",
            [
                speak("休息一下，我可以陪你听一小段音乐。", kind="notification", reason="rest_reminder"),
                display("休息一下，我可以陪你听一小段音乐。", kind="notification", reason="rest_reminder"),
            ],
            timestamp=100,
        )

        stats = pipeline.get_stats("xiaoli")
        self.assertEqual(stats.break_suggestion_count, 1)
        self.assertEqual(stats.music_break_suggestion_count, 1)

    def test_personalized_policy_can_change_planner_threshold(self) -> None:
        service = UserProfileService(ProfileStore(self.root / "profiles.json"))
        service.switch_user("xiaoli", display_name="小李", timestamp=1000)
        service.upsert_insight(
            "xiaoli",
            insight_type="study_time_pattern",
            content="用户倾向于夜间学习",
            confidence=0.85,
            evidence_count=4,
            timestamp=1000,
        )
        personalized_policy = PersonalizationPolicy(service).build("xiaoli")
        state = AgentState(current_user_id="xiaoli")
        state.focus.active = True
        state.focus.start_ts = 0
        state.focus.elapsed_sec = 400
        state.user.presence = "present"
        state.user.fatigue_level = "high"

        intents = plan_intents(
            state,
            state,
            Event(type="system_triggered", timestamp=1001, payload={"trigger": "focus_health_check"}),
            personalized_policy=personalized_policy,
        )

        self.assertTrue(personalized_policy.reduce_night_rest_pressure)
        self.assertFalse(any(intent.type == "suggest_rest" for intent in intents))


def _timestamp_at_hour(hour: int, *, day: int = 1) -> int:
    return int(datetime(2026, 1, day, hour, 0).timestamp())


if __name__ == "__main__":
    unittest.main()
