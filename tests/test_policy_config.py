from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent.config.policy_config import (
    ActionPolicyConfig,
    ContextPolicyConfig,
    CopyPolicyConfig,
    GuardPolicyConfig,
    RuntimeHistoryPolicyConfig,
)
from src.agent.config.policy_config import RetrievalPolicyConfig
from src.agent.user.personal_context import DEFAULT_RETRIEVAL_POLICY, PersonalContext
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.agent_context_builder import AgentContextBuilder
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.event import Event
from src.agent.llm_agent.schemas import ResponseDraft
from src.agent.memory.memory_candidate import MemoryCandidate
from src.agent.reducer import REDUCERS
from src.agent.state import AgentState
from src.services.runtime_history_service import RuntimeHistoryService
from src.storage.long_term_memory_store import LongTermMemoryStore


class PolicyConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_policy_config_keeps_existing_action_defaults(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="focus_start_requested", timestamp=1, payload={}),
            personal_context=None,
        )

        actions = ActionRealizer().realize(
            IntentPlan(intents=[AgentIntent(type="start_focus")]),
            response=ResponseDraft(),
            context=context,
        )

        self.assertEqual(actions[0].payload["duration_sec"], 1500)
        self.assertEqual(actions[1].payload["text"], "已开始专注 25 分钟。")

    def test_custom_cooldown_and_invalid_timestamp_policy_take_effect(self) -> None:
        state = AgentState()
        state.cooldown.reminder_last_ts["rest_reminder"] = 100
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="system_triggered", timestamp=130, payload={}),
            personal_context=None,
        )

        blocked = DeterministicGuard(
            policy_config=GuardPolicyConfig(reminder_cooldown_sec=60)
        ).filter(IntentPlan(intents=[AgentIntent(type="suggest_rest")]), context)
        allowed = DeterministicGuard(
            policy_config=GuardPolicyConfig(reminder_cooldown_sec=10)
        ).filter(IntentPlan(intents=[AgentIntent(type="suggest_rest")]), context)

        self.assertEqual(blocked.plan.intents[0].type, "no_op")
        self.assertEqual(allowed.plan.intents[0].type, "suggest_rest")

        state.cooldown.reminder_last_ts["rest_reminder"] = "bad"
        invalid_context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="system_triggered", timestamp=130, payload={}),
            personal_context=None,
        )
        strict = DeterministicGuard(
            policy_config=GuardPolicyConfig(allow_on_invalid_cooldown_timestamp=False)
        ).filter(IntentPlan(intents=[AgentIntent(type="suggest_rest")]), invalid_context)

        self.assertEqual(strict.plan.intents[0].type, "no_op")

    def test_custom_user_initiated_event_types_take_effect(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="system_triggered", timestamp=1, payload={}),
            personal_context=None,
        )

        default_blocked = DeterministicGuard().filter(
            IntentPlan(intents=[AgentIntent(type="answer_user", requires_llm=True)]),
            context,
        )
        custom_allowed = DeterministicGuard(
            policy_config=GuardPolicyConfig(user_initiated_event_types=frozenset({"system_triggered"}))
        ).filter(
            IntentPlan(intents=[AgentIntent(type="answer_user", requires_llm=True)]),
            context,
        )

        self.assertEqual(default_blocked.plan.intents[0].type, "no_op")
        self.assertEqual(custom_allowed.plan.intents[0].type, "answer_user")

    def test_custom_context_trimming_and_memory_bucket_limit_take_effect(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        for index in range(3):
            store.upsert_candidate(
                "u1",
                MemoryCandidate(
                    memory_type="behavior_preference",
                    content=f"Preference {index}",
                    confidence=0.9,
                    evidence=[{"event": "user_text_input", "timestamp": index}],
                ),
                timestamp=index,
            )
        state = AgentState(current_user_id="u1")
        state.runtime_history.recent_messages = [{"text": str(i)} for i in range(3)]
        state.runtime_history.recent_events = [{"type": str(i)} for i in range(4)]
        state.runtime_history.recent_actions = [{"type": str(i)} for i in range(3)]

        personal_context = PersonalContextBuilder(
            long_term_memory_store=store,
            policy_config=ContextPolicyConfig(
                max_recent_messages=1,
                max_recent_events=2,
                max_recent_actions=1,
                max_memory_items_per_bucket=1,
            ),
        ).build(user_id="u1", state=state)

        self.assertEqual(len(personal_context.runtime_history["recent_messages"]), 1)
        self.assertEqual(len(personal_context.runtime_history["recent_events"]), 2)
        self.assertEqual(len(personal_context.runtime_history["recent_actions"]), 1)
        self.assertEqual(len(personal_context.behavior_preferences), 1)

    def test_retrieval_policy_config_centralizes_ranking_weights(self) -> None:
        policy = RetrievalPolicyConfig()

        self.assertEqual(policy.source_weights["UserProfile"], 100.0)
        self.assertEqual(policy.source_weights["LongTermMemory"], 50.0)
        self.assertEqual(policy.source_weights["RuntimeHistory"], 25.0)
        self.assertEqual(policy.event_type_weights["user_text_input"]["behavior_preference"], 10.0)
        self.assertEqual(policy.event_type_weights["user_fatigue_updated"]["behavior_pattern"], 12.0)
        self.assertEqual(policy.conflict_penalty, 30.0)
        self.assertIsInstance(DEFAULT_RETRIEVAL_POLICY, RetrievalPolicyConfig)

    def test_retrieval_ranking_behavior_is_preserved_by_default_policy(self) -> None:
        context = PersonalContext(
            user_id="u1",
            profile_items=(
                {
                    "item_type": "explicit_user_preference",
                    "content": "reminder_style: gentle",
                    "source": "UserProfile",
                    "priority_score": 40.0,
                    "tags": ["reminder_style", "explicit_user_preference"],
                },
            ),
            behavior_preferences=(
                {
                    "memory_type": "behavior_preference",
                    "content": "User prefers gentle reminders.",
                    "source": "LongTermMemory",
                    "effective_confidence": 0.9,
                    "evidence_count": 3,
                    "priority_score": 10.0,
                    "tags": ["behavior_preference", "reminder_style"],
                },
            ),
            runtime_items=(
                {
                    "item_type": "recent_message",
                    "content": "user: gentle please",
                    "source": "RuntimeHistory",
                    "priority_score": 4.0,
                    "tags": ["recent_message", "user"],
                },
            ),
        )

        relevant = context.retrieve_relevant(event_type="user_text_input", text="gentle", limit=3)

        self.assertEqual(relevant[0]["source"], "UserProfile")
        self.assertEqual(relevant[1]["source"], "LongTermMemory")
        self.assertEqual(relevant[2]["source"], "RuntimeHistory")

    def test_personal_context_filters_noisy_internal_runtime_events(self) -> None:
        state = AgentState(current_user_id="u1")
        state.runtime_history.recent_events = [
            {"type": "timer_ticked", "timestamp": 1, "payload": {"remaining_sec": 10}},
            {
                "type": "system_triggered",
                "timestamp": 2,
                "payload": {"trigger": "agent_response_completed", "source": "agent_action_result"},
            },
            {
                "type": "system_triggered",
                "timestamp": 3,
                "payload": {"trigger": "focus_timer_started", "source": "agent_action_result"},
            },
            {"type": "user_attention_updated", "timestamp": 4, "payload": {"attention": "focused"}},
        ]

        personal_context = PersonalContextBuilder(
            long_term_memory_store=LongTermMemoryStore(self.root / "memory.json")
        ).build(user_id="u1", state=state)

        self.assertEqual(
            [event["type"] for event in personal_context.runtime_history["recent_events"]],
            ["user_attention_updated"],
        )

    def test_custom_uncertain_confidence_threshold_take_effect(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="User mildly prefers visual feedback.",
                confidence=0.6,
                evidence=[{"event": "user_text_input"}],
            ),
            timestamp=1,
        )

        personal_context = PersonalContextBuilder(
            long_term_memory_store=store,
            policy_config=ContextPolicyConfig(uncertain_confidence_threshold=0.7),
        ).build(user_id="u1", state=AgentState(current_user_id="u1"))

        self.assertEqual(personal_context.behavior_preferences, ())
        self.assertEqual(personal_context.uncertain_memories[0]["content"], "User mildly prefers visual feedback.")

    def test_custom_fallback_copy_take_effect(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=1, payload={"text": "hello"}),
            personal_context=None,
        )

        actions = ActionRealizer(copy_policy=CopyPolicyConfig(fallback_answer_text="Custom fallback.")).realize(
            IntentPlan(intents=[AgentIntent(type="answer_user", requires_llm=True)]),
            response=ResponseDraft(),
            context=context,
        )

        self.assertEqual(actions[0].payload["text"], "Custom fallback.")

    def test_action_realizer_uses_chinese_copy_policy_fallback(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="system_triggered", timestamp=1, payload={"trigger": "focus_health_check"}),
            personal_context=None,
        )

        actions = ActionRealizer().realize(
            IntentPlan(intents=[AgentIntent(type="suggest_rest")]),
            response=ResponseDraft(),
            context=context,
        )

        self.assertEqual(actions[0].payload["text"], "你已经专注一会儿了，要不要稍微休息一下？")

    def test_custom_duration_and_tts_volume_boundaries_take_effect(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="focus_start_requested", timestamp=1, payload={}),
            personal_context=None,
        )
        realizer = ActionRealizer(
            action_policy=ActionPolicyConfig(
                min_duration_sec=60,
                max_duration_sec=600,
                min_tts_volume=10,
                max_tts_volume=20,
            )
        )

        timer_actions = realizer.realize(
            IntentPlan(intents=[AgentIntent(type="start_focus", payload={"duration_sec": 9999})]),
            response=ResponseDraft(),
            context=context,
        )
        volume_actions = realizer.realize(
            IntentPlan(intents=[AgentIntent(type="set_tts_volume", payload={"volume": 99})]),
            response=ResponseDraft(),
            context=context,
        )

        self.assertEqual(timer_actions[0].payload["duration_sec"], 600)
        self.assertEqual(volume_actions[0].payload["volume"], 20)

    def test_runtime_history_policy_default_and_custom_windows_take_effect(self) -> None:
        defaults = RuntimeHistoryPolicyConfig()
        default_service = RuntimeHistoryService()
        self.assertEqual(default_service.policy_config.max_recent_events, defaults.max_recent_events)

        state = AgentState()
        service = RuntimeHistoryService(policy_config=RuntimeHistoryPolicyConfig(max_recent_events=2))
        for index in range(3):
            service.record_event(state, Event(type="system_triggered", timestamp=index, payload={"i": index}))
        service.trim(state)

        self.assertEqual(len(state.runtime_history.recent_events), 2)
        self.assertEqual(state.runtime_history.recent_events[0]["timestamp"], 1)

    def test_protocol_reducer_mapping_is_not_modified(self) -> None:
        self.assertEqual(
            set(REDUCERS),
            {
                "user_text_input",
                "speech_recognized",
                "focus_start_requested",
                "focus_stop_requested",
                "user_switched",
                "user_presence_updated",
                "user_attention_updated",
                "user_emotion_updated",
                "user_fatigue_updated",
                "light_level_updated",
                "temperature_humidity_updated",
                "noise_level_updated",
                "voice_input_started",
                "voice_input_stopped",
                "tts_started",
                "tts_finished",
                "timer_ticked",
                "timer_finished",
                "system_triggered",
            },
        )

    def test_prompt_hardcoding_is_not_moved_to_policy_config_and_production_mock_is_gone(self) -> None:
        llm_source = Path("src/services/llm_service.py").read_text(encoding="utf-8")
        memory_source = Path("src/agent/memory/long_term_memory_pipeline.py").read_text(encoding="utf-8")

        self.assertNotIn("policy_config", llm_source)
        self.assertIn("You are {role}", llm_source)
        self.assertNotIn("_mock_", llm_source)
        self.assertNotIn("_contains_any", llm_source)
        self.assertNotIn("OPENAI_API_KEY", llm_source)
        self.assertNotIn("EMBEDED_AGENT_LLM", llm_source)
        self.assertNotIn("Extract durable long-term memory candidates", memory_source)
        self.assertNotIn("Decide whether this interaction may contain durable", memory_source)
        self.assertNotIn("Review long-term memory candidates. Reject vague", memory_source)
        self.assertIn("memory_critic", memory_source)
        self.assertIn("read_prompt(self.observer_prompt_path)", memory_source)
        self.assertIn("read_prompt(self.extractor_prompt_path)", memory_source)
        self.assertIn("read_prompt(self.critic_prompt_path)", memory_source)

        observer_prompt = Path("src/agent/memory/prompts/memory_observer.md").read_text(encoding="utf-8")
        extractor_prompt = Path("src/agent/memory/prompts/memory_extractor.md").read_text(encoding="utf-8")
        self.assertIn("durable", observer_prompt.lower())
        self.assertIn("behavior_preference", extractor_prompt)


if __name__ == "__main__":
    unittest.main()
