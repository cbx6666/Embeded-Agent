from __future__ import annotations

import json
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.action import Action
from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.agent_context_builder import AgentContextBuilder
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.guard import DeterministicGuard, GuardConfig
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.event import Event
from src.agent.user.personal_context_builder import PersonalContextBuilder
import src.agent.memory.long_term_memory_pipeline as long_term_memory_pipeline_module
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.memory.memory_candidate import MemoryCandidate
from src.agent.memory.memory_validator import MemoryValidator
from src.agent.runtime.action_result import ActionResult
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.user_profile_service import UserProfileService
from src.storage.long_term_memory_store import LongTermMemoryStore
from src.storage.user_profile_store import UserProfileStore
from tests.fakes.fake_llm_service import CapturingFakeLLMService, FakeLLMService


ScriptedLLM = FakeLLMService
CapturingLLM = CapturingFakeLLMService


class LLMCenteredArchitectureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_multi_intent_complex_natural_language(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": _json(
                    summary="User wants to continue focus but reduce interruption.",
                    user_intent="continue focus with fewer reminders",
                    should_respond=True,
                ),
                "intent_planner": json.dumps(
                    {
                        "intents": [
                            {"type": "continue_focus", "priority": 90, "reason": "continue", "payload": {"duration_minutes": 20}},
                            {"type": "reduce_reminder_frequency", "priority": 70, "reason": "less interruption", "payload": {}},
                            {"type": "answer_user", "priority": 60, "reason": "ack", "payload": {}, "requires_llm": True},
                        ],
                        "reasoning": "Respect the user request.",
                        "risk_level": "low",
                        "interrupt_user": False,
                    }
                ),
                "safety_critic": _safety("approve"),
                "response_writer": _response("Got it. I will keep reminders lighter."),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=10, payload={"text": "I am tired but want 20 more minutes; stop reminding me so much."}),
            llm_service=llm,
        )

        self.assertEqual({"continue_focus", "reduce_reminder_frequency", "answer_user"}, {intent.type for intent in result.intents})
        self.assertIn("start_timer", {action.type for action in result.actions})
        self.assertIn("speak", {action.type for action in result.actions})
        self.assertEqual(llm.calls[:4], ["situation_analyst", "intent_planner", "safety_critic", "response_writer"])

    def test_llm_output_illegal_schema_is_rejected(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": _json(summary="test", should_respond=True),
                "intent_planner": json.dumps(
                    {
                        "intents": [{"type": "hack_system", "priority": 100, "reason": "bad", "payload": {}}],
                        "reasoning": "bad",
                        "risk_level": "low",
                    }
                ),
                "safety_critic": _safety("approve"),
                "response_writer": _response("This should not become an action."),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=20, payload={"text": "do something unsafe"}),
            llm_service=llm,
        )

        self.assertEqual(result.actions, [])
        self.assertEqual(result.intents[0].type, "no_op")
        self.assertIn("validation:", str(result.fallback_reason))

    def test_safety_critic_revise(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": _json(summary="User is busy.", should_respond=True),
                "intent_planner": json.dumps(
                    {
                        "intents": [{"type": "suggest_rest", "priority": 90, "reason": "tired", "payload": {}}],
                        "reasoning": "Suggest rest.",
                        "risk_level": "low",
                    }
                ),
                "safety_critic": json.dumps(
                    {
                        "decision": "revise",
                        "reason": "Avoid interrupting; acknowledge quietly.",
                        "revised_plan": {
                            "intents": [{"type": "display_update", "priority": 40, "reason": "quiet revision", "payload": {}}],
                            "reasoning": "Quiet display only.",
                            "risk_level": "low",
                        },
                    }
                ),
                "response_writer": _response("I will keep this quiet."),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=30, payload={"text": "not now"}),
            llm_service=llm,
        )

        self.assertEqual([intent.type for intent in result.intents], ["display_update"])
        self.assertEqual(result.safety_review.decision, "revise")  # type: ignore[union-attr]
        self.assertEqual({action.type for action in result.actions}, {"display"})

    def test_response_writer_removes_false_preference_commitment(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": _json(summary="User asks for fewer reminders.", should_respond=True),
                "intent_planner": json.dumps(
                    {
                        "intents": [
                            {"type": "answer_user", "priority": 60, "reason": "ack", "payload": {}, "requires_llm": True},
                            {"type": "reduce_reminder_frequency", "priority": 50, "reason": "less interruption", "payload": {}},
                        ],
                        "reasoning": "Acknowledge without persistence.",
                        "risk_level": "low",
                    }
                ),
                "safety_critic": _safety("approve"),
                "response_writer": _response("我记住了，以后我都会少提醒你。"),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=35, payload={"text": "别太频繁提醒我"}),
            llm_service=llm,
        )

        visible_texts = [str(action.payload.get("text", "")) for action in result.actions if action.type in {"speak", "display"}]
        self.assertIn("好的，我会尽量少打扰你。", visible_texts)
        self.assertNotIn("我记住了，以后我都会少提醒你。", visible_texts)

    def test_memory_extraction_and_consolidation(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        candidate = {
            "memory_type": "behavior_preference",
            "content": "User prefers fewer rest reminders.",
            "confidence": 0.82,
            "evidence": [
                {
                    "source_event_type": "user_text_input",
                    "timestamp": 40,
                    "user_text": "Please remind me less.",
                    "source": "dialogue",
                }
            ],
            "metadata": {"preference_key": "reminder_frequency", "preference_value": "low_frequency"},
        }
        llm = ScriptedLLM(
            {
                "memory_observer": json.dumps({"worth_remembering": True, "reason": "preference"}),
                "memory_extractor": json.dumps({"candidates": [candidate]}),
                "memory_critic": json.dumps({"approved_indexes": [0], "rejected_reasons": []}),
                "memory_consolidator": json.dumps({"candidates": [candidate]}),
            }
        )
        event = Event(type="user_text_input", timestamp=40, payload={"text": "Please remind me less."})
        result = LongTermMemoryPipeline(store).process_event(
            "u1",
            event,
            AgentState(current_user_id="u1"),
            llm,
        )

        self.assertEqual(len(result.stored), 1)
        self.assertEqual(store.list("u1")[0].content, "User prefers fewer rest reminders.")

        result2 = LongTermMemoryPipeline(store).process_event("u1", event, AgentState(current_user_id="u1"), llm)
        self.assertEqual(len(store.list("u1")), 1)
        self.assertGreaterEqual(len(result2.stored[0].evidence), 1)

    def test_memory_validator_rejects_memory_without_evidence(self) -> None:
        validator = MemoryValidator()
        candidate = MemoryCandidate(
            memory_type="behavior_preference",
            content="User prefers quiet reminders.",
            confidence=0.8,
            evidence=[],
        )

        self.assertEqual(validator.validate(candidate), "memory evidence is required")

    def test_memory_validator_rejects_ungrounded_evidence(self) -> None:
        validator = MemoryValidator()
        candidate = MemoryCandidate(
            memory_type="behavior_preference",
            content="User prefers quiet reminders.",
            confidence=0.8,
            evidence=[{"source": "llm"}],
        )

        self.assertEqual(
            validator.validate(candidate),
            "memory evidence is not grounded in event/dialogue/action outcome",
        )

    def test_memory_validator_rejects_mock_evidence(self) -> None:
        validator = MemoryValidator()
        candidate = MemoryCandidate(
            memory_type="behavior_preference",
            content="User prefers quiet reminders.",
            confidence=0.8,
            evidence=[{"source": "mock_llm", "snippet": "latest interaction"}],
        )

        self.assertEqual(
            validator.validate(candidate),
            "mock evidence cannot be stored as long-term memory",
        )

    def test_llm_service_finds_project_env_when_started_from_src(self) -> None:
        root = self.root / "project"
        src_dir = root / "src"
        src_dir.mkdir(parents=True)
        (root / ".env").write_text(
            "DEEPSEEK_API_KEY=test-key\nDEEPSEEK_BASE_URL=https://example.invalid/v1\nDEEPSEEK_MODEL=test-model\n",
            encoding="utf-8",
        )

        old_cwd = Path.cwd()
        try:
            os.chdir(src_dir)
            llm = LLMService()
        finally:
            os.chdir(old_cwd)

        self.assertEqual(llm.api_key, "test-key")
        self.assertEqual(llm.base_url, "https://example.invalid/v1")
        self.assertEqual(llm.model, "test-model")

    def test_llm_service_requires_deepseek_config(self) -> None:
        env_path = self.root / "empty.env"
        env_path.write_text("", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DeepSeek API is not configured."):
                LLMService(env_path=env_path)

    def test_llm_service_no_longer_has_mock_backend(self) -> None:
        source = Path("src/services/llm_service.py").read_text(encoding="utf-8")

        self.assertNotIn("_mock_", source)
        self.assertNotIn("mock_llm", source)
        self.assertNotIn("_contains_any", source)
        self.assertNotIn("_mock_intent_from_prompt", source)
        self.assertNotIn("_mock_memory_item", source)

    def test_llm_service_only_uses_deepseek_env(self) -> None:
        source = Path("src/services/llm_service.py").read_text(encoding="utf-8")
        env_path = self.root / "legacy.env"
        env_path.write_text(
            "OPENAI_API_KEY=openai\nEMBEDED_AGENT_LLM_API_KEY=embedded\n",
            encoding="utf-8",
        )

        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertNotIn("OPENAI_MODEL", source)
        self.assertNotIn("OPENAI_BASE_URL", source)
        self.assertNotIn("EMBEDED_AGENT_LLM", source)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DeepSeek API is not configured."):
                LLMService(env_path=env_path)

    def test_memory_decay_confidence_contradiction_and_user_isolation(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        quiet = MemoryCandidate(
            memory_type="behavior_preference",
            content="User prefers quiet reminders.",
            confidence=0.6,
            evidence=[{"event": "user_text_input", "timestamp": 100}],
            metadata={"preference_key": "reminder_style", "preference_value": "quiet"},
        )
        first = store.upsert_candidate("u1", quiet, timestamp=100)
        second = store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="User prefers quiet reminders.",
                confidence=0.7,
                evidence=[{"event": "user_text_input", "timestamp": 200}],
                metadata={"preference_key": "reminder_style", "preference_value": "quiet"},
            ),
            timestamp=200,
        )
        store.upsert_candidate("u2", quiet, timestamp=100)

        self.assertGreater(second.confidence, first.confidence)
        store.apply_decay(now=200 + 90 * 86400)
        self.assertLess(store.list("u1")[0].decay, 1.0)

        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="User now prefers energetic spoken reminders.",
                confidence=0.85,
                evidence=[{"event": "user_text_input", "timestamp": 300}],
                metadata={"preference_key": "reminder_style", "preference_value": "loud"},
            ),
            timestamp=300,
        )

        active_u1 = store.list("u1")
        active_u2 = store.list("u2")
        all_u1 = store.list("u1", include_inactive=True)
        self.assertEqual(len(active_u1), 1)
        self.assertIn("energetic", active_u1[0].content)
        self.assertEqual(active_u2[0].content, "User prefers quiet reminders.")
        self.assertTrue(any(item.status == "contradicted" for item in all_u1))

    def test_memory_store_canonicalizes_behavior_preference_key_value(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="User prefers gentle reminders.",
                confidence=0.65,
                evidence=[{"source_event_type": "user_text_input", "timestamp": 100, "source": "dialogue", "user_text": "be gentle"}],
                metadata={"preference_key": "reminder_style", "preference_value": "gentle"},
            ),
            timestamp=100,
        )

        second = store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="User likes a calm reminder style.",
                confidence=0.72,
                evidence=[{"source_event_type": "user_text_input", "timestamp": 200, "source": "dialogue", "user_text": "keep reminders calm"}],
                metadata={"preference_key": "reminder_style", "preference_value": "gentle"},
            ),
            timestamp=200,
        )

        memories = store.list("u1")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, second.id)
        self.assertEqual(len(memories[0].evidence), 2)
        self.assertEqual(memories[0].updated_at, 200)

    def test_personal_context_and_context_retrieval(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="User prefers quiet visual feedback.",
                confidence=0.9,
                evidence=[{"event": "user_text_input"}],
            ),
            timestamp=50,
        )
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="interaction_style",
                content="Use concise responses during focus.",
                confidence=0.8,
                evidence=[{"event": "user_text_input"}],
            ),
            timestamp=51,
        )
        state = AgentState(current_user_id="u1")
        personal_context = PersonalContextBuilder(long_term_memory_store=store).build(user_id="u1", state=state)
        context = AgentContextBuilder().build(
            previous_state=AgentState(current_user_id="u1"),
            current_state=state,
            event=Event(type="user_text_input", timestamp=52, payload={"text": "quiet please"}),
            personal_context=personal_context,
        )

        self.assertEqual(len(personal_context.behavior_preferences), 1)
        self.assertTrue(any("quiet" in item["content"] for item in context.relevant_memories))

    def test_personal_context_profile_memory_conflict_fusion(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        profile_service = UserProfileService(UserProfileStore(self.root / "profiles.json"))
        profile_service.update_preference("u1", "reminder_style", "gentle", timestamp=10)
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="User prefers loud reminders.",
                confidence=0.9,
                evidence=[{"event": "user_text_input", "timestamp": 11}],
                metadata={"preference_key": "reminder_style", "preference_value": "loud"},
            ),
            timestamp=11,
        )

        personal_context = PersonalContextBuilder(
            long_term_memory_store=store,
            user_profile_service=profile_service,
        ).build(
            user_id="u1",
            state=AgentState(current_user_id="u1"),
            event=Event(type="user_text_input", timestamp=12, payload={"text": "reminder style"}),
        )
        relevant = personal_context.retrieve_relevant(
            event_type="user_text_input",
            text="reminder",
            limit=3,
        )

        self.assertEqual(personal_context.behavior_preferences, ())
        self.assertEqual(personal_context.uncertain_memories[0]["conflict_with"], "UserProfile:reminder_style")
        self.assertEqual(relevant[0]["source"], "UserProfile")
        self.assertIn("reminder_style", relevant[0]["content"])

    def test_llm_fallback(self) -> None:
        llm = ScriptedLLM(
            {
                "situation_analyst": "{not-json",
                "intent_planner": "{not-json",
                "safety_critic": _safety("approve"),
            }
        )

        result = DecisionPipeline().decide(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=60, payload={"text": "hello"}),
            llm_service=llm,
        )

        self.assertIn("llm_fallback", str(result.fallback_reason))
        self.assertEqual([intent.type for intent in result.intents], ["answer_user"])
        self.assertIn("speak", {action.type for action in result.actions})

    def test_memory_pipeline_handles_llm_json_error(self) -> None:
        llm = ScriptedLLM(
            {
                "memory_observer": json.dumps({"worth_remembering": True, "reason": "possible memory"}),
                "memory_extractor": "{not-json",
            }
        )
        result = LongTermMemoryPipeline(LongTermMemoryStore(self.root / "memory.json")).process_event(
            "u1",
            Event(type="user_text_input", timestamp=61, payload={"text": "Please remind me less."}),
            AgentState(current_user_id="u1"),
            llm,
        )

        self.assertEqual(result.stored, [])
        self.assertTrue(result.stage_metadata["memory_extractor"]["fallback"])

    def test_memory_preference_extraction_with_fake_llm(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        user_text = "\u6211\u4e0d\u559c\u6b22\u592a\u9891\u7e41\u63d0\u9192\u6211\uff0c\u6700\u597d\u6e29\u548c\u4e00\u70b9"
        candidate = {
            "memory_type": "behavior_preference",
            "content": "\u7528\u6237\u504f\u597d\u66f4\u6e29\u548c\u3001\u4f4e\u9891\u7387\u7684\u63d0\u9192\u65b9\u5f0f\u3002",
            "confidence": 0.86,
            "evidence": [
                {
                    "source_event_type": "user_text_input",
                    "timestamp": 63,
                    "user_text": user_text,
                    "source": "dialogue",
                }
            ],
            "metadata": {
                "preference_key": "reminder_style",
                "preference_value": "gentle",
                "secondary_preference_key": "reminder_frequency",
                "secondary_preference_value": "low_frequency",
            },
        }
        llm = ScriptedLLM(
            {
                "memory_observer": json.dumps({"worth_remembering": True, "reason": "explicit preference"}),
                "memory_extractor": json.dumps({"candidates": [candidate]}),
                "memory_critic": json.dumps({"approved_indexes": [0], "rejected_reasons": []}),
                "memory_consolidator": json.dumps({"candidates": [candidate]}),
            }
        )

        result = LongTermMemoryPipeline(store).process_event(
            "u1",
            Event(type="user_text_input", timestamp=63, payload={"text": user_text}),
            AgentState(current_user_id="u1"),
            llm,
        )

        self.assertTrue(result.stored)
        stored = store.list("u1")[0]
        self.assertEqual(stored.memory_type, "behavior_preference")
        self.assertIn("\u6e29\u548c", stored.content)
        self.assertEqual(stored.metadata["preference_key"], "reminder_style")
        self.assertEqual(stored.metadata["preference_value"], "gentle")

    def test_memory_pipeline_uses_fake_llm_in_tests(self) -> None:
        llm = FakeLLMService()
        store = LongTermMemoryStore(self.root / "memory.json")

        LongTermMemoryPipeline(store).process_event(
            "u1",
            Event(type="user_text_input", timestamp=67, payload={"text": "I prefer quiet reminders."}),
            AgentState(current_user_id="u1"),
            llm,
        )

        self.assertIn("memory_observer", llm.calls)
        self.assertIn("memory_extractor", llm.calls)
        self.assertTrue(store.list("u1"))

    def test_fake_llm_service_still_supports_pipeline_tests(self) -> None:
        llm = FakeLLMService(
            {
                "memory_observer": json.dumps({"worth_remembering": True, "reason": "scripted"}),
                "memory_extractor": json.dumps({"candidates": []}),
            }
        )

        result = LongTermMemoryPipeline(LongTermMemoryStore(self.root / "memory.json")).process_event(
            "u1",
            Event(type="user_text_input", timestamp=68, payload={"text": "hello"}),
            AgentState(current_user_id="u1"),
            llm,
        )

        self.assertEqual(result.candidates, [])
        self.assertIn("memory_observer", llm.calls)
        self.assertIn("memory_extractor", llm.calls)

    def test_production_pipeline_has_no_keyword_semantic_rules(self) -> None:
        source = inspect.getsource(long_term_memory_pipeline_module)

        self.assertNotIn("_dialogue_preference_candidates", source)
        self.assertNotIn("wants_gentle", source)
        self.assertNotIn("dislikes_frequent", source)
        self.assertNotIn("likes_break", source)
        self.assertNotIn("dislikes_content", source)
        self.assertNotIn("\u592a\u9891\u7e41", source)
        self.assertNotIn("\u6e29\u548c", source)

    def test_pipeline_does_not_use_keyword_rules(self) -> None:
        self.test_production_pipeline_has_no_keyword_semantic_rules()

    def test_memory_candidate_requires_dialogue_evidence(self) -> None:
        candidate = MemoryCandidate(
            memory_type="behavior_preference",
            content="User prefers gentle reminders.",
            confidence=0.8,
            evidence=[{"source_event_type": "user_text_input", "timestamp": 64, "source": "dialogue"}],
            metadata={"preference_key": "reminder_style", "preference_value": "gentle"},
        )

        self.assertEqual(
            MemoryValidator().validate(candidate),
            "preference memory requires user_text_input/speech_recognized evidence with timestamp, source, and user_text/snippet",
        )

    def test_memory_candidate_rejects_non_dialogue_preference_evidence(self) -> None:
        candidate = MemoryCandidate(
            memory_type="behavior_preference",
            content="User prefers calm interaction.",
            confidence=0.8,
            evidence=[
                {
                    "source_event_type": "user_fatigue_updated",
                    "timestamp": 64,
                    "source": "sensor",
                    "snippet": "fatigue high",
                }
            ],
            metadata={"preference_key": "interaction_style", "preference_value": "calm"},
        )

        self.assertEqual(
            MemoryValidator().validate(candidate),
            "preference memory requires user_text_input/speech_recognized evidence with timestamp, source, and user_text/snippet",
        )

    def test_gentle_reminder_preference_affects_response_context_or_fallback(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="behavior_preference",
                content="用户偏好更温和、低频率的提醒方式。",
                confidence=0.9,
                evidence=[{"event": "user_text_input", "timestamp": 65}],
                metadata={
                    "preference_key": "reminder_style",
                    "preference_value": "gentle",
                    "secondary_preference_key": "reminder_frequency",
                    "secondary_preference_value": "low_frequency",
                },
            ),
            timestamp=65,
        )
        state = AgentState(current_user_id="u1")
        personal_context = PersonalContextBuilder(long_term_memory_store=store).build(
            user_id="u1",
            state=state,
            event=Event(type="system_triggered", timestamp=66, payload={"trigger": "focus_health_check"}),
        )
        context = AgentContextBuilder().build(
            previous_state=AgentState(current_user_id="u1"),
            current_state=state,
            event=Event(type="system_triggered", timestamp=66, payload={"trigger": "focus_health_check"}),
            personal_context=personal_context,
        )

        guidance = context.to_prompt_dict()["personalization_guidance"]
        self.assertTrue(guidance["relevant_long_term_memory"])
        self.assertTrue(any("gentle" in hint or "settings" in hint for hint in guidance["response_style_hints"]))

    def test_process_actions_observes_action_result_outcome(self) -> None:
        llm = CapturingLLM(
            {
                "memory_observer": json.dumps({"worth_remembering": False, "reason": "outcome only"}),
            }
        )
        LongTermMemoryPipeline(LongTermMemoryStore(self.root / "memory.json")).process_actions(
            "u1",
            [Action(type="set_light_state", payload={"state": "alert"})],
            62,
            action_results=[
                ActionResult(
                    action_type="set_light_state",
                    success=False,
                    timestamp=62,
                    reason="hardware_offline",
                )
            ],
            source_event=Event(type="system_triggered", timestamp=62, payload={"trigger": "test"}),
            state=AgentState(current_user_id="u1"),
            llm_service=llm,
        )

        observer_prompt = next(prompt for role, prompt in llm.prompts if role == "memory_observer")
        self.assertIn("action_results", observer_prompt)
        self.assertIn("hardware_offline", observer_prompt)

    def test_guard_rejection(self) -> None:
        state = AgentState()
        state.user.presence = "away"
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="system_triggered", timestamp=70, payload={"trigger": "focus_health_check"}),
            personal_context=None,
        )

        decision = DeterministicGuard().filter(
            IntentPlan(intents=[AgentIntent(type="suggest_rest", reason="rest")]),
            context,
        )

        self.assertEqual(decision.plan.intents[0].type, "no_op")
        self.assertEqual(decision.blocked_intents[0].type, "suggest_rest")

    def test_guard_config_controls_cooldown_without_rule_registry(self) -> None:
        state = AgentState()
        state.cooldown.reminder_last_ts["rest_reminder"] = 100
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=state,
            event=Event(type="system_triggered", timestamp=130, payload={"trigger": "focus_health_check"}),
            personal_context=None,
        )

        blocked = DeterministicGuard(config=GuardConfig(reminder_cooldown_sec=60)).filter(
            IntentPlan(intents=[AgentIntent(type="suggest_rest", reason="rest")]),
            context,
        )
        allowed = DeterministicGuard(config=GuardConfig(reminder_cooldown_sec=10)).filter(
            IntentPlan(intents=[AgentIntent(type="suggest_rest", reason="rest")]),
            context,
        )

        self.assertEqual(blocked.plan.intents[0].type, "no_op")
        self.assertEqual(allowed.plan.intents[0].type, "suggest_rest")

    def test_action_realization(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="focus_start_requested", timestamp=80, payload={"duration_sec": 600}),
            personal_context=None,
        )
        actions = ActionRealizer().realize(
            IntentPlan(
                intents=[
                    AgentIntent(type="start_focus", priority=80),
                    AgentIntent(type="set_tts_volume", priority=70, payload={"volume": 25}),
                ]
            ),
            response=type("Draft", (), {"speak_text": "", "display_text": ""})(),
            context=context,
        )

        self.assertEqual(actions[0].type, "start_timer")
        self.assertEqual(actions[0].payload["duration_sec"], 600)
        self.assertIn("set_tts_volume", {action.type for action in actions})

    def test_action_realizer_deduplicates_repeated_visible_actions(self) -> None:
        context = AgentContextBuilder().build(
            previous_state=AgentState(),
            current_state=AgentState(),
            event=Event(type="user_text_input", timestamp=81, payload={"text": "ok"}),
            personal_context=None,
        )

        actions = ActionRealizer().realize(
            IntentPlan(
                intents=[
                    AgentIntent(type="answer_user", priority=90, reason="ack"),
                    AgentIntent(type="display_update", priority=80, reason="second", payload={"text": "same"}),
                ]
            ),
            response=type("Draft", (), {"speak_text": "same", "display_text": "same"})(),
            context=context,
        )

        self.assertEqual(1, sum(1 for action in actions if action.type == "display" and action.payload.get("text") == "same"))

    def test_personal_context_builder_reads_long_term_memory(self) -> None:
        store = LongTermMemoryStore(self.root / "memory.json")
        store.upsert_candidate(
            "u1",
            MemoryCandidate(
                memory_type="active_constraint",
                content="Do not speak during silent mode.",
                confidence=0.9,
                evidence=[{"event": "user_text_input"}],
            ),
            timestamp=90,
        )

        personal_context = PersonalContextBuilder(long_term_memory_store=store).build(
            user_id="u1",
            state=AgentState(current_user_id="u1"),
        )

        self.assertEqual(personal_context.active_constraints[0]["content"], "Do not speak during silent mode.")


def _json(**values: object) -> str:
    payload = {
        "summary": values.get("summary", "summary"),
        "user_intent": values.get("user_intent", "intent"),
        "current_state": values.get("current_state", "state"),
        "risks": values.get("risks", []),
        "uncertainties": values.get("uncertainties", []),
        "should_respond": values.get("should_respond", False),
        "risk_level": values.get("risk_level", "low"),
    }
    return json.dumps(payload)


def _safety(decision: str) -> str:
    return json.dumps({"decision": decision, "reason": "ok", "revised_plan": None})


def _response(text: str) -> str:
    return json.dumps({"speak_text": text, "display_text": text, "tone": "calm"})


if __name__ == "__main__":
    unittest.main()
