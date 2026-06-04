from __future__ import annotations

from pathlib import Path

from src.agent import decision


ROOT = Path(__file__).resolve().parents[1]
DECISION_DIR = ROOT / "src" / "agent" / "decision"
MEMORY_DIR = ROOT / "src" / "agent" / "memory"


def test_rule_centered_files_do_not_exist() -> None:
    removed = [
        DECISION_DIR / "planner" / "rule_planner.py",
        DECISION_DIR / "candidate" / "candidate_generator.py",
        DECISION_DIR / "policy" / "policy_engine.py",
        DECISION_DIR / "arbiter" / "intent_arbiter.py",
        DECISION_DIR / "processor" / "processor_registry.py",
        DECISION_DIR / "understanding" / "text_understanding.py",
        DECISION_DIR / "policies" / "candidate_rules.yaml",
        MEMORY_DIR / "behavior" / "behavior_stats.py",
        MEMORY_DIR / "extractor" / "insight_extractor.py",
        MEMORY_DIR / "policy" / "memory_policy.py",
    ]

    for path in removed:
        assert not path.exists()


def test_decision_package_exports_llm_centered_entrypoints() -> None:
    assert set(decision.__all__) == {
        "ActionRealizer",
        "AgentIntent",
        "DecisionPipeline",
        "DecisionResult",
        "DeterministicGuard",
        "IntentPlan",
        "IntentPlanValidator",
    }


def test_core_uses_new_orchestrated_pipeline() -> None:
    source = (ROOT / "src" / "agent" / "core.py").read_text(encoding="utf-8")
    assert "DecisionPipeline" in source
    assert "LongTermMemoryPipeline" in source
    assert "PersonalContextBuilder" in source
    assert "RuntimeHistoryService" in source
    assert "LLMAgentOrchestrator" not in source
    assert "DecisionPipeline" in source
