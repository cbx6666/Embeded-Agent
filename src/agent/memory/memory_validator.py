from __future__ import annotations

"""长期记忆写入校验器。

它是什么：
MemoryValidator 是 LongTermMemoryStore 之前的确定性边界，校验候选类型、内容、证据、
置信度和来源。

它不是什么：
它不是 LLM critic，不判断“用户到底是不是这样的人”，不写 store，也不写 profile。

为什么存在：
LLM 可以参与提取，但最终写入必须由确定性代码把关，防止无证据、低质量或越权的内容进入
长期记忆。

边界：
所有 LongTermMemory 写入都必须先经过本类；UserProfile 字段不允许从这里写入。
"""

from typing import Any

from src.agent.memory.memory_candidate import ALLOWED_LONG_TERM_MEMORY_TYPES, MemoryCandidate


GROUNDED_EVIDENCE_KEYS = {
    "event",
    "event_type",
    "source_event_type",
    "dialogue",
    "snippet",
    "action",
    "action_type",
    "result",
    "outcome",
    "timestamp",
}

WEAK_EVIDENCE_SOURCES = {"llm", "model", "inference", "summary"}
MOCK_EVIDENCE_SOURCES = {"mock_llm", "local_mock"}
USER_EXPRESSION_EVENT_TYPES = {"user_text_input", "speech_recognized"}


class MemoryValidator:
    """候选长期记忆的确定性写入边界。"""

    def validate(self, candidate: MemoryCandidate) -> str | None:
        """返回 None 表示可写入，否则返回拒绝原因。"""

        if candidate.memory_type not in ALLOWED_LONG_TERM_MEMORY_TYPES:
            return f"invalid memory_type: {candidate.memory_type}"
        if not candidate.content:
            return "memory content is empty"
        if not candidate.evidence:
            return "memory evidence is required"
        evidence_error = _evidence_quality_error(candidate.evidence)
        if evidence_error:
            return evidence_error
        dialogue_error = _dialogue_preference_evidence_error(candidate)
        if dialogue_error:
            return dialogue_error
        if candidate.confidence < 0.0 or candidate.confidence > 1.0:
            return "memory confidence out of range"
        if candidate.memory_type == "behavior_preference" and candidate.source == "profile":
            return "explicit profile data must stay in UserProfile"
        return None


def _evidence_quality_error(evidence: list[dict[str, Any]]) -> str | None:
    """长期记忆证据必须能回指 event/dialogue/action outcome。"""

    if _only_mock_evidence(evidence):
        return "mock evidence cannot be stored as long-term memory"

    grounded = False
    for item in evidence:
        if not isinstance(item, dict) or not item:
            continue
        keys = {str(key) for key in item.keys()}
        if keys & GROUNDED_EVIDENCE_KEYS:
            grounded = True
            continue
        source = str(item.get("source", "")).strip().lower()
        if source and source not in WEAK_EVIDENCE_SOURCES and len(item) >= 2:
            grounded = True

    if not grounded:
        return "memory evidence is not grounded in event/dialogue/action outcome"
    return None


def _only_mock_evidence(evidence: list[dict[str, Any]]) -> bool:
    sources = {
        str(item.get("source", "")).strip().lower()
        for item in evidence
        if isinstance(item, dict) and item.get("source")
    }
    return bool(sources) and sources <= MOCK_EVIDENCE_SOURCES


def _dialogue_preference_evidence_error(candidate: MemoryCandidate) -> str | None:
    """用户偏好类长期记忆必须带可回指的 dialogue evidence。"""

    if candidate.memory_type != "behavior_preference":
        return None
    for item in candidate.evidence:
        if not isinstance(item, dict):
            continue
        source_event_type = str(item.get("source_event_type", "")).strip()
        has_event = source_event_type in USER_EXPRESSION_EVENT_TYPES
        has_timestamp = item.get("timestamp") is not None
        has_source = bool(item.get("source"))
        has_text = bool(item.get("user_text") or item.get("snippet"))
        if has_event and has_timestamp and has_source and has_text:
            return None
    return "preference memory requires user_text_input/speech_recognized evidence with timestamp, source, and user_text/snippet"
