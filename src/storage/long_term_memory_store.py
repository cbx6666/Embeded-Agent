from __future__ import annotations

"""LongTermMemory 持久化仓库。

它是什么：
LongTermMemoryStore 保存已经通过 MemoryValidator 的长期记忆，并负责去重、证据合并、
置信度更新、衰减计算和冲突标记。

它不是什么：
它不是 LLM 调用器，不判断“是否值得记住”，不保存 UserProfile，也不应该被
DecisionPipeline 直接读取。

为什么存在：
长期记忆需要独立持久化、可追溯证据、可衰减和可处理矛盾。仓库只做确定性的数据
维护；LLM 永远只能提出候选，不能直接写 state/store/profile。

边界：
写入者是 LongTermMemoryPipeline；读取者是 PersonalContextBuilder。
"""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from src.agent.memory.long_term_memory import LongTermMemory
from src.agent.memory.memory_candidate import MemoryCandidate


class LongTermMemoryStore:
    """JSON 文件形式的长期记忆仓库。"""

    def __init__(self, path: str | Path = "data/memory/long_term_memory.json") -> None:
        self.path = Path(path)

    def list(
        self,
        user_id: str | None = None,
        *,
        include_inactive: bool = False,
        now: int | None = None,
    ) -> list[LongTermMemory]:
        """读取全部或指定用户的长期记忆。

        默认只返回 active 记忆。传入 now 时会先刷新 decay，这让读取方可以用当前事件
        时间得到一致的有效置信度。
        """

        memories = self._load()
        if now is not None and _apply_decay(memories, int(now)):
            self._save(memories)
        if not include_inactive:
            memories = [item for item in memories if item.status == "active"]
        if user_id is None:
            return memories
        return [item for item in memories if item.user_id == user_id]

    def apply_decay(self, *, now: int | None = None) -> list[LongTermMemory]:
        """显式刷新 decay，主要供维护任务和行为测试使用。"""

        timestamp = int(time.time()) if now is None else int(now)
        memories = self._load()
        if _apply_decay(memories, timestamp):
            self._save(memories)
        return memories

    def upsert_candidate(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        timestamp: int | None = None,
    ) -> LongTermMemory:
        """写入或合并一条通过验证的候选记忆。"""

        now = int(time.time()) if timestamp is None else int(timestamp)
        memories = self._load()
        memory_id = _canonical_behavior_preference_id(memories, user_id, candidate) or _memory_id(
            user_id,
            candidate.memory_type,
            candidate.content,
        )
        contradicted_ids = _mark_contradictions(memories, user_id, candidate, now, memory_id)

        for item in memories:
            if item.id == memory_id:
                merged_evidence = _merge_evidence(item.evidence, candidate.evidence)
                item.confidence = _updated_confidence(
                    base=item.confidence,
                    candidate=candidate.confidence,
                    new_evidence_count=max(0, len(merged_evidence) - len(item.evidence)),
                )
                item.evidence = merged_evidence
                item.metadata.update(candidate.metadata)
                if candidate.content and candidate.confidence >= item.confidence:
                    item.content = candidate.content
                if contradicted_ids:
                    item.metadata["contradicts"] = contradicted_ids
                item.updated_at = now
                item.last_accessed_at = now
                item.decay = 1.0
                item.status = "active"
                item.contradiction_of = _candidate_contradiction_of(candidate, contradicted_ids)
                self._save(memories)
                return item

        stored = LongTermMemory(
            id=memory_id,
            user_id=user_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            confidence=candidate.confidence,
            evidence=list(candidate.evidence),
            source=candidate.source,
            metadata=dict(candidate.metadata),
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            contradiction_of=_candidate_contradiction_of(candidate, contradicted_ids),
        )
        if contradicted_ids:
            stored.metadata["contradicts"] = contradicted_ids
        memories.append(stored)
        self._save(memories)
        return stored

    def replace_user_memories(self, user_id: str, memories: list[LongTermMemory]) -> None:
        """替换指定用户的长期记忆集合，供维护或迁移工具使用。"""

        all_memories = [item for item in self._load() if item.user_id != user_id]
        all_memories.extend(memories)
        self._save(all_memories)

    def _load(self) -> list[LongTermMemory]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw = data.get("memories", [])
        if not isinstance(raw, list):
            return []
        memories: list[LongTermMemory] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                memories.append(LongTermMemory.from_dict(item))
            except (TypeError, ValueError):
                continue
        return memories

    def _save(self, memories: list[LongTermMemory]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            "memories": [
                item.to_dict()
                for item in sorted(memories, key=lambda x: (x.user_id, x.memory_type, x.content))
            ],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_id(user_id: str, memory_type: str, content: str) -> str:
    raw = f"{user_id}\n{memory_type}\n{content.strip().lower()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _canonical_behavior_preference_id(
    memories: list[LongTermMemory],
    user_id: str,
    candidate: MemoryCandidate,
) -> str | None:
    if candidate.memory_type != "behavior_preference":
        return None
    candidate_key = _metadata_value(candidate.metadata, "preference_key", "profile_key")
    candidate_value = _metadata_value(candidate.metadata, "preference_value", "profile_value")
    if not candidate_key or not candidate_value:
        return None

    matches = [
        memory
        for memory in memories
        if memory.user_id == user_id
        and memory.memory_type == "behavior_preference"
        and _metadata_value(memory.metadata, "preference_key", "profile_key") == candidate_key
        and _metadata_value(memory.metadata, "preference_value", "profile_value") == candidate_value
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item.status == "active", item.updated_at, item.confidence), reverse=True)
    return matches[0].id


def _merge_evidence(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(left)
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in merged}
    for item in right:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            merged.append(dict(item))
            seen.add(key)
    return merged[-20:]


def _updated_confidence(*, base: float, candidate: float, new_evidence_count: int) -> float:
    """重复证据会提升置信度，但不会让单次 LLM 候选无限放大。"""

    evidence_bonus = min(0.12, max(0, new_evidence_count) * 0.04)
    blended = base * 0.65 + candidate * 0.35 + evidence_bonus
    return max(0.0, min(1.0, max(base, blended)))


def _apply_decay(memories: list[LongTermMemory], now: int) -> bool:
    changed = False
    for memory in memories:
        if memory.status != "active":
            continue
        next_decay = _decay_for(memory, now)
        if abs(next_decay - memory.decay) > 0.0001:
            memory.decay = next_decay
            changed = True
    return changed


def _decay_for(memory: LongTermMemory, now: int) -> float:
    updated_at = int(memory.updated_at or memory.created_at or now)
    age_days = max(0.0, (int(now) - updated_at) / 86400.0)
    if age_days <= 0:
        return 1.0
    evidence_bonus_days = min(45.0, len(memory.evidence) * 7.0)
    confidence_bonus_days = max(0.0, memory.confidence - 0.5) * 30.0
    half_life_days = 30.0 + evidence_bonus_days + confidence_bonus_days
    return max(0.1, min(1.0, math.pow(0.5, age_days / half_life_days)))


def _mark_contradictions(
    memories: list[LongTermMemory],
    user_id: str,
    candidate: MemoryCandidate,
    now: int,
    candidate_id: str,
) -> list[str]:
    explicit_ids = _metadata_id_list(candidate.metadata.get("contradicts"))
    contradicted: list[str] = []
    for memory in memories:
        if memory.user_id != user_id or memory.id == candidate_id or memory.status != "active":
            continue
        if memory.id in explicit_ids or _same_preference_key_conflicts(memory, candidate):
            memory.status = "contradicted"
            memory.updated_at = now
            memory.metadata["contradicted_by"] = candidate_id
            contradicted.append(memory.id)
    return contradicted


def _metadata_id_list(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value if str(item).strip()}
    return set()


def _same_preference_key_conflicts(memory: LongTermMemory, candidate: MemoryCandidate) -> bool:
    if memory.memory_type != candidate.memory_type:
        return False
    memory_key = _metadata_value(memory.metadata, "preference_key", "profile_key")
    candidate_key = _metadata_value(candidate.metadata, "preference_key", "profile_key")
    if not memory_key or memory_key != candidate_key:
        return False
    memory_value = _metadata_value(memory.metadata, "preference_value", "profile_value")
    candidate_value = _metadata_value(candidate.metadata, "preference_value", "profile_value")
    return bool(memory_value and candidate_value and memory_value != candidate_value)


def _metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return str(value).strip().lower()
    return ""


def _candidate_contradiction_of(candidate: MemoryCandidate, contradicted_ids: list[str]) -> str | None:
    explicit_ids = _metadata_id_list(candidate.metadata.get("contradicts"))
    if explicit_ids:
        return sorted(explicit_ids)[0]
    return contradicted_ids[0] if contradicted_ids else None
