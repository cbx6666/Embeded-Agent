from __future__ import annotations

"""结构化用户记忆服务（LLM 异步抽取版）。

职责：

1. **异步写入**：``speech_recognized`` 到来后，``submit_speech_memory`` 把用户言谈放进
   后台队列，主链路（``AgentCore.handle_event``）立即返回，绝不等待 memory LLM。
   后台线程调用 ``MemoryExtractor`` 做**一次** LLM 抽取，得到多维度 ``MemoryItem``，
   再做合并/去重后落盘。
2. **相关性检索**：三个 LLM 决策入口在调用前用 ``retrieve_user_context`` 按 context_type
   做轻量打分检索，返回结构化、分组后的记忆，作为 prompt 上下文。

这里**没有** critic / consolidator / validator 多阶段链路，只有「单次抽取 + 确定性合并」。
memory LLM 抽取失败只会被吞掉并跳过，不影响主响应。
"""

import json
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from src.agent.memory.memory_extractor import MemoryExtractor
from src.agent.memory.memory_model import (
    GROUP_KEY_BY_TYPE,
    MemoryItem,
)
from src.agent.policy_config import MemoryPolicy

# 提交给后台队列的任务：(user_id, text, timestamp, recent_messages)
_MemoryJob = tuple[str, str, int, list[dict[str, Any]]]

# 记忆轮换：刚被检索用过的记忆在该窗口内降权，鼓励跨轮使用不同记忆，避免重复。
_ROTATION_WINDOW_SEC = 180
_ROTATION_PENALTY = 0.6
_RETRIEVAL_SAVE_DEBOUNCE_SEC = 2.0


class MemoryService:
    """JSON 落地、单后台线程异步 LLM 抽取的结构化用户记忆。"""

    def __init__(
        self,
        policy: MemoryPolicy | None = None,
        *,
        store_path: str | Path | None = None,
        extractor: MemoryExtractor | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.policy = policy or MemoryPolicy()
        self._path = Path(store_path) if store_path is not None else Path(self.policy.store_path)
        self._lock = threading.RLock()
        self._store: dict[str, list[MemoryItem]] = self._load()

        # 抽取器：显式传入优先；否则若给了 llm_client 则自建；都没有则降级为不抽取。
        if extractor is not None:
            self._extractor: MemoryExtractor | None = extractor
        elif llm_client is not None:
            self._extractor = MemoryExtractor(llm_client)
        else:
            self._extractor = None

        self._async = bool(self.policy.async_write)
        self._queue: queue.Queue[_MemoryJob] = queue.Queue(maxsize=256)
        self._accepting = True
        self._idle = threading.Condition(self._lock)
        self._pending = 0
        self._retrieval_save_dirty = False
        self._retrieval_save_timer: threading.Timer | None = None
        self._thread: threading.Thread | None = None
        if self._async:
            self._thread = threading.Thread(
                target=self._worker_loop, name="memory-worker", daemon=True
            )
            self._thread.start()

    # ---- 写入 ----------------------------------------------------------------
    def submit_speech_memory(
        self,
        user_id: str,
        text: str,
        timestamp: int,
        *,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> bool:
        """提交一次记忆抽取；异步模式立即返回，不阻塞主链路。

        返回 True 仅表示「已受理（同步抽取成功 / 已入队）」，不代表一定写入记忆
        （是否写入由 LLM 抽取结果决定）。
        """

        if self._extractor is None:
            return False
        if not self._should_attempt_extract(text):
            return False

        recent = list(recent_messages or [])
        if not self._async:
            self._extract_and_store(str(user_id), str(text), int(timestamp), recent)
            return True

        with self._lock:
            if not self._accepting:
                return False
            try:
                self._queue.put_nowait((str(user_id), str(text), int(timestamp), recent))
            except queue.Full:
                return False
            self._pending += 1
            return True

    # ---- 读取 ----------------------------------------------------------------
    def retrieve_user_context(
        self,
        user_id: str,
        *,
        query: str = "",
        context_type: str = "speech",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """按 context_type 做轻量相关性检索，返回结构化记忆。

        返回::

            {
              "by_type": {group_key: [{content, confidence, tags, evidence}, ...]},
              "top": [{type, content, confidence, tags}, ...],   # 扁平 top_k
            }
        """

        cap = self.policy.retrieve_top_k if top_k is None else int(top_k)
        now = int(time.time())
        with self._lock:
            items = list(self._store.get(str(user_id), []))
            if not items:
                return {"by_type": {}, "top": []}

            scored = sorted(
                items,
                key=lambda it: self._score(it, query=query, context_type=context_type, now=now),
                reverse=True,
            )
            top = self._diversify(scored, cap=cap)
            dirty = False
            for item in top:
                if item.last_used_at != now:
                    item.last_used_at = now
                    dirty = True
            if dirty:
                self._schedule_retrieval_save()

            by_type: dict[str, list[dict[str, Any]]] = {}
            for item in top:
                group = GROUP_KEY_BY_TYPE.get(item.type, item.type)
                by_type.setdefault(group, []).append(
                    {
                        "content": item.content,
                        "confidence": round(item.confidence, 3),
                        "tags": list(item.tags),
                        "evidence": item.evidence,
                    }
                )
            flat = [
                {
                    "type": item.type,
                    "content": item.content,
                    "confidence": round(item.confidence, 3),
                    "tags": list(item.tags),
                }
                for item in top
            ]
        return {"by_type": by_type, "top": flat}

    def all_memories(self, user_id: str) -> list[dict[str, Any]]:
        """返回某用户全部记忆字典（供 CLI / 调试）。"""

        with self._lock:
            return [item.to_dict() for item in self._store.get(str(user_id), [])]

    def _schedule_retrieval_save(self) -> None:
        self._retrieval_save_dirty = True
        timer = self._retrieval_save_timer
        if timer is not None:
            timer.cancel()
        self._retrieval_save_timer = threading.Timer(
            _RETRIEVAL_SAVE_DEBOUNCE_SEC,
            self._flush_retrieval_save,
        )
        self._retrieval_save_timer.daemon = True
        self._retrieval_save_timer.start()

    def _flush_retrieval_save(self) -> None:
        with self._lock:
            if not self._retrieval_save_dirty:
                return
            self._save()
            self._retrieval_save_dirty = False

    def flush_pending_writes(self) -> None:
        """立即落盘检索侧 last_used_at 更新（shutdown 时调用）。"""
        timer = self._retrieval_save_timer
        if timer is not None:
            timer.cancel()
            self._retrieval_save_timer = None
        with self._lock:
            if self._retrieval_save_dirty:
                self._save()
                self._retrieval_save_dirty = False

    # ---- 生命周期 ------------------------------------------------------------
    def wait_for_idle(self, *, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._idle:
            while self._pending:
                if deadline is None:
                    self._idle.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(timeout=remaining)
            return True

    def shutdown(self, timeout: float = 5.0) -> None:
        self.flush_pending_writes()
        with self._lock:
            self._accepting = False
        if self._thread is not None:
            self.wait_for_idle(timeout=timeout)

    # ---- 内部：后台抽取 ------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            try:
                user_id, text, timestamp, recent = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._lock:
                    if not self._accepting and self._pending == 0:
                        return
                continue
            try:
                self._extract_and_store(user_id, text, timestamp, recent)
            except Exception:  # noqa: BLE001 - memory LLM 失败绝不影响主链路
                pass
            finally:
                with self._idle:
                    self._pending = max(0, self._pending - 1)
                    self._idle.notify_all()

    def _extract_and_store(
        self,
        user_id: str,
        text: str,
        timestamp: int,
        recent_messages: list[dict[str, Any]],
    ) -> None:
        if self._extractor is None:
            return
        items = self._extractor.extract(
            user_id=user_id,
            text=text,
            timestamp=timestamp,
            recent_messages=recent_messages,
        )
        kept = [item for item in items if item.confidence >= self.policy.min_keep_confidence]
        if not kept:
            return
        with self._lock:
            for item in kept:
                self._merge_one(item)
            self._save()

    def _merge_one(self, new_item: MemoryItem) -> None:
        """同类同义记忆合并/更新；否则追加；超容量按最旧淘汰。"""

        bucket = self._store.setdefault(new_item.user_id, [])
        existing = self._find_similar(bucket, new_item)
        if existing is not None:
            existing.confidence = max(existing.confidence, new_item.confidence)
            existing.evidence = new_item.evidence or existing.evidence
            existing.content = new_item.content or existing.content
            existing.tags = _merge_tags(existing.tags, new_item.tags)
            existing.updated_at = new_item.updated_at
            return

        bucket.append(new_item)
        if len(bucket) > self.policy.max_memories_per_user:
            bucket.sort(key=lambda it: (it.last_used_at or 0, it.updated_at))
            del bucket[: len(bucket) - self.policy.max_memories_per_user]

    @staticmethod
    def _find_similar(bucket: list[MemoryItem], new_item: MemoryItem) -> MemoryItem | None:
        new_norm = _normalize_text(new_item.content)
        new_tags = set(new_item.tags)
        for item in bucket:
            if item.type != new_item.type:
                continue
            if _normalize_text(item.content) == new_norm:
                return item
            # 同类型且标签高度重合也视为同义。
            if new_tags and set(item.tags) and new_tags == set(item.tags):
                return item
            if new_tags and set(item.tags):
                overlap = len(new_tags & set(item.tags))
                union = len(new_tags | set(item.tags))
                if union and overlap / union >= 0.6:
                    return item
        return None

    # ---- 内部：相关性打分 ----------------------------------------------------
    def _score(self, item: MemoryItem, *, query: str, context_type: str, now: int) -> float:
        weights = self.policy.type_weights.get(context_type, {})
        type_weight = weights.get(item.type, self.policy.default_type_weight)

        query_lower = str(query or "").lower()
        query_tokens = _tokens(query_lower)

        tag_hits = sum(1 for tag in item.tags if tag and tag in query_lower)
        content_hits = len(query_tokens & _tokens(item.content.lower())) if query_tokens else 0

        # 轮换惩罚：刚被用过的记忆短期内降权，让回复结合的记忆更丰富多元，
        # 避免「同一条（如讲笑话）记忆每次都被检索出来」的 rich-get-richer 循环。
        rotation_penalty = 0.0
        if item.last_used_at:
            age = max(0, now - int(item.last_used_at))
            if age < _ROTATION_WINDOW_SEC:
                rotation_penalty = _ROTATION_PENALTY

        return (
            type_weight
            + tag_hits * 2.0
            + content_hits * 1.5
            + item.confidence
            - rotation_penalty
        )

    @staticmethod
    def _diversify(scored: list[MemoryItem], *, cap: int) -> list[MemoryItem]:
        """在按分数排序的基础上做类型多样化：先每种 type 取一条，再按分数补齐。

        这样 top_k 会尽量横跨 care_strategy / hobby / preference / habit 等不同维度，
        而不是被同一类型（如多条 hobby 笑话）占满，让 LLM 能结合更丰富的记忆。
        """

        if cap <= 0:
            return []
        picked: list[MemoryItem] = []
        picked_ids: set[int] = set()
        seen_types: set[str] = set()
        # 第一轮：每种 type 只取分数最高的一条。
        for item in scored:
            if len(picked) >= cap:
                break
            if item.type in seen_types:
                continue
            seen_types.add(item.type)
            picked.append(item)
            picked_ids.add(id(item))
        # 第二轮：剩余名额按分数补齐（同一 type 可再补）。
        for item in scored:
            if len(picked) >= cap:
                break
            if id(item) not in picked_ids:
                picked.append(item)
                picked_ids.add(id(item))
        return picked[:cap]

    # ---- 内部：预过滤 --------------------------------------------------------
    def _should_attempt_extract(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if len(normalized) < self.policy.min_extract_chars:
            return False
        if normalized.lower() in self.policy.trivial_texts:
            return False
        return True

    # ---- 内部：持久化 --------------------------------------------------------
    def _load(self) -> dict[str, list[MemoryItem]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        store: dict[str, list[MemoryItem]] = {}
        for user_id, items in data.items():
            if not isinstance(items, list):
                continue
            parsed: list[MemoryItem] = []
            for raw in items:
                if isinstance(raw, dict) and raw.get("content"):
                    parsed.append(MemoryItem.from_dict(raw))
            store[str(user_id)] = parsed
        return store

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            user_id: [item.to_dict() for item in items]
            for user_id, items in self._store.items()
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def _merge_tags(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    for tag in incoming:
        if tag not in merged:
            merged.append(tag)
    return merged


def _tokens(text: str) -> set[str]:
    lower = str(text or "").lower()
    words = re.findall(r"[a-z0-9]+", lower)
    cjk = re.findall(r"[\u4e00-\u9fff]", lower)
    return set(words) | set(cjk)
