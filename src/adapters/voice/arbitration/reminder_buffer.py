from __future__ import annotations

"""自主提醒缓冲：合并、过期、限长；会话结束后最多释放一条。"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.adapters.voice.arbitration.tts_job_policy import TTSJobSpec
from src.adapters.voice.runtime.logger import voice_log


@dataclass
class BufferedReminder:
    text: str
    source: str
    reason: str
    priority: int
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    coalesce_key: str | None = None
    spec_priority: int = 99


class ReminderBuffer:
    """成熟产品式 pending：同键合并、过期丢弃、限长、单次最多释放一条。"""

    def __init__(
        self,
        *,
        max_items: int = 8,
        default_expire_sec: float = 120.0,
        post_interaction_grace_sec: float = 2.0,
    ) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, BufferedReminder] = {}
        self._order: list[str] = []
        self._max_items = max_items
        self._default_expire_sec = default_expire_sec
        self._post_interaction_grace_sec = post_interaction_grace_sec
        self._last_user_interaction_end = 0.0

    def mark_user_interaction_ended(self, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        with self._lock:
            self._last_user_interaction_end = ts

    def in_post_interaction_grace(self, *, now: float | None = None) -> bool:
        ts = now if now is not None else time.time()
        with self._lock:
            if self._last_user_interaction_end <= 0:
                return False
            return (ts - self._last_user_interaction_end) < self._post_interaction_grace_sec

    def offer(
        self,
        *,
        text: str,
        source: str,
        reason: str,
        priority: int,
        payload: dict[str, Any],
        spec: TTSJobSpec,
        created_at: float | None = None,
    ) -> str:
        """入缓冲；返回 play | coalesced | dropped | buffer_full。"""
        now = created_at if created_at is not None else time.time()
        key = spec.coalesce_key or f"{source}:{reason}"
        expire = spec.expire_seconds or self._default_expire_sec

        with self._lock:
            self._purge_expired_locked(now=now, expire_default=expire)
            existing = self._items.get(key)
            if existing is not None:
                existing.text = text
                existing.source = source
                existing.reason = reason
                existing.priority = priority
                existing.payload = dict(payload)
                existing.created_at = now
                existing.spec_priority = spec.priority
                voice_log(f"提醒已合并（key={key}）：{text[:40]}")
                return "coalesced"

            if len(self._order) >= self._max_items:
                dropped_key = self._order.pop(0)
                self._items.pop(dropped_key, None)
                voice_log(f"提醒缓冲已满，丢弃最旧项（key={dropped_key}）")

            item = BufferedReminder(
                text=text,
                source=source,
                reason=reason,
                priority=priority,
                payload=dict(payload),
                created_at=now,
                coalesce_key=key,
                spec_priority=spec.priority,
            )
            self._items[key] = item
            self._order.append(key)
            voice_log(f"提醒已入缓冲（key={key}）：{text[:40]}（原因={reason}）")
            return "buffered"

    def pop_best(self, *, now: float | None = None) -> BufferedReminder | None:
        """取出当前最值得播的一条（优先级最高），其余保留。"""
        ts = now if now is not None else time.time()
        with self._lock:
            self._purge_expired_locked(now=ts, expire_default=self._default_expire_sec)
            if not self._order:
                return None
            best_key = min(
                self._order,
                key=lambda k: (self._items[k].spec_priority, self._items[k].created_at),
            )
            item = self._items.pop(best_key)
            self._order.remove(best_key)
            voice_log(f"从缓冲释放提醒（key={best_key}）：{item.text[:40]}")
            return item

    def clear_autonomous(self) -> int:
        with self._lock:
            keys = [k for k in self._order]
            removed = 0
            for key in keys:
                self._items.pop(key, None)
                removed += 1
            self._order.clear()
            if removed:
                voice_log(f"已清空提醒缓冲（{removed} 条）")
            return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._order)

    def _purge_expired_locked(self, *, now: float, expire_default: float) -> None:
        dead: list[str] = []
        for key in self._order:
            item = self._items.get(key)
            if item is None:
                dead.append(key)
                continue
            age = now - item.created_at
            limit = expire_default
            if age > limit:
                dead.append(key)
                voice_log(f"提醒已过期丢弃（key={key}，age={age:.0f}s）")
        for key in dead:
            self._items.pop(key, None)
            if key in self._order:
                self._order.remove(key)
