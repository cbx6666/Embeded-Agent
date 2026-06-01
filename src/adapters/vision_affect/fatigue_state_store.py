from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path


def _day_key(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


class FatigueStateStore:
    """疲劳状态落库：秒级明细 + 日/周汇总。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        second_state_retention_days: int = 14,
        cleanup_interval_sec: int = 300,
    ) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.RLock()
        self._second_state_retention_days = max(0, int(second_state_retention_days))
        self._cleanup_interval_sec = max(1, int(cleanup_interval_sec))
        self._last_cleanup_ts: int = 0
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record_second_state(
        self,
        *,
        timestamp: int,
        fatigue_level: str,
        confidence: float | None,
        source: str | None,
    ) -> None:
        with self._lock:
            self._record_second_state_locked(
                timestamp=timestamp,
                fatigue_level=fatigue_level,
                confidence=confidence,
                source=source,
            )

    def _record_second_state_locked(
        self,
        *,
        timestamp: int,
        fatigue_level: str,
        confidence: float | None,
        source: str | None,
    ) -> None:
        ts = int(timestamp)
        now = int(time.time())
        day = _day_key(ts)
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute(
                "SELECT fatigue_level FROM fatigue_second_states WHERE ts = ?",
                (ts,),
            )
            previous_row = cur.fetchone()
            previous_state = str(previous_row[0]) if previous_row else None

            cur.execute(
                """
                INSERT INTO fatigue_second_states(ts, fatigue_level, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ts) DO UPDATE SET
                    fatigue_level = excluded.fatigue_level,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (ts, fatigue_level, confidence, source, now),
            )

            if previous_state is None:
                self._apply_delta(cur, "fatigue_daily_summary", "fatigue_daily_totals", day, fatigue_level, 1, now)
            elif previous_state != fatigue_level:
                self._apply_delta(cur, "fatigue_daily_summary", "fatigue_daily_totals", day, previous_state, -1, now)
                self._apply_delta(cur, "fatigue_daily_summary", "fatigue_daily_totals", day, fatigue_level, 1, now)

            if self._should_cleanup(now):
                self._cleanup_expired_second_states(cur, now)
                self._last_cleanup_ts = now
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def query_seconds_by_state(self, *, start_ts: int, end_ts: int) -> dict[str, int]:
        with self._lock:
            left, right = sorted((int(start_ts), int(end_ts)))
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT fatigue_level, COUNT(*) AS seconds
                FROM fatigue_second_states
                WHERE ts BETWEEN ? AND ?
                GROUP BY fatigue_level
                """,
                (left, right),
            )
            return {str(state): int(seconds) for state, seconds in cur.fetchall()}

    def query_daily_summary(self, *, day: str) -> dict[str, dict[str, float | int]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT fatigue_level, seconds, ratio
                FROM fatigue_daily_summary
                WHERE period_key = ?
                ORDER BY seconds DESC
                """,
                (day,),
            )
            return {
                str(state): {"seconds": int(seconds), "ratio": float(ratio)}
                for state, seconds, ratio in cur.fetchall()
            }

    def _apply_delta(
        self,
        cur: sqlite3.Cursor,
        summary_table: str,
        totals_table: str,
        period_key: str,
        state: str,
        delta: int,
        now: int,
    ) -> None:
        if delta == 0:
            return
        cur.execute(
            f"""
            INSERT INTO {summary_table}(period_key, fatigue_level, seconds, ratio, updated_at)
            VALUES (?, ?, ?, 0.0, ?)
            ON CONFLICT(period_key, fatigue_level) DO UPDATE SET
                seconds = {summary_table}.seconds + excluded.seconds,
                updated_at = excluded.updated_at
            """,
            (period_key, state, delta, now),
        )
        cur.execute(
            f"DELETE FROM {summary_table} WHERE period_key = ? AND fatigue_level = ? AND seconds <= 0",
            (period_key, state),
        )
        cur.execute(
            f"""
            INSERT INTO {totals_table}(period_key, total_seconds, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(period_key) DO UPDATE SET
                total_seconds = {totals_table}.total_seconds + excluded.total_seconds,
                updated_at = excluded.updated_at
            """,
            (period_key, delta, now),
        )
        cur.execute(
            f"DELETE FROM {totals_table} WHERE period_key = ? AND total_seconds <= 0",
            (period_key,),
        )
        cur.execute(
            f"""
            UPDATE {summary_table}
            SET ratio = CASE
                WHEN COALESCE((SELECT total_seconds FROM {totals_table} WHERE period_key = ?), 0) <= 0 THEN 0.0
                ELSE CAST(seconds AS REAL) /
                     CAST((SELECT total_seconds FROM {totals_table} WHERE period_key = ?) AS REAL)
            END,
            updated_at = ?
            WHERE period_key = ?
            """,
            (period_key, period_key, now, period_key),
        )

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fatigue_second_states (
                ts INTEGER PRIMARY KEY,
                fatigue_level TEXT NOT NULL,
                confidence REAL,
                source TEXT,
                updated_at INTEGER NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fatigue_second_state_level ON fatigue_second_states(fatigue_level)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fatigue_daily_summary (
                period_key TEXT NOT NULL,
                fatigue_level TEXT NOT NULL,
                seconds INTEGER NOT NULL,
                ratio REAL NOT NULL DEFAULT 0.0,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (period_key, fatigue_level)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fatigue_daily_totals (
                period_key TEXT PRIMARY KEY,
                total_seconds INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def _should_cleanup(self, now: int) -> bool:
        return (now - self._last_cleanup_ts) >= self._cleanup_interval_sec

    def _cleanup_expired_second_states(self, cur: sqlite3.Cursor, now: int) -> None:
        if self._second_state_retention_days <= 0:
            return
        threshold_ts = now - self._second_state_retention_days * 86400
        cur.execute("DELETE FROM fatigue_second_states WHERE ts < ?", (threshold_ts,))
