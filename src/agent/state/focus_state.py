from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FocusState:
    """专注状态：保存本轮专注的计时与来源。"""

    active: bool = False
    start_ts: int | None = None
    target_duration_sec: int | None = None
    elapsed_sec: int = 0
    remaining_sec: int | None = None
    triggered_by: str | None = None  # 例如 "user" / "auto"
    last_focus_end_ts: int | None = None
