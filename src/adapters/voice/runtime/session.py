from __future__ import annotations

"""语音会话数据结构。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VoiceSession:
    session_id: str
    start_time: float
    audio_path: Path | None = None
    is_active: bool = False
