"""视觉情绪与疲劳：检测逻辑仅在本包内；向上只发标准 Event。

- `VisionAffectConfig`：阈值与节奏
- `VisionAffectInputAdapter`：后台采集 + 投递事件
- `vision_dependencies_met()`：是否已安装 opencv + mediapipe
"""

from __future__ import annotations

from src.adapters.vision_affect.adapter import (
    EventEmitSink,
    VisionAffectInputAdapter,
    vision_dependencies_met,
)
from src.adapters.vision_affect.config import VisionAffectConfig
from src.adapters.vision_affect.pipeline import FatigueLevel

__all__ = [
    "EventEmitSink",
    "FatigueLevel",
    "VisionAffectConfig",
    "VisionAffectInputAdapter",
    "vision_dependencies_met",
]
