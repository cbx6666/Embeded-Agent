from __future__ import annotations

"""姿势识别适配器（适配层示例实现）。

职责：把模型/检测输出转成标准 Event 并通过 AgentCore.handle_event 发送给内核。
实现包含简单的置信度阈值与防抖（去抖动）示例，便于联调。
"""

import time
from typing import Any

from src.agent.event.factories import make_posture_event


class PostureAdapter:
    """简单姿势适配器：负责阈值、去抖并上报 Event。"""

    def __init__(
        self,
        core: Any,
        min_confidence: float = 0.6,
        debounce_seconds: float = 2.0,
        summary_threshold_seconds: float = 120.0,
    ) -> None:
        """core: AgentCore 实例或任何暴露 handle_event(event) 的对象。"""
        self.core = core
        self.min_confidence = float(min_confidence)
        self.debounce_seconds = float(debounce_seconds)
        self._last_posture: str | None = None
        self._last_ts: float | None = None
        # accumulate seconds of 'bad' posture per posture label
        self.summary_threshold_seconds = float(summary_threshold_seconds)
        self._bad_posture_accum: dict[str, float] = {}
        self._last_summary_ts: dict[str, float] = {}

    def publish_posture(
        self,
        posture: str,
        confidence: float | None = None,
        frame_id: int | str | None = None,
        bbox: dict[str, Any] | None = None,
        source: str = "camera_v1",
        timestamp: int | None = None,
    ) -> bool:
        """尝试上报姿势事件。

        返回 True 表示已上报；False 表示因置信度/防抖被抑制。
        """
        now = time.time()
        if confidence is None:
            conf = 1.0
        else:
            conf = float(confidence)

        # 置信度门限
        if conf < self.min_confidence:
            return False

        # 防抖：只有当姿势变化或超过 debounce_seconds 时上报
        if self._last_posture == posture:
            if self._last_ts is not None and (now - self._last_ts) < self.debounce_seconds:
                return False

        # 构造并发送事件（包含可选字段）
        event = make_posture_event(
            posture=posture,
            confidence=conf,
            frame_id=frame_id,
            bbox=bbox,
            duration_sec=None,
            person_id=None,
            keypoints_summary=None,
            severity=None,
            source=source,
            timestamp=timestamp,
        )
        try:
            # 期望 core 有 handle_event 方法
            self.core.handle_event(event)
        except Exception:
            # 不抛异常给上层；记录为未发送
            return False

        # 更新 debounce 状态
        # 计算累积不良姿势时间（仅对被视为“不良”的姿势统计）
        last_ts = self._last_ts or now
        delta = max(0.0, now - last_ts)
        # 定义哪些 posture 被视为不良以便汇总（可调整）
        bad_postures = {"slouch", "lying"}
        if posture in bad_postures:
            prev = self._bad_posture_accum.get(posture, 0.0)
            self._bad_posture_accum[posture] = prev + delta

        # 如果累积超过 summary 阈值，发送 summary 事件并重置累积
        accum = self._bad_posture_accum.get(posture, 0.0)
        last_sum = self._last_summary_ts.get(posture, 0.0)
        # 防止频繁 summary：至少间隔 summary_threshold_seconds
        if accum >= self.summary_threshold_seconds and (now - last_sum) >= self.summary_threshold_seconds:
            try:
                from src.agent.event.factories import make_posture_summary_event

                summary_event = make_posture_summary_event(
                    posture=posture,
                    accumulated_sec=accum,
                    confidence=conf,
                    source=source,
                )
                self.core.handle_event(summary_event)
                # reset accumulator for this posture
                self._bad_posture_accum[posture] = 0.0
                self._last_summary_ts[posture] = now
            except Exception:
                pass

        self._last_posture = posture
        self._last_ts = now
        return True
