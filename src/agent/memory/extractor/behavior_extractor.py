from __future__ import annotations

"""从 Event/Action 中提取可长期统计的行为信号。

BehaviorExtractor 只做“标准化信号提取”，不累计统计、不生成 insight。
这样事件字段变化时只需要改这里，BehaviorUpdater 可以保持单一职责。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agent.action import Action
from src.agent.event import Event
from src.agent.state import AgentState


@dataclass
class BehaviorSignal:
    """供 BehaviorUpdater 消费的标准化行为信号。"""

    type: str
    timestamp: int
    payload: dict[str, Any] = field(default_factory=dict)


class BehaviorExtractor:
    """把项目内的 Event/Action 转成长期行为统计信号。"""

    def extract_event(self, event: Event, state: AgentState) -> list[BehaviorSignal]:
        """从单个 Event 中提取长期行为信号。

        这里不保存原始聊天内容，只提取和行为建模相关的计数/时段/接受拒绝信号。
        """
        if event.type == "focus_start_requested":
            # Reducer 会忽略“已在专注中”的重复开始请求；长期统计也只记录真正启动的会话。
            if not state.focus.active or state.focus.start_ts != event.timestamp:
                return []
            return [
                BehaviorSignal(
                    type="focus_started",
                    timestamp=event.timestamp,
                    payload={"hour": _hour_of_timestamp(event.timestamp)},
                )
            ]

        if event.type in {"focus_stop_requested", "timer_finished"}:
            duration = _latest_focus_duration(state)
            if duration is None:
                return []
            return [
                BehaviorSignal(
                    type="focus_completed",
                    timestamp=event.timestamp,
                    payload={"duration_sec": duration},
                )
            ]

        if event.type == "user_fatigue_updated":
            fatigue_level = str(event.payload.get("fatigue_level", state.user.fatigue_level))
            if fatigue_level not in {"moderate", "high"}:
                return []
            return [
                BehaviorSignal(
                    type="fatigue_detected",
                    timestamp=event.timestamp,
                    payload={"focus_elapsed_sec": state.focus.elapsed_sec},
                )
            ]

        if event.type == "user_emotion_updated":
            emotion = str(event.payload.get("emotion", state.user.emotion))
            if emotion not in {"tired", "stressed"}:
                return []
            return [
                BehaviorSignal(
                    type="fatigue_detected",
                    timestamp=event.timestamp,
                    payload={"focus_elapsed_sec": state.focus.elapsed_sec},
                )
            ]

        if event.type == "user_attention_updated":
            attention = str(event.payload.get("attention", state.user.attention))
            if attention != "distracted":
                return []
            return [BehaviorSignal(type="distraction_detected", timestamp=event.timestamp)]

        if event.type == "break_suggestion_accepted":
            return [
                BehaviorSignal(
                    type="break_suggestion_accepted",
                    timestamp=event.timestamp,
                    payload={"content_type": event.payload.get("content_type")},
                )
            ]

        if event.type == "break_suggestion_rejected":
            return [
                BehaviorSignal(
                    type="break_suggestion_rejected",
                    timestamp=event.timestamp,
                    payload={"content_type": event.payload.get("content_type")},
                )
            ]

        return []

    def extract_action(self, action: Action, timestamp: int) -> list[BehaviorSignal]:
        """从 Action 中提取长期行为信号。

        这里只统计 Agent 是否给过休息建议，不记录具体对话历史。
        """
        if action.payload.get("kind") != "notification":
            return []
        reason = str(action.payload.get("reason", ""))
        if reason not in {"rest_reminder", "fatigue_warning"}:
            return []

        text = str(action.payload.get("text", ""))
        content_type = _infer_break_content_type(text)
        return [
            BehaviorSignal(
                type="break_suggestion_shown",
                timestamp=timestamp,
                payload={"content_type": content_type},
            )
        ]


def _hour_of_timestamp(timestamp: int) -> int:
    """按本地时间提取小时，便于后续学习用户真实作息倾向。"""
    return int(datetime.fromtimestamp(timestamp).hour)


def _latest_focus_duration(state: AgentState) -> int | None:
    """从归约后的状态中读取最近一次已归档专注时长。"""
    if not state.memory.focus_sessions:
        return None
    latest = state.memory.focus_sessions[-1]
    value = latest.get("actual_duration_sec")
    if value is None:
        return None
    return int(value)


def _infer_break_content_type(text: str) -> str | None:
    """从提醒文案里粗略识别休息内容类型，第一版只区分音乐。"""
    if "音乐" in text or "轻音乐" in text or "古风" in text:
        return "音乐"
    return None
