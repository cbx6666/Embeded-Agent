from __future__ import annotations

"""动作结果到内部事件的回流转换。"""

from src.agent.action import Action
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult
from src.agent.state import AgentState


def build_internal_events_from_results(
    state: AgentState,
    event: Event,
    actions: list[Action],
    results: list[ActionResult],
) -> list[Event]:
    """根据动作执行结果保守地生成内部事件。"""
    del state

    if not actions:
        return _failure_events(results)

    internal_events: list[Event] = []
    internal_events.extend(_failure_events(results))

    successful_types = {
        result.action_type
        for result in results
        if result.success
    }
    event_ts = _event_timestamp(event, results)

    if successful_types & {"speak", "display"}:
        internal_events.append(
            Event(
                type="system_triggered",
                timestamp=event_ts,
                payload={
                    "trigger": "agent_response_completed",
                    "source": "agent_action_result",
                    "source_event_type": event.type,
                    "action_count": len(actions),
                },
            )
        )

    if "start_timer" in successful_types:
        timer_payload = _first_success_payload(results, "start_timer")
        internal_events.append(
            Event(
                type="system_triggered",
                timestamp=event_ts,
                payload={
                    "trigger": "focus_timer_started",
                    "source": "agent_action_result",
                    "source_event_type": event.type,
                    "duration_sec": timer_payload.get("duration_sec"),
                },
            )
        )

    if "stop_timer" in successful_types:
        internal_events.append(
            Event(
                type="system_triggered",
                timestamp=event_ts,
                payload={
                    "trigger": "focus_timer_stopped",
                    "source": "agent_action_result",
                    "source_event_type": event.type,
                },
            )
        )

    return internal_events


def _failure_events(results: list[ActionResult]) -> list[Event]:
    """把失败的动作结果转换为 action_failed 内部事件。"""
    failure_events: list[Event] = []
    for result in results:
        if result.success:
            continue
        failure_events.append(
            Event(
                type="system_triggered",
                timestamp=result.timestamp,
                payload={
                    "trigger": "action_failed",
                    "source": "agent_action_result",
                    "action_type": result.action_type,
                    "reason": result.reason,
                },
            )
        )
    return failure_events


def _event_timestamp(event: Event, results: list[ActionResult]) -> int:
    """优先使用最后一个结果的时间戳，否则回退到原事件时间。"""
    if results:
        return int(results[-1].timestamp)
    return int(event.timestamp)


def _first_success_payload(results: list[ActionResult], action_type: str) -> dict[str, object]:
    for result in results:
        if result.success and result.action_type == action_type:
            return dict(result.payload)
    return {}
