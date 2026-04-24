from __future__ import annotations

from src.agent.action import Action
from src.agent.action_result import ActionResult
from src.agent.event import Event
from src.agent.state import AgentState


def build_internal_events_from_results(
    state: AgentState,
    event: Event,
    actions: list[Action],
    results: list[ActionResult],
) -> list[Event]:
    del state

    if not actions or all(action.type == "none" for action in actions):
        return _failure_events(results)

    internal_events: list[Event] = []
    internal_events.extend(_failure_events(results))

    successful_types = {
        result.action_type
        for result in results
        if result.success and result.action_type != "none"
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
        internal_events.append(
            Event(
                type="system_triggered",
                timestamp=event_ts,
                payload={
                    "trigger": "focus_timer_started",
                    "source": "agent_action_result",
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
                },
            )
        )

    return internal_events


def _failure_events(results: list[ActionResult]) -> list[Event]:
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
    if results:
        return int(results[-1].timestamp)
    return int(event.timestamp)
