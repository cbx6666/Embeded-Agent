from __future__ import annotations

from collections import deque

from src.agent.action import Action
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.internal_events import build_internal_events_from_results
from src.agent.trace import AgentDecisionTrace


class AgentLoop:
    def __init__(
        self,
        core: AgentCore,
        max_steps: int = 3,
    ) -> None:
        self.core = core
        self.max_steps = max(1, int(max_steps))
        self.recent_traces: list[AgentDecisionTrace] = []
        self._max_recent_traces = 50

    def run_once(self, event: Event) -> list[Action]:
        queue: deque[Event] = deque([event])
        all_actions: list[Action] = []
        step = 0

        while queue and step < self.max_steps:
            current_event = queue.popleft()
            step += 1

            actions, results = self.core.handle_event_with_results(current_event)
            all_actions.extend(actions)
            self._record_trace(current_event, step, actions, results)

            internal_events = build_internal_events_from_results(
                state=self.core.state,
                event=current_event,
                actions=actions,
                results=results,
            )
            queue.extend(internal_events)

        return all_actions

    def _record_trace(
        self,
        event: Event,
        loop_step: int,
        actions: list[Action],
        results,
    ) -> None:
        trace = AgentDecisionTrace(
            event_type=event.type,
            timestamp=event.timestamp,
            state_summary=_summarize_state(self.core.state),
            intents=[
                {
                    "type": intent.type,
                    "priority": intent.priority,
                    "reason": intent.reason,
                    "payload": dict(intent.payload),
                    "requires_llm": intent.requires_llm,
                }
                for intent in self.core.last_intents
            ],
            actions=[
                {
                    "type": action.type,
                    "payload": dict(action.payload),
                }
                for action in actions
            ],
            results=[
                {
                    "action_type": result.action_type,
                    "success": result.success,
                    "timestamp": result.timestamp,
                    "reason": result.reason,
                    "payload": dict(result.payload),
                }
                for result in results
            ],
            loop_step=loop_step,
        )
        self.recent_traces.append(trace)
        self.recent_traces = self.recent_traces[-self._max_recent_traces :]


def _summarize_state(state) -> dict[str, object]:
    return {
        "mode": state.interaction.mode,
        "dialogue_state": state.interaction.dialogue_state,
        "user_presence": state.user.presence,
        "user_attention": state.user.attention,
        "user_fatigue": state.user.fatigue_level,
        "focus_active": state.focus.active,
        "focus_elapsed_sec": state.focus.elapsed_sec,
        "focus_remaining_sec": state.focus.remaining_sec,
        "environment_light_level": state.environment.light_level,
        "environment_noise_level": state.environment.noise_level,
        "cooldown": dict(state.cooldown.reminder_last_ts),
    }
