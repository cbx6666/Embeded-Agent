from __future__ import annotations

"""Agent 核心调度模块。"""

import json
import threading
import time
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.event import Event
from src.agent.policy import decide_actions
from src.agent.reducer import reduce_state
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore


class AgentCore:
    """系统核心调度器。"""

    def __init__(
        self,
        output: ConsoleOutput,
        timer_service: TimerService,
        memory_service: MemoryService,
        llm_service: LLMService,
        store: JsonStore,
    ) -> None:
        self.output = output
        self.timer_service = timer_service
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.store = store
        self.state = AgentState.from_dict(self.store.load_state_dict())
        self._lock = threading.RLock()

    def handle_event(self, event: Event) -> list[Action]:
        """处理单个标准事件。"""
        with self._lock:
            previous_state = AgentState.from_dict(self.state.to_dict())
            self.state = reduce_state(self.state, event)
            self.memory_service.record_event(self.state, event)
            if event.type in {"user_text_input", "speech_recognized"}:
                text = str(event.payload.get("text", "")).strip()
                if text:
                    self.memory_service.record_message(
                        self.state,
                        role="user",
                        text=text,
                        timestamp=event.timestamp,
                    )
            actions = decide_actions(previous_state, self.state, event, self.llm_service)
            self._execute_actions(actions, event.timestamp)
            self.memory_service.trim(self.state)
            self.store.save_state(self.state)
            return actions

    def render_state(self) -> str:
        with self._lock:
            return json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2)

    def render_history(self) -> str:
        with self._lock:
            history = {
                "recent_events": self.state.memory.recent_events,
                "recent_messages": self.state.memory.recent_messages,
                "focus_sessions": self.state.memory.focus_sessions,
                "emotion_samples": self.state.memory.emotion_samples,
                "emotion_summaries": self.state.memory.emotion_summaries,
            }
            return json.dumps(history, ensure_ascii=False, indent=2)

    def shutdown(self) -> None:
        with self._lock:
            self.timer_service.stop()
            self.store.save_state(self.state)

    def _execute_actions(self, actions: list[Action], action_ts: int) -> None:
        for action in actions:
            if action.type == "start_timer":
                duration_sec = int(action.payload.get("duration_sec", 0))
                self.timer_service.start(duration_sec, self._on_timer_tick)
                continue

            if action.type == "stop_timer":
                self.timer_service.stop()
                continue

            if action.type in {"speak", "display", "render_pet_expression", "set_light_state"}:
                self.output.execute(action)
                text = str(action.payload.get("text", "")).strip()
                if text and action.type in {"speak", "display"}:
                    role = "agent" if action.type == "speak" else "display"
                    self.memory_service.record_message(
                        self.state,
                        role=role,
                        text=text,
                        timestamp=action_ts,
                    )
                self.state.interaction.last_agent_response_time = action_ts
                if action.type in {"speak", "display"}:
                    self.state.interaction.dialogue_state = "idle"
                self._mark_cooldown_if_needed(action, action_ts)
                continue

    def _mark_cooldown_if_needed(self, action: Action, action_ts: int) -> None:
        if action.payload.get("kind") != "notification":
            return
        reason = action.payload.get("reason")
        if reason:
            self.state.cooldown.reminder_last_ts[str(reason)] = action_ts

    def _on_timer_tick(self, remaining_sec: int) -> None:
        event_type = "timer_finished" if remaining_sec <= 0 else "timer_ticked"
        event = Event(
            type=event_type,
            timestamp=int(time.time()),
            payload={"remaining_sec": remaining_sec, "timer": "focus"},
        )
        self.handle_event(event)



def build_default_core(
    store_path: str | Path = "data/runtime_store.json",
    timer_background: bool = True,
    output: ConsoleOutput | None = None,
) -> AgentCore:
    return AgentCore(
        output=output or ConsoleOutput(),
        timer_service=TimerService(background=timer_background),
        memory_service=MemoryService(),
        llm_service=LLMService(),
        store=JsonStore(store_path),
    )
