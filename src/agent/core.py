from __future__ import annotations

"""Agent core dispatcher."""

import json
import threading
import time
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.action_result import ActionResult
from src.agent.event import Event
from src.agent.intent import AgentIntent
from src.agent.policy import decide_actions_with_intents
from src.agent.reducer import reduce_state
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore


class AgentCore:
    """Single-event agent core."""

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
        self.last_intents: list[AgentIntent] = []
        self.last_action_results: list[ActionResult] = []
        self._lock = threading.RLock()

    def handle_event(self, event: Event) -> list[Action]:
        """Process a single event and return actions."""
        actions, _ = self.handle_event_with_results(event)
        return actions

    def handle_event_with_results(
        self,
        event: Event,
    ) -> tuple[list[Action], list[ActionResult]]:
        """Process a single event and return actions with action results."""
        with self._lock:
            # 先拷贝一份事件发生前的状态，方便 planner 判断“状态如何变化了”，
            # 而不只是看到 reducer 更新后的结果状态。
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

            # 当前策略层分两步：
            # planner 先产出意图，再由 realizer 把意图落成具体动作。
            intents, actions = decide_actions_with_intents(
                previous_state=previous_state,
                current_state=self.state,
                event=event,
                llm_service=self.llm_service,
            )
            # 等状态和记忆都更新完成后，再执行动作，避免动作阶段读到旧上下文。
            results = self._execute_actions(actions, event.timestamp)

            self.last_intents = intents
            self.last_action_results = results
            self.memory_service.trim(self.state)
            self.store.save_state(self.state)
            return actions, results

    def render_state(self) -> str:
        with self._lock:
            return json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2)

    def render_history(self) -> str:
        with self._lock:
            history = {
                "recent_events": self.state.memory.recent_events,
                "recent_messages": self.state.memory.recent_messages,
                "reminder_records": self.state.memory.reminder_records,
                "attention_records": self.state.memory.attention_records,
                "environment_records": self.state.memory.environment_records,
                "focus_sessions": self.state.memory.focus_sessions,
                "focus_session_count": self.state.memory.focus_session_count,
                "focus_total_duration_sec": self.state.memory.focus_total_duration_sec,
                "distraction_event_count": self.state.memory.distraction_event_count,
                "state_change_counts": self.state.memory.state_change_counts,
                "emotion_samples": self.state.memory.emotion_samples,
                "emotion_summaries": self.state.memory.emotion_summaries,
            }
            return json.dumps(history, ensure_ascii=False, indent=2)

    def shutdown(self) -> None:
        with self._lock:
            self.timer_service.stop()
            self.store.save_state(self.state)

    def _execute_actions(self, actions: list[Action], action_ts: int) -> list[ActionResult]:
        # 每个动作都独立执行，并且都要产出一个 ActionResult，
        # 这样外层闭环才能稳定地基于执行结果生成内部反馈事件。
        return [self._execute_action(action, action_ts) for action in actions]

    def _execute_action(self, action: Action, action_ts: int) -> ActionResult:
        try:
            if action.type == "start_timer":
                duration_sec = int(action.payload.get("duration_sec", 0))
                self.timer_service.start(duration_sec, self._on_timer_tick)
                return ActionResult(
                    action_type=action.type,
                    success=True,
                    timestamp=action_ts,
                    payload=dict(action.payload),
                )

            if action.type == "stop_timer":
                self.timer_service.stop()
                return ActionResult(
                    action_type=action.type,
                    success=True,
                    timestamp=action_ts,
                    payload=dict(action.payload),
                )

            if action.type == "none":
                return ActionResult(
                    action_type=action.type,
                    success=True,
                    timestamp=action_ts,
                    payload=dict(action.payload),
                )

            self.output.execute(action)
            self.memory_service.record_action(self.state, action.type, action.payload, action_ts)

            if action.type in {
                "speak",
                "display",
                "render_pet_expression",
                "set_light_state",
                "start_voice_capture",
                "stop_voice_capture",
                "set_tts_voice",
                "set_tts_volume",
                "set_tts_speed",
            }:
                # 只有面向用户可感知的输出动作，才会写回短期消息记忆
                # 和最近一次 agent 响应时间等交互状态。
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

            return ActionResult(
                action_type=action.type,
                success=True,
                timestamp=action_ts,
                payload=dict(action.payload),
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return ActionResult(
                action_type=action.type,
                success=False,
                timestamp=action_ts,
                reason=str(exc),
                payload=dict(action.payload),
            )

    def _mark_cooldown_if_needed(self, action: Action, action_ts: int) -> None:
        if action.payload.get("kind") != "notification":
            return
        reason = action.payload.get("reason")
        if reason:
            self.state.cooldown.reminder_last_ts[str(reason)] = action_ts

    def _on_timer_tick(self, remaining_sec: int) -> None:
        # timer 回调不会直接改状态，而是重新包装成标准 Event，
        # 这样定时行为也能走和外部输入完全一致的 reducer / planner / realizer 链路。
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
