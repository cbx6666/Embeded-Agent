"""
AgentCore 主调度模块。

本模块负责处理单个标准 Event：先通过 reducer 更新运行时 AgentState，再构建
ProfileSnapshot，调用 LLM-centered DecisionPipeline 生成 Intent 和 Action，
最后通过 DeviceAdapter 执行动作并记录短期 trace/memory。

上游输入是 adapters、CLI 或 runtime loop 产生的 Event；下游输出是 Action 和
ActionResult。本模块不理解用户语义、不手写业务规则、不直接操作硬件细节，也
不让 LLM 直接修改状态或长期记忆。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.intent_model import AgentIntent
from src.agent.event import Event
from src.agent.memory.memory_pipeline import MemoryPipeline
from src.agent.memory.memory_store import MemoryStore
from src.agent.reducer import reduce_state
from src.agent.runtime.action_result import ActionResult
from src.agent.runtime.device_adapter import DeviceAdapter
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import UserProfileService
from src.storage.json_store import JsonStore
from src.storage.profile_store import ProfileStore


class AgentCore:
    """LLM-centered Agent 的单事件调度器。

    输入 Event，输出动作及执行结果。它只串联 reducer、memory、decision 和
    device boundary，不把语义理解逻辑写回 Core。
    """

    def __init__(
        self,
        output: ConsoleOutput,
        timer_service: TimerService,
        memory_service: MemoryService,
        llm_service: LLMService,
        store: JsonStore,
        profile_service: UserProfileService | None = None,
        memory_pipeline: MemoryPipeline | None = None,
        decision_pipeline: DecisionPipeline | None = None,
    ) -> None:
        self.output = output
        self.timer_service = timer_service
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.store = store
        self.profile_service = profile_service or UserProfileService(
            ProfileStore(_default_profile_store_path(store))
        )
        self.memory_pipeline = memory_pipeline or MemoryPipeline(
            MemoryStore(_default_memory_store_path(store))
        )
        self.decision_pipeline = decision_pipeline or DecisionPipeline(
            profile_snapshot_builder=self.memory_pipeline.profile_snapshot_builder
        )
        self.state = AgentState.from_dict(self.store.load_state_dict())
        self.state.current_user_id = self.profile_service.ensure_user_id(self.state.current_user_id)
        self.device_adapter = DeviceAdapter(
            output=self.output,
            timer_service=self.timer_service,
            timer_callback=self._on_timer_tick,
        )
        self.last_intents: list[AgentIntent] = []
        self.last_decision_result: DecisionResult | None = None
        self.last_action_results: list[ActionResult] = []
        self._lock = threading.RLock()

    def handle_event(self, event: Event) -> tuple[list[Action], list[ActionResult]]:
        """处理一条标准事件并返回动作与执行结果。

        这是 Agent 主链路唯一运行入口。任何 LLM 输出都会经过 validator、
        guard 和 action realizer 后才可能进入设备层。
        """

        with self._lock:
            previous_state = AgentState.from_dict(self.state.to_dict())
            self.state = reduce_state(self.state, event)

            if _should_touch_profile(event):
                self.profile_service.touch_user(self.state.current_user_id, timestamp=event.timestamp)

            self.memory_service.record_event(self.state, event)
            self._record_user_message_if_needed(event)
            self.memory_pipeline.process_event(
                self.state.current_user_id,
                event,
                self.state,
                self.llm_service,
            )
            profile_snapshot = self.memory_pipeline.build_profile_snapshot(
                self.state.current_user_id,
                self.state,
                event,
                self.profile_service,
            )

            decision_result = self.decision_pipeline.decide(
                previous_state=previous_state,
                current_state=self.state,
                event=event,
                llm_service=self.llm_service,
                profile_service=self.profile_service,
                profile_snapshot=profile_snapshot,
            )
            actions = decision_result.actions
            results = self._execute_actions(actions, event.timestamp)
            self.memory_pipeline.process_actions(
                self.state.current_user_id,
                actions,
                event.timestamp,
                source_event=event,
                state=self.state,
                llm_service=self.llm_service,
            )

            self.last_intents = decision_result.intents
            self.last_decision_result = decision_result
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

    def render_profile(self) -> str:
        with self._lock:
            return self.profile_service.render_profile(self.state.current_user_id)

    def render_users(self) -> str:
        with self._lock:
            return self.profile_service.render_users(current_user_id=self.state.current_user_id)

    def switch_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        timestamp: int | None = None,
    ) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self.profile_service.switch_user(user_id, display_name=display_name, timestamp=ts)
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self.profile_service.render_switch_result(user_id)

    def set_user_preference(
        self,
        key: str,
        value: object,
        *,
        timestamp: int | None = None,
    ) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self.profile_service.update_preference(
                self.state.current_user_id,
                key,
                value,
                timestamp=ts,
            )
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self.profile_service.render_preference_update_result(user_id, key)

    def set_user_info(
        self,
        key: str,
        value: object,
        *,
        timestamp: int | None = None,
    ) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self.profile_service.update_info(
                self.state.current_user_id,
                key,
                value,
                timestamp=ts,
            )
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self.profile_service.render_info_update_result(user_id, key)

    def shutdown(self) -> None:
        with self._lock:
            self.timer_service.stop()
            self.store.save_state(self.state)

    def _record_user_message_if_needed(self, event: Event) -> None:
        if event.type not in {"user_text_input", "speech_recognized"}:
            return
        text = str(event.payload.get("text", "")).strip()
        if text:
            self.memory_service.record_message(
                self.state,
                role="user",
                text=text,
                timestamp=event.timestamp,
            )

    def _execute_actions(self, actions: list[Action], action_ts: int) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in actions:
            result = self.device_adapter.execute(action, action_ts)
            results.append(result)
            if result.success:
                self._after_successful_action(action, action_ts)
        return results

    def _after_successful_action(self, action: Action, action_ts: int) -> None:
        self.memory_service.record_action(self.state, action.type, action.payload, action_ts)
        if action.type not in {
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
            return

        text = str(action.payload.get("text", "")).strip()
        if text and action.type in {"speak", "display"}:
            role = "agent" if action.type == "speak" else "display"
            self.memory_service.record_message(self.state, role=role, text=text, timestamp=action_ts)
        self.state.interaction.last_agent_response_time = action_ts
        if action.type in {"speak", "display"}:
            self.state.interaction.dialogue_state = "idle"
        self._mark_cooldown_if_needed(action, action_ts)

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
    profile_store_path: str | Path = "data/user_profiles.json",
    memory_store_path: str | Path = "data/memory_store.json",
    timer_background: bool = True,
    output: ConsoleOutput | None = None,
) -> AgentCore:
    profile_service = UserProfileService(ProfileStore(profile_store_path))
    return AgentCore(
        output=output or ConsoleOutput(),
        timer_service=TimerService(background=timer_background),
        memory_service=MemoryService(),
        llm_service=LLMService(),
        store=JsonStore(store_path),
        profile_service=profile_service,
        memory_pipeline=MemoryPipeline(MemoryStore(memory_store_path)),
    )


def _default_profile_store_path(store: JsonStore) -> Path:
    return store.path.with_name("user_profiles.json")


def _default_memory_store_path(store: JsonStore) -> Path:
    return store.path.with_name("memory_store.json")


def _should_touch_profile(event: Event) -> bool:
    return event.type in {
        "user_text_input",
        "speech_recognized",
        "focus_start_requested",
        "focus_stop_requested",
        "user_switched",
        "user_presence_updated",
        "user_attention_updated",
        "user_emotion_updated",
        "user_fatigue_updated",
    }
