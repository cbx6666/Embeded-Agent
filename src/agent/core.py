from __future__ import annotations

"""AgentCore 主调度模块。

它是什么：
AgentCore 是单事件调度中枢，负责串联 Event -> RuntimeHistory ->
LongTermMemoryPipeline -> PersonalContextBuilder -> DecisionPipeline -> Action。

它不是什么：
它不理解用户语义，不直接读取 LongTermMemoryStore，不直接读取 UserProfileStore，不让 LLM
直接写 state/store/profile，也不把短期历史和长期记忆混成一个“memory”入口。

为什么存在：
Core 的职责是保持数据流清晰，把每一轮事件交给各语义边界处理。personalization 相关入口
在这里收敛为 RuntimeHistoryService、LongTermMemoryPipeline、PersonalContextBuilder 和
DecisionPipeline。

边界：
显式资料的权威来源是 UserProfile；行为偏好和模式的权威来源是 LongTermMemory；
最近对话和动作的权威来源是 RuntimeHistory；决策上下文只能来自 PersonalContextBuilder。
"""

import json
import threading
import time
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.context.personal_context_builder import PersonalContextBuilder
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.intent_model import AgentIntent
from src.agent.event import Event
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.reducer import reduce_state
from src.agent.runtime.action_result import ActionResult
from src.agent.runtime.device_adapter import DeviceAdapter
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import UserProfileService
from src.storage.json_store import JsonStore
from src.storage.long_term_memory_store import LongTermMemoryStore
from src.storage.user_profile_store import UserProfileStore


class AgentCore:
    """LLM-centered Agent 的单事件调度器。"""

    def __init__(
        self,
        output: ConsoleOutput,
        timer_service: TimerService,
        runtime_history_service: RuntimeHistoryService,
        llm_service: LLMService,
        store: JsonStore,
        long_term_memory_pipeline: LongTermMemoryPipeline | None = None,
        personal_context_builder: PersonalContextBuilder | None = None,
        decision_pipeline: DecisionPipeline | None = None,
    ) -> None:
        self.output = output
        self.timer_service = timer_service
        self.runtime_history_service = runtime_history_service
        self.llm_service = llm_service
        self.store = store

        long_term_store = (
            long_term_memory_pipeline.store
            if long_term_memory_pipeline is not None
            else LongTermMemoryStore(_default_long_term_memory_store_path(store))
        )
        self.long_term_memory_pipeline = long_term_memory_pipeline or LongTermMemoryPipeline(long_term_store)
        self.personal_context_builder = personal_context_builder or PersonalContextBuilder(
            long_term_memory_store=long_term_store,
            user_profile_service=UserProfileService(UserProfileStore(_default_profile_store_path(store))),
        )
        self.decision_pipeline = decision_pipeline or DecisionPipeline()

        self.state = AgentState.from_dict(self.store.load_state_dict())
        self.state.current_user_id = self._profile_service().ensure_user_id(self.state.current_user_id)

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
        """处理一条标准事件并返回动作与执行结果。"""

        with self._lock:
            previous_state = AgentState.from_dict(self.state.to_dict())
            self.state = reduce_state(self.state, event)

            if _should_touch_profile(event):
                self._profile_service().touch_user(self.state.current_user_id, timestamp=event.timestamp)

            self.runtime_history_service.record_event(self.state, event)
            self._record_user_message_if_needed(event)

            self.long_term_memory_pipeline.process_event(
                self.state.current_user_id,
                event,
                self.state,
                self.llm_service,
            )

            personal_context = self.personal_context_builder.build(
                user_id=self.state.current_user_id,
                state=self.state,
                event=event,
            )

            decision_result = self.decision_pipeline.decide(
                previous_state=previous_state,
                current_state=self.state,
                event=event,
                llm_service=self.llm_service,
                personal_context=personal_context,
            )
            actions = decision_result.actions
            results = self._execute_actions(actions, event.timestamp)

            self.long_term_memory_pipeline.process_actions(
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

            self.runtime_history_service.trim(self.state)
            self.store.save_state(self.state)
            return actions, results

    def render_state(self) -> str:
        with self._lock:
            return json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2)

    def render_history(self) -> str:
        with self._lock:
            return json.dumps(self.state.runtime_history.to_decision_dict(), ensure_ascii=False, indent=2)

    def render_profile(self) -> str:
        with self._lock:
            return self._profile_service().render_profile(self.state.current_user_id)

    def render_users(self) -> str:
        with self._lock:
            return self._profile_service().render_users(current_user_id=self.state.current_user_id)

    def switch_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        timestamp: int | None = None,
    ) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self._profile_service().switch_user(user_id, display_name=display_name, timestamp=ts)
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self._profile_service().render_switch_result(user_id)

    def set_user_preference(
        self,
        key: str,
        value: object,
        *,
        timestamp: int | None = None,
    ) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self._profile_service().update_preference(
                self.state.current_user_id,
                key,
                value,
                timestamp=ts,
            )
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self._profile_service().render_preference_update_result(user_id, key)

    def set_user_info(
        self,
        key: str,
        value: object,
        *,
        timestamp: int | None = None,
    ) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self._profile_service().update_info(
                self.state.current_user_id,
                key,
                value,
                timestamp=ts,
            )
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self._profile_service().render_info_update_result(user_id, key)

    def shutdown(self) -> None:
        with self._lock:
            self.timer_service.stop()
            self.store.save_state(self.state)

    def _record_user_message_if_needed(self, event: Event) -> None:
        if event.type not in {"user_text_input", "speech_recognized"}:
            return
        text = str(event.payload.get("text", "")).strip()
        if text:
            self.runtime_history_service.record_message(
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
        self.runtime_history_service.record_action(self.state, action, action_ts)

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
            self.runtime_history_service.record_message(self.state, role=role, text=text, timestamp=action_ts)

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

    def _profile_service(self) -> UserProfileService:
        service = self.personal_context_builder.user_profile_service
        if service is None:
            raise RuntimeError("PersonalContextBuilder must have UserProfileService for profile commands")
        return service


def build_default_core(
    store_path: str | Path = "data/runtime_store.json",
    profile_store_path: str | Path = "data/user_profiles.json",
    long_term_memory_store_path: str | Path = "data/long_term_memory.json",
    timer_background: bool = True,
    output: ConsoleOutput | None = None,
) -> AgentCore:
    long_term_store = LongTermMemoryStore(long_term_memory_store_path)
    profile_service = UserProfileService(UserProfileStore(profile_store_path))
    return AgentCore(
        output=output or ConsoleOutput(),
        timer_service=TimerService(background=timer_background),
        runtime_history_service=RuntimeHistoryService(),
        llm_service=LLMService(),
        store=JsonStore(store_path),
        long_term_memory_pipeline=LongTermMemoryPipeline(long_term_store),
        personal_context_builder=PersonalContextBuilder(
            long_term_memory_store=long_term_store,
            user_profile_service=profile_service,
        ),
    )


def _default_profile_store_path(store: JsonStore) -> Path:
    return store.path.with_name("user_profiles.json")


def _default_long_term_memory_store_path(store: JsonStore) -> Path:
    return store.path.with_name("long_term_memory.json")


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
