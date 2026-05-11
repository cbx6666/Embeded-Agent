from __future__ import annotations

"""Agent 单事件核心调度器。"""

import json
import threading
import time
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.decision.intent import AgentIntent
from src.agent.decision.policy import decide_actions_with_intents
from src.agent.event import Event
from src.agent.memory.memory_pipeline import MemoryPipeline
from src.agent.reducer import reduce_state
from src.agent.runtime.action_result import ActionResult
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import UserProfileService
from src.storage.behavior_store import BehaviorStore
from src.storage.json_store import JsonStore
from src.storage.profile_store import ProfileStore


class AgentCore:
    """负责处理单个事件的核心调度器。"""

    def __init__(
        self,
        output: ConsoleOutput,
        timer_service: TimerService,
        memory_service: MemoryService,
        llm_service: LLMService,
        store: JsonStore,
        profile_service: UserProfileService | None = None,
        memory_pipeline: MemoryPipeline | None = None,
    ) -> None:
        """初始化输出、服务依赖和持久化状态。"""
        self.output = output
        self.timer_service = timer_service
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.store = store
        self.profile_service = profile_service or UserProfileService(
            ProfileStore(_default_profile_store_path(store))
        )
        self.memory_pipeline = memory_pipeline or MemoryPipeline(
            BehaviorStore(_default_behavior_store_path(store)),
            self.profile_service,
        )
        self.state = AgentState.from_dict(self.store.load_state_dict())
        self.state.current_user_id = self.profile_service.ensure_user_id(self.state.current_user_id)
        self.last_intents: list[AgentIntent] = []
        self.last_action_results: list[ActionResult] = []
        self._lock = threading.RLock()

    def handle_event_with_results(
        self,
        event: Event,
    ) -> tuple[list[Action], list[ActionResult]]:
        """处理单个事件，并返回动作及其执行结果。"""
        with self._lock:
            # 先保留旧状态，供 planner 比较“事件前后”差异使用。
            previous_state = AgentState.from_dict(self.state.to_dict())
            self.state = reduce_state(self.state, event)
            
            if _should_touch_profile(event):
                self.profile_service.touch_user(self.state.current_user_id, timestamp=event.timestamp)
            self.memory_pipeline.process_event(self.state.current_user_id, event, self.state)
            self.memory_service.record_event(self.state, event)

            # 用户输入会写入短期消息记忆，供后续规则和 LLM 使用。
            if event.type in {"user_text_input", "speech_recognized"}:
                text = str(event.payload.get("text", "")).strip()
                if text:
                    self.memory_service.record_message(
                        self.state,
                        role="user",
                        text=text,
                        timestamp=event.timestamp,
                    )

            # 长期行为模型只产出策略快照；Planner/Realizer 不直接读写 profile。
            personalized_policy = self.memory_pipeline.build_personalized_policy(self.state.current_user_id)

            # 先得到意图，再把意图落成动作；AgentCore 不直接拼装动作细节。
            intents, actions = decide_actions_with_intents(
                previous_state=previous_state,
                current_state=self.state,
                event=event,
                llm_service=self.llm_service,
                profile_service=self.profile_service,
                personalized_policy=personalized_policy,
            )
            results = self._execute_actions(actions, event.timestamp)
            self.memory_pipeline.process_actions(self.state.current_user_id, actions, event.timestamp)

            self.last_intents = intents
            self.last_action_results = results
            self.memory_service.trim(self.state)
            self.store.save_state(self.state)
            return actions, results

    def render_state(self) -> str:
        """将当前状态渲染为格式化 JSON 文本。"""
        with self._lock:
            return json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2)

    def render_history(self) -> str:
        """将当前短期记忆渲染为格式化 JSON 文本。"""
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
        """渲染当前用户的长期画像和偏好。"""
        with self._lock:
            return self.profile_service.render_profile(self.state.current_user_id)

    def render_users(self) -> str:
        """渲染已知用户列表。"""
        with self._lock:
            return self.profile_service.render_users(current_user_id=self.state.current_user_id)

    def switch_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        timestamp: int | None = None,
    ) -> str:
        """切换当前活跃用户，并保证长期 profile 存在。"""
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self.profile_service.switch_user(
                user_id,
                display_name=display_name,
                timestamp=ts,
            )
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
        """更新当前用户的显式偏好。"""
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
        """更新当前用户的基础资料。"""
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
        """停止内部服务并持久化当前状态。"""
        with self._lock:
            self.timer_service.stop()
            self.store.save_state(self.state)

    def _execute_actions(self, actions: list[Action], action_ts: int) -> list[ActionResult]:
        """顺序执行动作列表，并为每个动作生成执行结果。"""
        return [self._execute_action(action, action_ts) for action in actions]

    def _execute_action(self, action: Action, action_ts: int) -> ActionResult:
        """执行单个动作，并将执行结果封装为 ActionResult。"""
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

            # 这类动作既会影响交互状态，也可能需要写回消息记忆和冷却记录。
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
        except Exception as exc:  # pragma: no cover
            return ActionResult(
                action_type=action.type,
                success=False,
                timestamp=action_ts,
                reason=str(exc),
                payload=dict(action.payload),
            )

    def _mark_cooldown_if_needed(self, action: Action, action_ts: int) -> None:
        """在提醒类动作执行后记录对应的冷却时间。"""
        if action.payload.get("kind") != "notification":
            return
        reason = action.payload.get("reason")
        if reason:
            self.state.cooldown.reminder_last_ts[str(reason)] = action_ts

    def _on_timer_tick(self, remaining_sec: int) -> None:
        """将定时器回调重新包装成标准事件并走统一链路。"""
        event_type = "timer_finished" if remaining_sec <= 0 else "timer_ticked"
        event = Event(
            type=event_type,
            timestamp=int(time.time()),
            payload={"remaining_sec": remaining_sec, "timer": "focus"},
        )
        self.handle_event_with_results(event)


def build_default_core(
    store_path: str | Path = "data/runtime_store.json",
    profile_store_path: str | Path = "data/user_profiles.json",
    behavior_store_path: str | Path = "data/behavior_stats.json",
    timer_background: bool = True,
    output: ConsoleOutput | None = None,
) -> AgentCore:
    """使用默认服务依赖构造一个 AgentCore 实例。"""
    profile_service = UserProfileService(ProfileStore(profile_store_path))
    return AgentCore(
        output=output or ConsoleOutput(),
        timer_service=TimerService(background=timer_background),
        memory_service=MemoryService(),
        llm_service=LLMService(),
        store=JsonStore(store_path),
        profile_service=profile_service,
        memory_pipeline=MemoryPipeline(
            BehaviorStore(behavior_store_path),
            profile_service,
        ),
    )


def _default_profile_store_path(store: JsonStore) -> Path:
    """Place test profile stores next to their runtime store by default."""
    return store.path.with_name("user_profiles.json")


def _default_behavior_store_path(store: JsonStore) -> Path:
    """Place long-term behavior stats next to the runtime store by default."""
    return store.path.with_name("behavior_stats.json")


def _should_touch_profile(event: Event) -> bool:
    """Only user-facing/user-sensing events should update last_seen_at."""
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
