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
from src.agent.config.policy_config import (
    ActionPolicyConfig,
    ContextPolicyConfig,
    CopyPolicyConfig,
    DecisionPolicyConfig,
    GuardPolicyConfig,
    RuntimeHistoryPolicyConfig,
)
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import AgentIntent
from src.agent.event import Event
from src.agent.memory.memory_background_worker import MemoryBackgroundWorker
from src.agent.memory.memory_gate import should_process_action_memory, should_process_event_memory
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.reducer import reduce_state
from src.agent.execution.action_result import ActionResult
from src.agent.execution.device_adapter import DeviceAdapter
from src.agent.execution.trace import RuntimeTrace
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
        memory_worker: MemoryBackgroundWorker | None = None,
    ) -> None:
        self.output = output
        self.timer_service = timer_service
        self.runtime_history_service = runtime_history_service
        self.llm_service = llm_service
        self.store = store
        self._event_handled_callback: Callable[[AgentState], None] | None = None

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
        self.memory_worker = memory_worker or MemoryBackgroundWorker(
            self.long_term_memory_pipeline,
            self.llm_service,
        )

        self.state = AgentState.from_dict(self.store.load_state_dict())
        self.state.current_user_id = self._profile_service().ensure_user_id(self.state.current_user_id)

        self.device_adapter = DeviceAdapter(
            output=self.output,
            timer_service=self.timer_service,
            timer_callback=self._on_timer_tick,
        )

        self.last_intents: list[AgentIntent] = []
        self.last_decision_result: DecisionResult | None = None
        self.last_effective_decision_result: DecisionResult | None = None
        self.last_action_results: list[ActionResult] = []
        self.last_effective_action_results: list[ActionResult] = []
        self.last_runtime_trace: RuntimeTrace | None = None
        self._lock = threading.RLock()

    def handle_event(self, event: Event) -> tuple[list[Action], list[ActionResult]]:
        """处理一条标准事件并返回动作与执行结果。"""

        with self._lock:
            previous_state = AgentState.from_dict(self.state.to_dict())
            trace = RuntimeTrace()
            trace.add("event", "received", event=_event_to_dict(event))

            self.state = reduce_state(self.state, event)
            trace.add(
                "reducer",
                "state_reduced",
                previous_state=_trace_state_summary(previous_state),
                current_state=_trace_state_summary(self.state),
            )

            if _should_touch_profile(event):
                self._profile_service().touch_user(self.state.current_user_id, timestamp=event.timestamp)

            self.runtime_history_service.record_event(self.state, event)
            self._record_user_message_if_needed(event)

            allow_event_memory, event_memory_reason = should_process_event_memory(event)
            event_memory_user_id = self.state.current_user_id
            event_memory_state = AgentState.from_dict(self.state.to_dict())
            if not allow_event_memory:
                trace.add(
                    "memory_pipeline",
                    "event_skipped",
                    memory_event_gate_allowed=False,
                    memory_event_skip_reason=event_memory_reason,
                    memory_event_enqueued=False,
                    memory_event_enqueue_error=None,
                    memory_event_submit_result=None,
                    memory_event_sync_called=False,
                    memory_event_pipeline_called=False,
                    memory_event_pipeline_called_sync=False,
                )

            personal_context = self.personal_context_builder.build(
                user_id=self.state.current_user_id,
                state=self.state,
                event=event,
            )
            trace.add("personal_context", "built", personal_context=personal_context.to_dict())

            decision_result = self.decision_pipeline.decide(
                previous_state=previous_state,
                current_state=self.state,
                event=event,
                llm_service=self.llm_service,
                personal_context=personal_context,
            )
            trace.extend(decision_result.stage_metadata.get("trace"))
            actions = decision_result.actions
            results = self._execute_actions(actions, event.timestamp, source_event=event)
            trace.add(
                "action_result",
                "executed",
                results=[_action_result_to_dict(result) for result in results],
            )

            # 当前轮决策和动作执行完成后只提交后台任务，不等待 Memory Pipeline。
            # 因此本轮 PersonalContext 不会读取到本轮刚产生的长期记忆。
            if allow_event_memory:
                try:
                    submit_result = self.memory_worker.submit_event_memory(
                        user_id=event_memory_user_id,
                        event=event,
                        state=event_memory_state,
                    )
                except Exception as exc:
                    submit_payload = {
                        "accepted": False,
                        "task_id": None,
                        "reason": "submit_error",
                        "queue_size": -1,
                    }
                    trace.add(
                        "memory_pipeline",
                        "event_enqueue_failed",
                        memory_event_gate_allowed=True,
                        memory_event_skip_reason="",
                        memory_event_enqueued=False,
                        memory_event_enqueue_error=str(exc),
                        memory_event_submit_result=submit_payload,
                        memory_event_sync_called=False,
                        memory_event_pipeline_called=False,
                        memory_event_pipeline_called_sync=False,
                    )
                else:
                    submit_payload = submit_result.to_dict()
                    trace.add(
                        "memory_pipeline",
                        "event_enqueued" if submit_result.accepted else "event_enqueue_failed",
                        memory_event_gate_allowed=True,
                        memory_event_skip_reason="",
                        memory_event_enqueued=submit_result.accepted,
                        memory_event_enqueue_error=(
                            None if submit_result.accepted else submit_result.reason
                        ),
                        memory_event_submit_result=submit_payload,
                        memory_event_sync_called=False,
                        memory_event_pipeline_called=False,
                        memory_event_pipeline_called_sync=False,
                    )

            allow_action_memory, action_memory_reason = should_process_action_memory(
                actions=actions,
                results=results,
                source_event=event,
            )
            # 动作记忆同样采用“只确认入队”的最终一致性语义；提交失败只进入
            # trace，不得改变当前轮已经生成的响应和动作结果。
            if allow_action_memory:
                try:
                    submit_result = self.memory_worker.submit_action_memory(
                        user_id=self.state.current_user_id,
                        actions=actions,
                        timestamp=event.timestamp,
                        action_results=results,
                        source_event=event,
                        state=self.state,
                    )
                except Exception as exc:
                    submit_payload = {
                        "accepted": False,
                        "task_id": None,
                        "reason": "submit_error",
                        "queue_size": -1,
                    }
                    trace.add(
                        "memory_pipeline",
                        "action_enqueue_failed",
                        memory_action_gate_allowed=True,
                        memory_action_skip_reason="",
                        memory_action_enqueued=False,
                        memory_action_enqueue_error=str(exc),
                        memory_action_submit_result=submit_payload,
                        memory_action_sync_called=False,
                        memory_action_pipeline_called_sync=False,
                        memory_async_submit_success=False,
                        memory_async_submit_error=str(exc),
                    )
                else:
                    submit_payload = submit_result.to_dict()
                    trace.add(
                        "memory_pipeline",
                        "action_enqueued" if submit_result.accepted else "action_enqueue_failed",
                        memory_action_gate_allowed=True,
                        memory_action_skip_reason="",
                        memory_action_enqueued=submit_result.accepted,
                        memory_action_enqueue_error=(
                            None if submit_result.accepted else submit_result.reason
                        ),
                        memory_action_submit_result=submit_payload,
                        memory_action_sync_called=False,
                        memory_action_pipeline_called_sync=False,
                        memory_async_submit_success=submit_result.accepted,
                        memory_async_submit_error=(
                            None if submit_result.accepted else submit_result.reason
                        ),
                    )
            else:
                trace.add(
                    "memory_pipeline",
                    "action_skipped",
                    memory_action_gate_allowed=False,
                    memory_action_skip_reason=action_memory_reason,
                    memory_action_enqueued=False,
                    memory_action_enqueue_error=None,
                    memory_action_submit_result=None,
                    memory_action_sync_called=False,
                    memory_action_pipeline_called_sync=False,
                    memory_async_submit_success=False,
                    memory_async_submit_error=None,
                )

            self.last_intents = decision_result.intents
            self.last_decision_result = decision_result
            self.last_action_results = results
            if actions:
                self.last_effective_decision_result = decision_result
                self.last_effective_action_results = results

            self.runtime_history_service.trim(self.state)
            self.store.save_state(self.state)
            self.last_runtime_trace = trace
            
            # 调用事件处理后的回调
            if self._event_handled_callback is not None:
                try:
                    self._event_handled_callback(self.state)
                except:
                    pass
                    
            return actions, results

    def handle_event_with_results(self, event: Event) -> tuple[list[Action], list[ActionResult]]:
        """语音/传感器适配器兼容入口，等价于 handle_event。"""
        return self.handle_event(event)

    def set_event_handled_callback(
        self,
        callback: Callable[[AgentState], None] | None,
    ) -> None:
        """设置事件处理完成后的回调。"""
        self._event_handled_callback = callback

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
            # 先停止接收新的记忆任务并限时排空队列，再关闭其他运行时服务。
            self.memory_worker.shutdown(timeout=5.0)
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

    def _execute_actions(self, actions: list[Action], action_ts: int, *, source_event: Event) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in actions:
            result = self.device_adapter.execute(action, action_ts)
            results.append(result)
            if result.success:
                self._after_successful_action(action, action_ts, source_event=source_event)
        return results

    def _after_successful_action(self, action: Action, action_ts: int, *, source_event: Event) -> None:
        self.runtime_history_service.record_action(self.state, action, action_ts)
        self._sync_focus_state_from_timer_action(action, action_ts, source_event=source_event)

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

    def _sync_focus_state_from_timer_action(self, action: Action, action_ts: int, *, source_event: Event) -> None:
        if action.type == "start_timer":
            self.state = reduce_state(
                self.state,
                Event(
                    type="system_triggered",
                    timestamp=action_ts,
                    payload={
                        "trigger": "focus_timer_started",
                        "source": "agent_action_result",
                        "source_event_type": source_event.type,
                        "duration_sec": action.payload.get("duration_sec"),
                    },
                ),
            )
            return

        if action.type == "stop_timer":
            self.state = reduce_state(
                self.state,
                Event(
                    type="system_triggered",
                    timestamp=action_ts,
                    payload={
                        "trigger": "focus_timer_stopped",
                        "source": "agent_action_result",
                        "source_event_type": source_event.type,
                    },
                ),
            )

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
    store_path: str | Path = "data/runtime/runtime_store.json",
    profile_store_path: str | Path = "data/user/user_profiles.json",
    long_term_memory_store_path: str | Path = "data/memory/long_term_memory.json",
    timer_background: bool = True,
    output: ConsoleOutput | None = None,
    *,
    runtime_history_policy: RuntimeHistoryPolicyConfig | None = None,
    context_policy: ContextPolicyConfig | None = None,
    decision_policy: DecisionPolicyConfig | None = None,
    guard_policy: GuardPolicyConfig | None = None,
    action_policy: ActionPolicyConfig | None = None,
    copy_policy: CopyPolicyConfig | None = None,
) -> AgentCore:
    long_term_store = LongTermMemoryStore(long_term_memory_store_path)
    profile_service = UserProfileService(UserProfileStore(profile_store_path))

    runtime_history_service = RuntimeHistoryService(policy_config=runtime_history_policy)
    personal_context_builder = PersonalContextBuilder(
        long_term_memory_store=long_term_store,
        user_profile_service=profile_service,
        policy_config=context_policy,
    )

    decision_pipeline = None
    if decision_policy or guard_policy or action_policy or copy_policy:
        guard = DeterministicGuard(policy_config=guard_policy) if guard_policy else DeterministicGuard()
        realizer = ActionRealizer(
            action_policy=action_policy,
            copy_policy=copy_policy,
        ) if (action_policy or copy_policy) else ActionRealizer()
        decision_pipeline = DecisionPipeline(
            guard=guard,
            action_realizer=realizer,
            decision_policy=decision_policy,
        )

    return AgentCore(
        output=output or ConsoleOutput(),
        timer_service=TimerService(background=timer_background),
        runtime_history_service=runtime_history_service,
        llm_service=LLMService(),
        store=JsonStore(store_path),
        long_term_memory_pipeline=LongTermMemoryPipeline(long_term_store),
        personal_context_builder=personal_context_builder,
        decision_pipeline=decision_pipeline,
    )


def _default_profile_store_path(store: JsonStore) -> Path:
    if store.path.parent.name == "runtime":
        return store.path.parent.parent / "user" / "user_profiles.json"
    return store.path.with_name("user_profiles.json")


def _default_long_term_memory_store_path(store: JsonStore) -> Path:
    if store.path.parent.name == "runtime":
        return store.path.parent.parent / "memory" / "long_term_memory.json"
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
        "user_posture_updated",
        "user_activity_updated",
    }


def _event_to_dict(event: Event) -> dict[str, object]:
    return {
        "type": event.type,
        "timestamp": event.timestamp,
        "payload": dict(event.payload),
    }


def _trace_state_summary(state: AgentState) -> dict[str, object]:
    return {
        "current_user_id": state.current_user_id,
        "interaction": {
            "mode": state.interaction.mode,
            "dialogue_state": state.interaction.dialogue_state,
            "in_conversation": state.interaction.in_conversation,
        },
        "focus": {
            "active": state.focus.active,
            "elapsed_sec": state.focus.elapsed_sec,
            "remaining_sec": state.focus.remaining_sec,
            "target_duration_sec": state.focus.target_duration_sec,
        },
        "user": {
            "presence": state.user.presence,
            "attention": state.user.attention,
            "behavior": state.user.behavior,
            "emotion": state.user.emotion,
            "fatigue_level": state.user.fatigue_level,
            "posture": state.user.posture,
            "current_activity": state.user.current_activity,
        },
        "history_counts": {
            "recent_events": len(state.runtime_history.recent_events),
            "recent_messages": len(state.runtime_history.recent_messages),
            "recent_actions": len(state.runtime_history.recent_actions),
        },
        "cooldowns": dict(state.cooldown.reminder_last_ts),
    }


def _action_result_to_dict(result: ActionResult) -> dict[str, object]:
    return {
        "action_type": result.action_type,
        "success": result.success,
        "timestamp": result.timestamp,
        "reason": result.reason,
        "payload": dict(result.payload),
    }
