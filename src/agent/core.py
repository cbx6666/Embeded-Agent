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
from collections.abc import Callable
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.config.policy_config import (
    ActionPolicyConfig,
    AutonomousCheckPolicyConfig,
    AutonomousScheduleConfig,
    ContextPolicyConfig,
    CopyPolicyConfig,
    DecisionPolicyConfig,
    GuardPolicyConfig,
    RuntimeHistoryPolicyConfig,
)
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.autonomous_check_policy import AutonomousCheckPolicy
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import AgentIntent
from src.agent.event import Event, EventPriorityRouter, EventRoute
from src.agent.event.dedicated_event_handler import DedicatedEventHandler
from src.agent.memory.memory_background_worker import MemoryBackgroundWorker
from src.agent.memory.memory_gate import should_process_action_memory, should_process_event_memory
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.reducer import reduce_state
from src.agent.scheduling import AutonomousScheduler
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
        event_priority_router: EventPriorityRouter | None = None,
        autonomous_check_policy: AutonomousCheckPolicy | None = None,
        autonomous_scheduler: AutonomousScheduler | None = None,
        autonomous_schedule_config: AutonomousScheduleConfig | None = None,
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
        self.event_priority_router = event_priority_router or EventPriorityRouter()
        self.autonomous_check_policy = autonomous_check_policy or AutonomousCheckPolicy()
        self.dedicated_event_handler = DedicatedEventHandler(self._profile_service())
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
        self.autonomous_scheduler = autonomous_scheduler or AutonomousScheduler(
            state_provider=self._snapshot_state,
            event_sink=self.handle_event,
            config=autonomous_schedule_config,
        )

    def handle_event(self, event: Event) -> tuple[list[Action], list[ActionResult]]:
        """处理一条标准事件并返回动作与执行结果。"""

        with self._lock:
            previous_state = AgentState.from_dict(self.state.to_dict())
            trace = RuntimeTrace()
            trace.add("event", "received", event=_event_to_dict(event))

            # Event 先归约为事实，再分类处理方式。Router 看到的是最新状态，但它本身
            # 不修改状态；这保证 P2/P3 即使跳过决策，传感器事实也不会丢失。
            self.state = reduce_state(self.state, event)
            trace.add(
                "reducer",
                "state_reduced",
                previous_state=_trace_state_summary(previous_state),
                current_state=_trace_state_summary(self.state),
            )

            route = self.event_priority_router.classify(event)
            route_payload = _event_route_to_dict(route)
            trace.add(
                "event_route",
                "classified",
                **route_payload,
                decision_skipped_by_router=not route.should_enter_decision,
                requires_dedicated_handler=route.handling
                in {"profile_handler", "settings_handler", "feedback_signal"},
            )

            # P4 专用处理先落显式资料；随后所有事件统一进入短期事实记录。
            self._handle_dedicated_route(event=event, route=route, trace=trace)
            self._record_event_facts(event)
            (
                allow_event_memory,
                event_memory_user_id,
                event_memory_state,
            ) = self._prepare_event_memory(event=event, trace=trace)

            # 决策阶段只负责选择 P0A/P0B/P1；P2-P4 在函数内直接形成 no-op。
            decision_result = self._decide_by_route(
                event=event,
                previous_state=previous_state,
                route=route,
                route_payload=route_payload,
                trace=trace,
            )
            actions = decision_result.actions
            results = self._execute_actions(actions, event.timestamp, source_event=event)
            trace.add(
                "action_result",
                "executed",
                results=[_action_result_to_dict(result) for result in results],
            )

            # 当前轮决策和动作执行完成后只提交后台任务，不等待 Memory Pipeline。
            # 因此本轮 PersonalContext 不会读取到本轮刚产生的长期记忆。
            self._submit_event_memory(
                allowed=allow_event_memory,
                user_id=event_memory_user_id,
                event=event,
                state=event_memory_state,
                trace=trace,
            )
            self._submit_action_memory(
                event=event,
                actions=actions,
                results=results,
                trace=trace,
            )
            # 最后统一保存观察字段和状态，避免各路由分支自行持久化。
            self._finalize_event(
                decision_result=decision_result,
                actions=actions,
                results=results,
                trace=trace,
            )
            return actions, results

    def _handle_dedicated_route(
        self,
        *,
        event: Event,
        route: EventRoute,
        trace: RuntimeTrace,
    ) -> None:
        """处理 P4 profile/settings 事件；反馈信号只进入历史和异步记忆。"""

        # P4 profile/settings：使用专用确定性处理器，不进入 DecisionPipeline。
        if route.handling not in {"profile_handler", "settings_handler"}:
            return
        dedicated_result = self.dedicated_event_handler.handle(
            event=event,
            state=self.state,
        )
        trace.add(
            "dedicated_event_handler",
            "handled" if dedicated_result.handled else "skipped",
            **dedicated_result.to_dict(),
        )

    def _record_event_facts(self, event: Event) -> None:
        """记录 profile 活跃时间、运行时事件和用户消息事实。"""

        if _should_touch_profile(event) and event.type != "user_switched":
            self._profile_service().touch_user(
                self.state.current_user_id,
                timestamp=event.timestamp,
            )
        self.runtime_history_service.record_event(self.state, event)
        self._record_user_message_if_needed(event)

    def _prepare_event_memory(
        self,
        *,
        event: Event,
        trace: RuntimeTrace,
    ) -> tuple[bool, str, AgentState]:
        """执行事件记忆 Gate，并冻结提交后台任务所需的当前状态快照。"""

        allowed, reason = should_process_event_memory(event)
        user_id = self.state.current_user_id
        state_snapshot = AgentState.from_dict(self.state.to_dict())
        if allowed:
            return True, user_id, state_snapshot

        trace.add(
            "memory_pipeline",
            "event_skipped",
            memory_event_gate_allowed=False,
            memory_event_skip_reason=reason,
            memory_event_enqueued=False,
            memory_event_enqueue_error=None,
            memory_event_submit_result=None,
            memory_event_sync_called=False,
            memory_event_pipeline_called=False,
            memory_event_pipeline_called_sync=False,
        )
        return False, user_id, state_snapshot

    def _decide_by_route(
        self,
        *,
        event: Event,
        previous_state: AgentState,
        route: EventRoute,
        route_payload: dict[str, object],
        trace: RuntimeTrace,
    ) -> DecisionResult:
        """按照 Router 结果选择 P0A、P0B、P1 或无决策路径。"""

        # P2/P3/P4：事实、遥测和专用事件已经处理完成，不创建 PersonalContext。
        if not route.should_enter_decision:
            return self._decision_skipped_by_router(
                route=route,
                route_payload=route_payload,
                trace=trace,
            )

        # P0B：意图由事件类型唯一确定，直接构造规则计划，保持 0 LLM。
        if route.handling == "rule_intent_builder":
            decision_result = self._decide_structured_event(
                event=event,
                previous_state=previous_state,
                trace=trace,
            )
        # P1：系统时间只产生检查机会，必须先经过状态、趋势和 cooldown 门控。
        elif route.handling == "low_frequency_check":
            decision_result = self._decide_autonomous_check(
                event=event,
                previous_state=previous_state,
                trace=trace,
            )
        # P0A：开放自然语言才构建完整 PersonalContext 并进入 Orchestrator。
        elif route.handling == "orchestrator":
            decision_result = self._decide_open_semantic_event(
                event=event,
                previous_state=previous_state,
                trace=trace,
            )
        # 防御分支：Router 若错误放行未知 handling，不得默认升级成 LLM 调用。
        else:
            decision_result = _skipped_decision_result(
                reason=f"unsupported decision handling: {route.handling}",
                decision_source="event_priority_router",
            )

        decision_result.stage_metadata.setdefault("event_route", route_payload)
        trace.extend(decision_result.stage_metadata.get("trace"))
        return decision_result

    def _decide_structured_event(
        self,
        *,
        event: Event,
        previous_state: AgentState,
        trace: RuntimeTrace,
    ) -> DecisionResult:
        """执行 P0B RuleIntentBuilder 路径。"""

        trace.add(
            "personal_context",
            "skipped",
            decision_source="rule_intent_builder",
            structured_decision=True,
            reason="structured decisions use minimal AgentContext",
        )
        return self.decision_pipeline.decide_structured(
            previous_state=previous_state,
            current_state=self.state,
            event=event,
        )

    def _decide_autonomous_check(
        self,
        *,
        event: Event,
        previous_state: AgentState,
        trace: RuntimeTrace,
    ) -> DecisionResult:
        """执行 P1 gate，并分别处理 skip、rule 和 LLM 三种结果。"""

        gate = self.autonomous_check_policy.evaluate(
            event=event,
            state=self.state,
        )
        trace.add(
            "autonomous_check_policy",
            "evaluated",
            **gate.to_dict(),
        )

        # P1 skip：业务前提或持续性不足时，到这里立即结束，不构建上下文。
        if gate.mode == "skip":
            trace.add(
                "personal_context",
                "skipped",
                decision_source="autonomous_check_policy",
                reason=gate.reason,
            )
            return _skipped_decision_result(
                reason=gate.reason,
                decision_source="autonomous_check_policy",
                metadata={"autonomous_check": gate.to_dict()},
            )

        # P1 rule：单一、确定的异常使用预构造计划，仍经过统一后处理链。
        if gate.mode == "rule" and gate.plan is not None:
            self._mark_autonomous_check_admitted(gate.trigger, event.timestamp)
            trace.add(
                "personal_context",
                "skipped",
                decision_source="autonomous_check_policy",
                reason="autonomous rule plan uses minimal AgentContext",
            )
            return self.decision_pipeline.decide_prebuilt(
                previous_state=previous_state,
                current_state=self.state,
                event=event,
                plan=gate.plan,
                decision_source="autonomous_check_policy",
                source_metadata={"autonomous_check": gate.to_dict()},
            )

        # P1 LLM：只有多信号或确需语义权衡时，才支付 PersonalContext 和 LLM 成本。
        self._mark_autonomous_check_admitted(gate.trigger, event.timestamp)
        decision_result = self._decide_open_semantic_event(
            event=event,
            previous_state=previous_state,
            trace=trace,
        )
        decision_result.stage_metadata["autonomous_check"] = gate.to_dict()
        return decision_result

    def _decide_open_semantic_event(
        self,
        *,
        event: Event,
        previous_state: AgentState,
        trace: RuntimeTrace,
    ) -> DecisionResult:
        """构建 PersonalContext 并执行 Orchestrator 路径。"""

        personal_context = self.personal_context_builder.build(
            user_id=self.state.current_user_id,
            state=self.state,
            event=event,
        )
        trace.add(
            "personal_context",
            "built",
            personal_context=personal_context.to_dict(),
        )
        return self.decision_pipeline.decide(
            previous_state=previous_state,
            current_state=self.state,
            event=event,
            llm_service=self.llm_service,
            personal_context=personal_context,
        )

    @staticmethod
    def _decision_skipped_by_router(
        *,
        route: EventRoute,
        route_payload: dict[str, object],
        trace: RuntimeTrace,
    ) -> DecisionResult:
        """为 P2/P3/P4 生成可追踪的 no-op 决策结果。"""

        trace.add(
            "personal_context",
            "skipped",
            decision_skipped_by_router=True,
            event_route_reason=route.reason,
        )
        trace.add(
            "decision_pipeline",
            "skipped",
            decision_skipped_by_router=True,
            event_route_reason=route.reason,
        )
        return DecisionResult(
            intents=[AgentIntent(type="no_op", reason=route.reason)],
            actions=[],
            used_llm=False,
            fallback_reason="decision_skipped_by_router",
            decision_reason=route.reason,
            stage_metadata={
                "event_route": route_payload,
                "decision_skipped_by_router": True,
            },
        )

    def _submit_event_memory(
        self,
        *,
        allowed: bool,
        user_id: str,
        event: Event,
        state: AgentState,
        trace: RuntimeTrace,
    ) -> None:
        """异步提交事件记忆；Gate 拒绝时调用方已经记录 skip trace。"""

        if not allowed:
            return
        try:
            submit_result = self.memory_worker.submit_event_memory(
                user_id=user_id,
                event=event,
                state=state,
            )
        except Exception as exc:
            trace.add(
                "memory_pipeline",
                "event_enqueue_failed",
                memory_event_gate_allowed=True,
                memory_event_skip_reason="",
                memory_event_enqueued=False,
                memory_event_enqueue_error=str(exc),
                memory_event_submit_result=_failed_memory_submit_payload(),
                memory_event_sync_called=False,
                memory_event_pipeline_called=False,
                memory_event_pipeline_called_sync=False,
            )
            return

        trace.add(
            "memory_pipeline",
            "event_enqueued" if submit_result.accepted else "event_enqueue_failed",
            memory_event_gate_allowed=True,
            memory_event_skip_reason="",
            memory_event_enqueued=submit_result.accepted,
            memory_event_enqueue_error=(
                None if submit_result.accepted else submit_result.reason
            ),
            memory_event_submit_result=submit_result.to_dict(),
            memory_event_sync_called=False,
            memory_event_pipeline_called=False,
            memory_event_pipeline_called_sync=False,
        )

    def _submit_action_memory(
        self,
        *,
        event: Event,
        actions: list[Action],
        results: list[ActionResult],
        trace: RuntimeTrace,
    ) -> None:
        """执行 Action Memory Gate，并按最终一致性语义提交后台任务。"""

        allowed, reason = should_process_action_memory(
            actions=actions,
            results=results,
            source_event=event,
        )
        # Gate 拒绝：没有长期价值的设备反馈只保留在 RuntimeHistory/trace。
        if not allowed:
            trace.add(
                "memory_pipeline",
                "action_skipped",
                memory_action_gate_allowed=False,
                memory_action_skip_reason=reason,
                memory_action_enqueued=False,
                memory_action_enqueue_error=None,
                memory_action_submit_result=None,
                memory_action_sync_called=False,
                memory_action_pipeline_called_sync=False,
                memory_async_submit_success=False,
                memory_async_submit_error=None,
            )
            return

        # Gate 放行：只确认后台入队；失败不能改变已经完成的响应和动作结果。
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
            trace.add(
                "memory_pipeline",
                "action_enqueue_failed",
                memory_action_gate_allowed=True,
                memory_action_skip_reason="",
                memory_action_enqueued=False,
                memory_action_enqueue_error=str(exc),
                memory_action_submit_result=_failed_memory_submit_payload(),
                memory_action_sync_called=False,
                memory_action_pipeline_called_sync=False,
                memory_async_submit_success=False,
                memory_async_submit_error=str(exc),
            )
            return

        trace.add(
            "memory_pipeline",
            "action_enqueued" if submit_result.accepted else "action_enqueue_failed",
            memory_action_gate_allowed=True,
            memory_action_skip_reason="",
            memory_action_enqueued=submit_result.accepted,
            memory_action_enqueue_error=(
                None if submit_result.accepted else submit_result.reason
            ),
            memory_action_submit_result=submit_result.to_dict(),
            memory_action_sync_called=False,
            memory_action_pipeline_called_sync=False,
            memory_async_submit_success=submit_result.accepted,
            memory_async_submit_error=(
                None if submit_result.accepted else submit_result.reason
            ),
        )

    def _finalize_event(
        self,
        *,
        decision_result: DecisionResult,
        actions: list[Action],
        results: list[ActionResult],
        trace: RuntimeTrace,
    ) -> None:
        """保存本轮结果、裁剪历史、持久化状态并触发观察回调。"""

        self.last_intents = decision_result.intents
        self.last_decision_result = decision_result
        self.last_action_results = results
        if actions:
            self.last_effective_decision_result = decision_result
            self.last_effective_action_results = results

        self.runtime_history_service.trim(self.state)
        self.store.save_state(self.state)
        self.last_runtime_trace = trace

        if self._event_handled_callback is None:
            return
        try:
            self._event_handled_callback(self.state)
        except Exception:
            pass

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
        self.autonomous_scheduler.stop()
        with self._lock:
            # 先停止接收新的记忆任务并限时排空队列，再关闭其他运行时服务。
            self.memory_worker.shutdown(timeout=5.0)
            self.timer_service.stop()
            self.store.save_state(self.state)

    def start_autonomous_scheduler(self) -> None:
        """启动系统时间驱动的低频 P1 检查。"""

        self.autonomous_scheduler.start()

    def _snapshot_state(self) -> AgentState:
        """为调度线程提供不可共享的状态快照。"""

        with self._lock:
            return AgentState.from_dict(self.state.to_dict())

    def _mark_autonomous_check_admitted(self, trigger: str, timestamp: int) -> None:
        """记录 P1 gate 准入时间，阻止手工重复注入绕过频率限制。"""

        if trigger:
            self.state.cooldown.autonomous_check_last_ts[str(trigger)] = int(timestamp)

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
    autonomous_check_policy: AutonomousCheckPolicyConfig | None = None,
    autonomous_schedule_config: AutonomousScheduleConfig | None = None,
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
        autonomous_check_policy=AutonomousCheckPolicy(
            policy_config=autonomous_check_policy,
            guard_policy_config=guard_policy,
        ),
        autonomous_schedule_config=autonomous_schedule_config,
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


def _event_route_to_dict(route: EventRoute) -> dict[str, object]:
    return {
        "event_route_priority": route.priority,
        "event_route_handling": route.handling,
        "event_route_reason": route.reason,
        "should_enter_decision": route.should_enter_decision,
        "should_allow_llm": route.should_allow_llm,
    }


def _skipped_decision_result(
    *,
    reason: str,
    decision_source: str,
    metadata: dict[str, object] | None = None,
) -> DecisionResult:
    return DecisionResult(
        intents=[AgentIntent(type="no_op", reason=reason)],
        actions=[],
        used_llm=False,
        fallback_reason=reason,
        decision_reason=reason,
        stage_metadata={
            "decision_source": decision_source,
            "used_llm": False,
            "llm_mode": "none",
            "llm_roles_called": [],
            "llm_call_count": 0,
            **dict(metadata or {}),
        },
    )


def _failed_memory_submit_payload() -> dict[str, object]:
    """构造 MemoryBackgroundWorker 提交异常时的稳定 trace 结构。"""

    return {
        "accepted": False,
        "task_id": None,
        "reason": "submit_error",
        "queue_size": -1,
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
        "cooldowns": {
            "reminders": dict(state.cooldown.reminder_last_ts),
            "autonomous_checks": dict(state.cooldown.autonomous_check_last_ts),
        },
    }


def _action_result_to_dict(result: ActionResult) -> dict[str, object]:
    return {
        "action_type": result.action_type,
        "success": result.success,
        "timestamp": result.timestamp,
        "reason": result.reason,
        "payload": dict(result.payload),
    }
