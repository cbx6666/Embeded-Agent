from __future__ import annotations

"""AgentCore 单事件调度中枢。

事件主链路：

    Event
      -> reduce_state（更新 State）
      -> RuntimeHistoryService（更新短期历史）
      -> EventRouter（分流）
      -> speech_llm / behavior_distraction / wellness_care / environment_care
         / sensor_status / rule / state_only
      -> ActionRealizer 产出的 Action
      -> DeviceAdapter 执行
      -> 异步记忆抽取 + 持久化

LLM 入口：``speech_recognized``、``behavior_distraction_check``、``wellness_care_check``、
``environment_care_check``；``sensor_status_report`` 走确定性规则播报（不调 LLM、不检索
Memory）。结构化控制走规则；高频感知事件只更新 State / RuntimeHistory。
"""

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.action.realizer import ActionRealizer
from src.agent.context.memory_usage_hints import build_memory_usage_hints
from src.agent.core.models import Action, ActionResult, DecisionResult, Intent
from src.agent.decision.autonomous_check_meta import DEFERRED_OUTCOMES, STRONG_REVERT_TRIGGERS
from src.agent.decision.behavior_distraction_handler import BehaviorDistractionHandler
from src.agent.decision.environment_care_handler import EnvironmentCareHandler
from src.agent.decision.rule_handler import RuleHandler
from src.agent.decision.sensor_status_handler import SensorStatusHandler
from src.agent.decision.speech_llm_handler import SpeechLLMHandler
from src.agent.decision.wellness_care_handler import WellnessCareHandler
from src.agent.device.adapter import DeviceAdapter
from src.agent.event.event_model import Event
from src.agent.event.router import EventRouter
from src.agent.guard.guard import Guard
from src.agent.llm.client import LLMClient
from src.agent.llm.prompt_builder import build_focus_complete_prompt
from src.agent.memory.memory_service import MemoryService
from src.agent.media.media_controller import MediaController
from src.agent.policy_config import (
    BEHAVIOR_DISTRACTION_PRIORITY,
    ENVIRONMENT_CARE_PRIORITY,
    SENSOR_REPORT_PRIORITY,
    SPEECH_PRIORITY,
    WELLNESS_CARE_PRIORITY,
    ActionPolicy,
    BehaviorDistractionCheckPolicy,
    EnvironmentCareCheckPolicy,
    GuardPolicy,
    LLMRoutingPolicy,
    MediaPolicy,
    MemoryPolicy,
    SchedulePolicy,
    SensorReportPolicy,
    WellnessCareCheckPolicy,
)
from src.agent.scheduler.autonomous_scheduler import AutonomousScheduler
from src.agent.state.agent_state import AgentState
from src.agent.state.reducer import reduce_state
from src.agent.state.runtime_history import RuntimeHistoryService
from src.agent.core.event_ingress import AgentEventIngress
from src.agent.state.state_persistence import should_persist_runtime_state
from src.services.llm_service import LLMService
from src.services.timer_service import TimerService
from src.services.user_profile_service import UserProfileService
from src.storage.json_store import JsonStore
from src.storage.user_profile_store import UserProfileStore

_PROFILE_TOUCH_EVENTS = frozenset(
    {
        "speech_recognized",
        "focus_start_requested",
        "focus_stop_requested",
        "user_presence_updated",
        "user_attention_updated",
        "user_emotion_updated",
        "user_fatigue_updated",
        "user_posture_updated",
        "user_activity_updated",
    }
)

_PROFILE_TOUCH_PERSIST_IMMEDIATE = frozenset(
    {
        "speech_recognized",
        "focus_start_requested",
        "focus_stop_requested",
    }
)

_TRIM_EVERY_N_EVENTS = 4


class AgentCore:
    """单事件调度器，串联状态、分流、决策与执行。"""

    def __init__(
        self,
        *,
        output: object,
        timer_service: TimerService,
        store: JsonStore,
        llm_service: object,
        profile_service: UserProfileService,
        memory_service: MemoryService | None = None,
        runtime_history_service: RuntimeHistoryService | None = None,
        router: EventRouter | None = None,
        rule_handler: RuleHandler | None = None,
        speech_handler: SpeechLLMHandler | None = None,
        wellness_handler: WellnessCareHandler | None = None,
        environment_care_handler: EnvironmentCareHandler | None = None,
        behavior_distraction_handler: BehaviorDistractionHandler | None = None,
        sensor_handler: SensorStatusHandler | None = None,
        schedule_policy: SchedulePolicy | None = None,
        action_policy: ActionPolicy | None = None,
        guard_policy: GuardPolicy | None = None,
        llm_routing_policy: LLMRoutingPolicy | None = None,
        wellness_care_check_policy: WellnessCareCheckPolicy | None = None,
        environment_care_check_policy: EnvironmentCareCheckPolicy | None = None,
        behavior_distraction_check_policy: BehaviorDistractionCheckPolicy | None = None,
        sensor_report_policy: SensorReportPolicy | None = None,
        media_policy: MediaPolicy | None = None,
        media_controller: MediaController | None = None,
    ) -> None:
        self.output = output
        self.timer_service = timer_service
        self.store = store
        self.llm_service = llm_service
        self.llm_client = LLMClient(llm_service)
        self.profile_service = profile_service
        self.memory = memory_service or MemoryService()
        self.runtime_history_service = runtime_history_service or RuntimeHistoryService()

        action_policy = action_policy or ActionPolicy()
        realizer = ActionRealizer(action_policy)
        self.media_policy = media_policy or MediaPolicy()
        self.media_controller = media_controller or MediaController(
            music_root=self.media_policy.music_root,
            policy=self.media_policy,
        )
        guard = Guard(guard_policy or GuardPolicy())
        routing = llm_routing_policy or LLMRoutingPolicy()
        self.routing = routing
        self.wellness_care_check_policy = wellness_care_check_policy or WellnessCareCheckPolicy()
        self.environment_care_check_policy = (
            environment_care_check_policy or EnvironmentCareCheckPolicy()
        )
        self.behavior_distraction_check_policy = (
            behavior_distraction_check_policy or BehaviorDistractionCheckPolicy()
        )
        self.sensor_report_policy = sensor_report_policy or SensorReportPolicy()

        self.router = router or EventRouter(routing)
        self.rule_handler = rule_handler or RuleHandler(realizer=realizer)
        self.speech_handler = speech_handler or SpeechLLMHandler(
            realizer=realizer,
            policy=routing,
            media_controller=self.media_controller,
        )
        self.behavior_distraction_handler = behavior_distraction_handler or BehaviorDistractionHandler(
            realizer=realizer,
            guard=guard,
            policy=routing,
            check_policy=self.behavior_distraction_check_policy,
        )
        self.wellness_handler = wellness_handler or WellnessCareHandler(
            realizer=realizer,
            guard=guard,
            policy=routing,
            check_policy=self.wellness_care_check_policy,
            media_controller=self.media_controller,
        )
        self.environment_care_handler = environment_care_handler or EnvironmentCareHandler(
            realizer=realizer,
            guard=guard,
            policy=routing,
            check_policy=self.environment_care_check_policy,
        )
        self.sensor_handler = sensor_handler or SensorStatusHandler(
            policy=routing, sensor_policy=self.sensor_report_policy
        )
        self.schedule_policy = schedule_policy or SchedulePolicy()

        self.state = AgentState.from_dict(self.store.load_state_dict())
        self.state.current_user_id = self.profile_service.ensure_user_id(self.state.current_user_id)
        from src.agent.state.session_bootstrap import reset_ephemeral_session_on_cold_start

        if reset_ephemeral_session_on_cold_start(self.state):
            self.timer_service.stop()
            self.store.save_state(self.state)

        self.device_adapter = DeviceAdapter(
            output=self.output,
            timer_service=self.timer_service,
            timer_callback=self._on_timer_tick,
            media_controller=self.media_controller,
        )

        self._lock = threading.RLock()
        self._activity_lock = threading.Lock()
        self._active_priority: int | None = None
        self._event_handled_callback: Callable[[AgentState], None] | None = None
        self._last_persist_mono = 0.0
        self._persist_dirty = False
        self._trim_counter = 0
        self._event_ingress: AgentEventIngress | None = None
        self.last_intents: list[Intent] = []
        self.last_actions: list[Action] = []
        self.last_action_results: list[ActionResult] = []
        self.last_decision_result: DecisionResult | None = None

        self.autonomous_scheduler = AutonomousScheduler(
            state_provider=self._snapshot_state,
            event_sink=self.handle_event,
            config=self.schedule_policy,
            busy_priority_provider=self.current_activity_priority,
        )

    def _event_priority(self, event: Event) -> int | None:
        """返回事件对应的 LLM 任务优先级；高频 state_only / rule 事件返回 None。"""

        if event.type == "speech_recognized":
            return SPEECH_PRIORITY
        if event.type == "system_triggered" and str(event.payload.get("source", "")) == "agent_autonomy":
            trigger = str(event.payload.get("trigger", ""))
            if trigger == "behavior_distraction_check":
                return BEHAVIOR_DISTRACTION_PRIORITY
            if trigger == "wellness_care_check":
                return WELLNESS_CARE_PRIORITY
            if trigger == "environment_care_check":
                return ENVIRONMENT_CARE_PRIORITY
            if trigger == "sensor_status_report":
                return SENSOR_REPORT_PRIORITY
        return None

    def current_activity_priority(self) -> int | None:
        """当前正在运行的 LLM 任务优先级（供调度器做防打断 / 冻结判断）。"""

        with self._activity_lock:
            return self._active_priority

    # ---- 主链路 --------------------------------------------------------------
    def handle_event(self, event: Event) -> tuple[list[Action], list[ActionResult]]:
        ingress = self._event_ingress
        if ingress is not None:
            worker = ingress.worker_thread
            if worker is None or threading.current_thread() is not worker:
                llm_priority = self._event_priority(event)
                return ingress.submit(event, llm_priority=llm_priority)
        return self._process_event(event)

    def _process_event(self, event: Event) -> tuple[list[Action], list[ActionResult]]:
        priority = self._event_priority(event)
        if priority is not None:
            with self._activity_lock:
                self._active_priority = priority
        try:
            return self._handle_event_locked(event)
        finally:
            if priority is not None:
                with self._activity_lock:
                    self._active_priority = None

    def _handle_event_locked(self, event: Event) -> tuple[list[Action], list[ActionResult]]:
        with self._lock:
            route = self.router.classify(event)
            if route.kind == "rule":
                previous_state = AgentState.from_dict(self.state.to_dict())
            else:
                previous_state = self.state
            self.state = reduce_state(self.state, event)

            if event.type in _PROFILE_TOUCH_EVENTS:
                self.profile_service.touch_user(
                    self.state.current_user_id,
                    timestamp=event.timestamp,
                    persist=event.type in _PROFILE_TOUCH_PERSIST_IMMEDIATE,
                )
            self.runtime_history_service.record_event(self.state, event)

            if event.type == "speech_recognized":
                text = str(event.payload.get("text", "")).strip()
                if text:
                    self.runtime_history_service.record_message(
                        self.state, role="user", text=text, timestamp=event.timestamp
                    )
                    self.memory.submit_speech_memory(
                        self.state.current_user_id,
                        text,
                        event.timestamp,
                        recent_messages=list(self.state.runtime_history.recent_messages[-6:]),
                    )

            if event.type == "tts_finished":
                self._sync_reminder_from_tts_finished(event)
                self._sync_media_counter_from_tts_finished(event)

        # LLM / 记忆检索在锁外执行，避免阻塞 snapshot、感知入队与 TTS 回调以外的读者。
        decision = self._decide(route, event, previous_state)

        with self._lock:
            self._finalize_autonomous_check(event, decision)
            self._log_autonomous_check(event, decision)
            results = self._execute_actions(decision.actions, event.timestamp, source_event=event)
            self._finalize(decision, results, event=event)
            return decision.actions, results

    def _decide(self, route, event: Event, previous_state: AgentState) -> DecisionResult:
        if route.kind == "speech_llm":
            text = str(event.payload.get("text", ""))
            return self.speech_handler.decide(
                state=self.state,
                event=event,
                llm_client=self.llm_client,
                user_context=self._user_context(query=text, context_type="speech"),
            )

        if route.kind == "behavior_distraction":
            if self._behavior_distraction_on_cooldown(event.timestamp):
                return DecisionResult(
                    intents=[Intent("no_op", "behavior distraction check on cooldown")],
                    source="behavior_distraction",
                    reason="behavior distraction check on cooldown",
                )
            return self.behavior_distraction_handler.decide(
                state=self.state,
                event=event,
                llm_client=self.llm_client,
                user_context=self._user_context(
                    query=self._behavior_distraction_query(), context_type="behavior_distraction"
                ),
            )

        if route.kind == "wellness_care":
            if self._autonomous_check_on_cooldown("wellness_care_check", event.timestamp):
                return DecisionResult(
                    intents=[Intent("no_op", "wellness care check on cooldown")],
                    source="wellness_care",
                    reason="wellness care check on cooldown",
                )
            return self.wellness_handler.decide(
                state=self.state,
                event=event,
                llm_client=self.llm_client,
                user_context=self._user_context(
                    query=self._wellness_query(), context_type="wellness_care"
                ),
            )

        if route.kind == "environment_care":
            if self._autonomous_check_on_cooldown("environment_care_check", event.timestamp):
                return DecisionResult(
                    intents=[Intent("no_op", "environment care check on cooldown")],
                    source="environment_care",
                    reason="environment care check on cooldown",
                )
            return self.environment_care_handler.decide(
                state=self.state,
                event=event,
                llm_client=self.llm_client,
                user_context=self._user_context(
                    query=self._environment_care_query(), context_type="environment_care"
                ),
            )

        if route.kind == "sensor_status":
            return self.sensor_handler.decide(
                state=self.state,
                event=event,
                llm_client=self.llm_client,
            )

        if route.kind == "rule":
            decision = self.rule_handler.decide(
                event=event,
                previous_state=previous_state,
                current_state=self.state,
            )
            self._personalize_focus_complete(event, decision)
            return decision

        return DecisionResult(
            intents=[Intent("no_op", route.reason)],
            source="state_only",
            reason=route.reason,
        )

    _SKIP_REMINDER_TTS_SOURCES = frozenset({"wake_ack", "media_ack"})

    def _finalize_autonomous_check(self, event: Event, decision: DecisionResult) -> None:
        """自主检查决策后：按 handler 元数据决定是否写入准入冷却、是否回退调度周期。"""

        if event.type != "system_triggered":
            return
        if str(event.payload.get("source", "")) != "agent_autonomy":
            return
        trigger = str(event.payload.get("trigger", ""))
        if trigger not in self._AUTONOMOUS_CHECK_LABELS:
            return

        log_fields = dict(getattr(decision, "log_fields", {}) or {})
        deferred = bool(log_fields.get("deferred")) or (
            str(log_fields.get("final_action_reason") or "") in DEFERRED_OUTCOMES
        )

        should_mark = log_fields.get("should_mark_admitted")
        if should_mark is None:
            should_mark = not deferred

        admission_marked = False
        if should_mark:
            self._mark_autonomous_check_admitted(trigger, event.timestamp)
            admission_marked = True

        should_revert = log_fields.get("should_revert_schedule")
        if should_revert is None:
            should_revert = deferred and trigger in STRONG_REVERT_TRIGGERS

        schedule_reverted = False
        if should_revert:
            self.autonomous_scheduler.revert_emission(trigger)
            schedule_reverted = True

        log_fields["deferred"] = deferred
        log_fields["admission_marked"] = admission_marked
        log_fields["schedule_reverted"] = schedule_reverted
        if deferred and schedule_reverted:
            log_fields.setdefault("check_outcome", "deferred_revert")
        decision.log_fields = log_fields

    _AUTONOMOUS_CHECK_LABELS = {
        "behavior_distraction_check": "玩手机分心检查",
        "wellness_care_check": "疲劳/情绪关怀检查",
        "environment_care_check": "环境关怀检查",
        "sensor_status_report": "环境详细播报",
    }

    # 各自主检查的调度周期（秒），用于结构化日志的 period_sec 字段。
    _AUTONOMOUS_CHECK_PERIODS = {
        "behavior_distraction_check": 20,
        "wellness_care_check": 30,
        "environment_care_check": 60,
        "sensor_status_report": 300,
    }

    def _log_autonomous_check(self, event: Event, decision: DecisionResult) -> None:
        """每次自检都输出一行人类可读摘要 + 一行结构化 JSON，便于判断为何未播。"""

        if event.type != "system_triggered":
            return
        if str(event.payload.get("source", "")) != "agent_autonomy":
            return
        trigger = str(event.payload.get("trigger", ""))
        label = self._AUTONOMOUS_CHECK_LABELS.get(trigger, trigger)
        period_sec = self._AUTONOMOUS_CHECK_PERIODS.get(
            trigger, self.schedule_policy.interval_for(trigger)
        )
        actions = decision.actions or []
        speak_action_generated = any(getattr(action, "type", "") == "speak" for action in actions)
        action_generated = bool(actions)
        log_fields = dict(getattr(decision, "log_fields", {}) or {})
        deferred = bool(log_fields.get("deferred"))
        schedule_reverted = bool(log_fields.get("schedule_reverted"))
        reason = str(getattr(decision, "reason", "") or "")
        reply = str(getattr(decision, "reply_text", "") or "").strip()
        if speak_action_generated:
            outcome = "已生成播报动作"
        elif deferred and schedule_reverted:
            outcome = "延后（周期已回退）"
        elif deferred:
            outcome = "延后（等待空闲）"
        elif reply and "cooldown" in reason.lower():
            outcome = "关怀已生成但被冷却拦截"
        elif reply and "guard blocked" in reason:
            outcome = "文案已生成但未播报"
        else:
            outcome = "未触发"
        used_llm = "LLM" if getattr(decision, "used_llm", False) else "本地"

        # 结构化 JSON：trigger + period + handler 填的 log_fields + 实际结果。
        structured: dict[str, object] = {
            "trigger": trigger,
            "period_sec": period_sec,
        }
        structured.update(log_fields)
        structured["action_generated"] = action_generated
        structured["speak_action_generated"] = speak_action_generated
        structured["deferred"] = deferred
        structured.setdefault("defer_reason", log_fields.get("defer_reason"))
        structured["schedule_reverted"] = schedule_reverted
        structured.setdefault("admission_marked", log_fields.get("admission_marked"))
        structured["used_llm"] = bool(getattr(decision, "used_llm", False))
        if structured.get("final_action_reason") is None:
            structured["final_action_reason"] = self._infer_final_reason(
                decision, speak_action_generated, reason
            )

        extra = f"｜文案：{reply[:50]}" if reply and not speak_action_generated else ""
        line = f"[自检] {label}（{trigger}）→ {outcome}｜{used_llm}｜原因：{reason}{extra}"
        json_line = "[自检-JSON] " + json.dumps(structured, ensure_ascii=False)
        decision.log_fields = structured
        show = getattr(self.output, "show_text", None)
        if callable(show):
            show(line)
            show(json_line)
        else:
            print(line, flush=True)
            print(json_line, flush=True)

    @staticmethod
    def _infer_final_reason(decision: DecisionResult, speak_action_generated: bool, reason: str) -> str:
        """旧 handler 未填 final_action_reason 时，从 reason 文本归一化一个未播原因码。"""

        if speak_action_generated:
            return "speak_action_generated"
        lowered = reason.lower()
        if "cooldown" in lowered:
            return "cooldown"
        if "away" in lowered or "not present" in lowered:
            return "user_away"
        if "guard blocked" in lowered:
            return "guard_blocked"
        if "voice session" in lowered:
            return "voice_session_active_deferred"
        if "tts" in lowered or "speaking" in lowered:
            return "tts_speaking_deferred"
        if "no sensor" in lowered or "no environment" in lowered:
            return "no_environment_data"
        if "llm" in lowered:
            return "llm_failed"
        return "no_trigger"

    def _personalize_focus_complete(self, event: Event, decision: DecisionResult) -> None:
        """专注结束（complete_focus）时，用 LLM 生成一句带**轮换兴趣**的个性化关怀，覆盖默认文案。

        失败 / 空回复时保留 RuleHandler 给的默认完成文案，不影响停表等动作。
        """

        if event.type != "timer_finished":
            return
        intents = getattr(decision, "intents", None) or []
        if not any(getattr(i, "type", "") == "complete_focus" for i in intents):
            return
        spoken = [
            a
            for a in (decision.actions or [])
            if getattr(a, "type", "") in {"speak", "display"}
        ]
        if not spoken:
            return

        try:
            user_context = self._user_context(
                query=self._wellness_query(), context_type="wellness_care"
            )
            prompt = build_focus_complete_prompt(state=self.state, user_context=user_context)
            data = self.llm_client.complete_json(
                self.routing.focus_complete_prompt,
                prompt,
                temperature=self.routing.reply_temperature,
            )
            from src.agent.llm.reply_validator import normalize_reply, validate_tts_reply

            reply = normalize_reply(data.get("reply", ""))
            valid, _invalid_reason = validate_tts_reply(reply)
            if not valid:
                reply = ""
        except Exception:  # noqa: BLE001 - LLM 失败保留默认完成文案
            reply = ""
        if not reply:
            return

        for action in spoken:
            action.payload["text"] = reply
        try:
            decision.reply_text = reply
        except Exception:  # noqa: BLE001 - reply_text 不可写时忽略
            pass

    def _user_context(self, *, query: str, context_type: str) -> dict[str, object]:
        """组装供 LLM 使用的结构化用户上下文：profile + 显式偏好 + 多维记忆 + 最近互动 +
        本轮临时 memory_usage_hints（不落盘）。

        这是 profile / memory / runtime 进入 LLM 的**唯一**构造入口；各 handler 不再
        各自拼接，避免重复逻辑。
        """

        user_id = self.state.current_user_id
        retrieved = self.memory.retrieve_user_context(
            user_id, query=query, context_type=context_type
        )
        profile = self.profile_service.profile_context(user_id)
        info = profile.get("info", {}) if isinstance(profile, dict) else {}
        preference_map = profile.get("preference", {}) if isinstance(profile, dict) else {}
        by_type = retrieved.get("by_type", {})

        recent_messages = [
            {"role": item.get("role"), "text": item.get("text")}
            for item in list(self.state.runtime_history.recent_messages[-4:])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        recent_reminders = [
            {"reason": item.get("reason"), "timestamp": item.get("timestamp")}
            for item in list(self.state.runtime_history.reminder_records[-5:])
            if isinstance(item, dict)
        ]
        recent_interaction = {"recent_messages": recent_messages}
        runtime = {"recent_reminders": recent_reminders}
        current_state = self._current_state_brief()
        display_memories = self._memories_for_context(by_type, context_type=context_type)

        compact_info = _compact_mapping(info)
        compact_pref = _compact_mapping(preference_map)

        rotation_seed = int(self.state.interaction.care_rotation_index)
        hints = build_memory_usage_hints(
            profile=compact_info,
            preferences=compact_pref,
            memories=display_memories,
            recent_interaction=recent_interaction,
            runtime=runtime,
            context_type=context_type,
            current_state=current_state,
            rotation_seed=rotation_seed,
            user_query=query if context_type == "speech" else None,
        )
        # 本轮确实点缀了兴趣时，推进轮换游标，让下一次关怀换一个兴趣。
        if context_type == "wellness_care" and hints.get("personalization_candidates"):
            self.state.interaction.care_rotation_index = rotation_seed + 1

        context: dict[str, object] = {
            "profile": compact_info,
            "preferences": compact_pref,
            "memories": display_memories,
            "focus_session": self._focus_session_brief(),
            "recent_interaction": recent_interaction,
            "memory_usage_hints": hints,
        }
        self._log_memory_trace(
            context_type=context_type,
            query=query,
            retrieved_top=retrieved.get("top", []),
            hints=context["memory_usage_hints"],
        )
        return context

    def _log_memory_trace(
        self,
        *,
        context_type: str,
        query: str,
        retrieved_top: list,
        hints: dict,
    ) -> None:
        """每次 LLM 调用前，输出一行结构化记忆检索追溯，便于核对「这轮用了哪些记忆」。"""

        used = [
            {
                "type": item.get("type"),
                "content": str(item.get("content") or "")[:40],
                "confidence": item.get("confidence"),
            }
            for item in list(retrieved_top or [])
            if isinstance(item, dict)
        ]
        trace = {
            "context_type": context_type,
            "query": str(query or "")[:60],
            "retrieved_count": len(used),
            "retrieved": used,
            "recommended_angle": hints.get("recommended_angle") if isinstance(hints, dict) else None,
            "recommended_content": (
                (hints.get("recommended_content") or {}).get("label")
                if isinstance(hints, dict) and isinstance(hints.get("recommended_content"), dict)
                else None
            ),
            "personalization_level": (
                hints.get("personalization_level") if isinstance(hints, dict) else None
            ),
            "personalization_candidates_count": len(
                hints.get("personalization_candidates") or []
            )
            if isinstance(hints, dict)
            else 0,
            "personalization_candidates_preview": [
                str(c.get("label") or "")[:40]
                for c in (hints.get("personalization_candidates") or [])[:5]
                if isinstance(c, dict)
            ]
            if isinstance(hints, dict)
            else [],
        }
        line = "[记忆检索] " + json.dumps(trace, ensure_ascii=False)
        show = getattr(self.output, "show_text", None)
        if callable(show):
            show(line)
        else:
            print(line, flush=True)

    def _current_state_brief(self) -> dict[str, object]:
        """供 memory_usage_hints 推断 focus 的当前状态快照（疲劳/情绪/姿态/环境/是否在专注计时）。"""

        user = self.state.user
        env = self.state.environment
        return {
            "fatigue_level": user.fatigue_level,
            "emotion": user.emotion,
            "posture": user.posture,
            "attention": user.attention,
            "behavior": user.behavior,
            "focus_active": bool(self.state.focus.active),
            "light_level": env.light_level,
            "noise_level": env.noise_level,
            "temperature_level": env.temperature_level,
            "humidity_level": env.humidity_level,
        }

    def _focus_session_brief(self) -> dict[str, object]:
        """供 LLM 明确区分「正在专注计时」与「普通状态」。"""

        focus = self.state.focus
        if not focus.active:
            return {"active": False}
        remaining = self.compute_focus_remaining_sec()
        return {
            "active": True,
            "remaining_sec": remaining,
            "remaining_minutes": round(remaining / 60, 1) if remaining else 0,
            "target_duration_sec": focus.target_duration_sec,
            "elapsed_sec": focus.elapsed_sec,
        }

    def _memories_for_context(self, by_type: dict, *, context_type: str) -> dict:
        """未在专注计时时，wellness 关怀不向 LLM 暴露 habit/work_style（多为专注习惯），避免误提计时。"""

        if context_type != "wellness_care" or self.state.focus.active:
            return by_type
        filtered: dict = {}
        for key, items in by_type.items():
            if key in {"habits", "work_styles"}:
                continue
            filtered[key] = items
        return filtered

    def _behavior_distraction_query(self) -> str:
        """分心检索 query：场景语义词（分心/专注/学习/工作/打断/沉浸）+ 当前行为状态。"""

        parts = ["分心 专注 学习 工作 打断 沉浸"]
        for value in (
            str(self.state.user.behavior or ""),
            str(self.state.user.attention or ""),
            str(self.state.user.current_activity or ""),
        ):
            if value and value not in {"none", "neutral", "idle", "unknown", "working"}:
                parts.append(value)
        return " ".join(parts)

    def _wellness_query(self) -> str:
        """疲劳/情绪/姿态关怀检索 query：只用**场景语义词**（疲劳/情绪/休息/放松/恢复/
        姿态等），不写死任何具体爱好；具体偏好由检索按 type 权重自然命中。
        """

        parts = ["疲劳 累 休息 放松 恢复 换脑 情绪 安慰 陪伴 坐姿 活动"]
        if self.state.focus.active:
            parts.append("学习状态")
        fatigue = str(self.state.user.fatigue_level or "")
        emotion = str(self.state.user.emotion or "")
        posture = str(self.state.user.posture or "")
        for value in (fatigue, emotion, posture):
            if value and value not in {"none", "neutral", "idle", "unknown"}:
                parts.append(value)
        return " ".join(parts)

    def _environment_care_query(self) -> str:
        """环境关怀检索 query：只用环境相关场景语义词（光照/温度/湿度/噪声/学习环境/打扰偏好）。"""

        parts = ["光照 温度 湿度 噪声 学习环境 是否喜欢被打扰"]
        for level in (
            self.state.environment.light_level,
            self.state.environment.noise_level,
            self.state.environment.temperature_level,
            self.state.environment.humidity_level,
        ):
            if str(level or "").lower() in {"high", "low", "dark", "dim", "hot", "cold", "noisy", "loud", "humid", "dry"}:
                parts.append(str(level))
        return " ".join(parts)

    # ---- 执行 ----------------------------------------------------------------
    def _execute_actions(
        self, actions: list[Action], action_ts: int, *, source_event: Event
    ) -> list[ActionResult]:
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

        if action.type in {"speak", "display"}:
            text = str(action.payload.get("text", "")).strip()
            if text:
                role = "agent" if action.type == "speak" else "display"
                self.runtime_history_service.record_message(
                    self.state, role=role, text=text, timestamp=action_ts
                )
            self.state.interaction.last_agent_response_time = action_ts
            # speak 的 dialogue_state 由 tts_started/tts_finished 事件驱动，勿在此提前置 idle。
            if action.type == "display":
                self.state.interaction.dialogue_state = "idle"

        if action.type == "speak" and action.payload.get("kind") == "notification":
            reason = str(action.payload.get("reason") or "")
            # reminder_last_ts / 媒体计数均在真实 TTS 播完后（tts_finished）更新；
            # 无语音链路（纯文本 / 测试）时在此兜底，避免 Guard 永远认为未提醒过。
            if self.device_adapter.voice_runtime is None:
                if reason and reason != "media_suggestion":
                    self.state.cooldown.reminder_last_ts[reason] = action_ts
                if reason:
                    self._sync_media_suggestion_counter(reason)

    def _sync_reminder_from_tts_finished(self, event: Event) -> None:
        """仅在语音真实播完后写入 reminder_last_ts；cancel / 非自主提醒不计入。"""

        from src.adapters.voice.arbitration.tts_job_policy import is_autonomous_spec, resolve_job_spec

        if event.payload.get("cancelled"):
            return
        source = str(event.payload.get("source") or "")
        if source in self._SKIP_REMINDER_TTS_SOURCES:
            return
        kind = str(event.payload.get("kind") or "")
        reason = str(event.payload.get("reason") or "")
        if not reason or reason in {"status_report", "media_suggestion"}:
            return
        spec = resolve_job_spec(source=source, reason=reason, kind=kind)
        if not is_autonomous_spec(spec):
            return
        self.state.cooldown.reminder_last_ts[reason] = int(event.timestamp)

    def _sync_media_counter_from_tts_finished(self, event: Event) -> None:
        """仅在语音真实播完后更新媒体询问间隔计数；cancel 不计入。"""

        if event.payload.get("cancelled"):
            return
        if str(event.payload.get("kind") or "") != "notification":
            return
        reason = str(event.payload.get("reason") or "")
        if reason:
            self._sync_media_suggestion_counter(reason)

    def _sync_media_suggestion_counter(self, reason: str) -> None:
        """媒体询问：首次可问；之后每 2 次纯 wellness 播报后才可再问。"""

        cd = self.state.cooldown
        if reason == "media_suggestion":
            cd.media_suggestion_ever_asked = True
            cd.wellness_cares_since_media_ask = 0
            return
        if reason in {"rest_reminder", "emotion_reminder", "posture_reminder"}:
            if cd.media_suggestion_ever_asked:
                cd.wellness_cares_since_media_ask = int(cd.wellness_cares_since_media_ask) + 1

    def _sync_focus_state_from_timer_action(
        self, action: Action, action_ts: int, *, source_event: Event
    ) -> None:
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
        elif action.type == "stop_timer":
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

    def _finalize(
        self,
        decision: DecisionResult,
        results: list[ActionResult],
        *,
        event: Event,
    ) -> None:
        self.last_intents = decision.intents
        self.last_actions = decision.actions
        self.last_action_results = results
        self.last_decision_result = decision
        self._trim_counter += 1
        if self._trim_counter % _TRIM_EVERY_N_EVENTS == 0:
            self.runtime_history_service.trim(self.state)
        self._maybe_persist_state(event, decision)
        if self._event_handled_callback is not None:
            try:
                self._event_handled_callback(self.state)
            except Exception:
                pass

    # ---- 周期检查冷却 --------------------------------------------------------
    def _autonomous_check_on_cooldown(self, trigger: str, timestamp: int) -> bool:
        last_ts = self.state.cooldown.autonomous_check_last_ts.get(trigger)
        if last_ts is None:
            return False
        try:
            elapsed = int(timestamp) - int(last_ts)
        except (TypeError, ValueError):
            return False
        # 持久化时间戳超前或系统时钟回拨时 elapsed 为负，不应永久锁死检查。
        if elapsed < 0:
            return False
        cooldown = self.schedule_policy.interval_for(trigger) or 30
        return elapsed < cooldown

    def _mark_autonomous_check_admitted(self, trigger: str, timestamp: int) -> None:
        self.state.cooldown.autonomous_check_last_ts[trigger] = int(timestamp)

    def _behavior_distraction_on_cooldown(self, timestamp: int) -> bool:
        return self._autonomous_check_on_cooldown("behavior_distraction_check", timestamp)

    def _mark_behavior_distraction_admitted(self, timestamp: int) -> None:
        self._mark_autonomous_check_admitted("behavior_distraction_check", timestamp)

    # ---- 计时器回调 ----------------------------------------------------------
    def _compute_focus_remaining_locked(self) -> int:
        """按 start_ts + target_duration 用墙钟推算剩余秒数（恢复会话时比持久化 remaining 更准）。"""

        focus = self.state.focus
        if not focus.active:
            return 0
        try:
            target = int(focus.target_duration_sec or 0)
            start_ts = int(focus.start_ts or 0)
        except (TypeError, ValueError):
            return max(0, int(focus.remaining_sec or 0))
        if target > 0 and start_ts > 0:
            return max(0, target - (int(time.time()) - start_ts))
        return max(0, int(focus.remaining_sec or 0))

    def compute_focus_remaining_sec(self) -> int:
        with self._lock:
            return self._compute_focus_remaining_locked()

    def resume_focus_timer_if_needed(self) -> int | None:
        """进程重启后恢复专注倒计时后台线程（状态里 focus.active 但 TimerService 默认未启动）。"""

        with self._lock:
            if not self.state.focus.active:
                return None
            remaining = self._compute_focus_remaining_locked()
            self.state.focus.remaining_sec = remaining
            self.store.save_state(self.state)
        if remaining <= 0:
            self.handle_event(
                Event(
                    type="timer_finished",
                    timestamp=int(time.time()),
                    payload={"remaining_sec": 0, "timer": "focus"},
                )
            )
            return 0
        self.timer_service.start(remaining, self.device_adapter.timer_callback)
        return remaining

    def _maybe_persist_state(
        self,
        event: Event,
        decision: DecisionResult,
        *,
        force: bool = False,
    ) -> None:
        if force or should_persist_runtime_state(
            event,
            decision=decision,
            last_persist_mono=self._last_persist_mono,
        ):
            self.store.save_state(self.state)
            self._last_persist_mono = time.monotonic()
            self._persist_dirty = False
        else:
            self._persist_dirty = True

    def _on_timer_tick(self, remaining_sec: int) -> None:
        ts = int(time.time())
        if remaining_sec <= 0:
            self.handle_event(
                Event(
                    type="timer_finished",
                    timestamp=ts,
                    payload={"remaining_sec": 0, "timer": "focus"},
                )
            )
            return
        # 轻量更新：不走 LLM 主链路，且锁忙时跳过，避免倒计时被语音/自检卡住。
        if not self._lock.acquire(blocking=False):
            return
        try:
            if not self.state.focus.active or self.state.focus.start_ts is None:
                return
            self.state = reduce_state(
                self.state,
                Event(
                    type="timer_ticked",
                    timestamp=ts,
                    payload={"remaining_sec": remaining_sec, "timer": "focus"},
                ),
            )
            noop = DecisionResult(intents=[Intent("no_op", "timer tick")])
            self._maybe_persist_state(
                Event(
                    type="timer_ticked",
                    timestamp=ts,
                    payload={"remaining_sec": remaining_sec, "timer": "focus"},
                ),
                noop,
            )
            if self._event_handled_callback is not None:
                try:
                    self._event_handled_callback(self.state)
                except Exception:
                    pass
        finally:
            self._lock.release()

    # ---- 生命周期 ------------------------------------------------------------
    def start_autonomous_scheduler(self) -> None:
        if self._event_ingress is None:
            self._event_ingress = AgentEventIngress(self._process_event)
            self._event_ingress.start()
        self.autonomous_scheduler.start()

    def set_event_handled_callback(self, callback: Callable[[AgentState], None] | None) -> None:
        self._event_handled_callback = callback

    def shutdown(self) -> None:
        self.autonomous_scheduler.stop()
        if self._event_ingress is not None:
            self._event_ingress.stop()
            self._event_ingress = None
        with self._lock:
            self.profile_service.flush_profiles()
            self.memory.shutdown(timeout=5.0)
            self.timer_service.stop()
            self.store.save_state(self.state)
            self._persist_dirty = False

    def _snapshot_state(self) -> AgentState:
        with self._lock:
            return AgentState.from_dict(self.state.to_dict())

    # ---- 渲染 / 用户资料 -----------------------------------------------------
    def render_state(self) -> str:
        with self._lock:
            return json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2)

    def render_history(self) -> str:
        with self._lock:
            return json.dumps(self.state.runtime_history.to_decision_dict(), ensure_ascii=False, indent=2)

    def render_profile(self) -> str:
        with self._lock:
            return self.profile_service.render_profile(self.state.current_user_id)

    def render_users(self) -> str:
        with self._lock:
            return self.profile_service.render_users(current_user_id=self.state.current_user_id)

    def switch_user(self, user_id: str, *, display_name: str | None = None, timestamp: int | None = None) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self.profile_service.switch_user(user_id, display_name=display_name, timestamp=ts)
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self.profile_service.render_switch_result(user_id)

    def set_user_preference(self, key: str, value: object, *, timestamp: int | None = None) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self.profile_service.update_preference(self.state.current_user_id, key, value, timestamp=ts)
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self.profile_service.render_preference_update_result(user_id, key)

    def set_user_info(self, key: str, value: object, *, timestamp: int | None = None) -> str:
        with self._lock:
            ts = int(time.time()) if timestamp is None else int(timestamp)
            user_id = self.profile_service.update_info(self.state.current_user_id, key, value, timestamp=ts)
            self.state.current_user_id = user_id
            self.store.save_state(self.state)
            return self.profile_service.render_info_update_result(user_id, key)


def _compact_mapping(mapping: object) -> dict[str, object]:
    """过滤掉空值，返回紧凑的 profile / preference 字典供 prompt 使用。"""

    if not isinstance(mapping, dict):
        return {}
    compact: dict[str, object] = {}
    for key, value in mapping.items():
        if value in (None, [], "", {}):
            continue
        compact[str(key)] = value
    return compact


def build_default_core(
    *,
    output: object | None = None,
    store_path: str | Path = "data/runtime/runtime_store.json",
    profile_store_path: str | Path = "data/user/user_profiles.json",
    memory_store_path: str | Path = "data/memory/user_memory.json",
    timer_background: bool = True,
    llm_service: object | None = None,
    memory_async: bool = True,
    schedule_policy: SchedulePolicy | None = None,
    action_policy: ActionPolicy | None = None,
    guard_policy: GuardPolicy | None = None,
    llm_routing_policy: LLMRoutingPolicy | None = None,
    wellness_care_check_policy: WellnessCareCheckPolicy | None = None,
    environment_care_check_policy: EnvironmentCareCheckPolicy | None = None,
    behavior_distraction_check_policy: BehaviorDistractionCheckPolicy | None = None,
    sensor_report_policy: SensorReportPolicy | None = None,
    media_policy: MediaPolicy | None = None,
) -> AgentCore:
    profile_service = UserProfileService(UserProfileStore(profile_store_path))
    resolved_llm = llm_service or LLMService()
    memory_service = MemoryService(
        MemoryPolicy(store_path=str(memory_store_path), async_write=memory_async),
        llm_client=LLMClient(resolved_llm),
    )
    return AgentCore(
        output=output or ConsoleOutput(),
        timer_service=TimerService(background=timer_background),
        store=JsonStore(store_path),
        llm_service=resolved_llm,
        profile_service=profile_service,
        memory_service=memory_service,
        schedule_policy=schedule_policy,
        action_policy=action_policy,
        guard_policy=guard_policy,
        llm_routing_policy=llm_routing_policy,
        wellness_care_check_policy=wellness_care_check_policy,
        environment_care_check_policy=environment_care_check_policy,
        behavior_distraction_check_policy=behavior_distraction_check_policy,
        sensor_report_policy=sensor_report_policy,
        media_policy=media_policy,
    )
