from __future__ import annotations

"""sensor_status_report 传感器状态播报处理器。

每 5 分钟一次的低优先级播报：用**确定性**方式把当前所有传感器读数（温度 / 湿度 /
光照 / 噪声）逐项、带准确数值地播报出来，区别于疲劳/情绪/环境关怀的主动关怀提醒。

之所以不走 LLM：环境播报的核心是「准确的数值」，LLM 容易漏读或改写数值；这里直接由
具体读数拼装播报文本，保证温湿度光照等信息每次都完整、准确。

确定性边界：
- 用户不在场 → no_op（调度器会尽快重试，不消耗 300s 周期）。
- 语音会话进行中（dialogue_state 为 listening/thinking）→ no_op（调度器会尽快重试）。
- TTS 正在播放（dialogue_state == speaking）及其他情况 → 仍产出 speak，由 VoiceRuntime 入 TTS 队列。
- 不因其他关怀提醒刚播过而延后（否则 30s wellness 会让 60s gap 永远不满足）。
- 没有任何可用读数 → no_op。

允许的动作只有 speak / display / no_op；播报用 kind="status_report"，不写提醒冷却。
"""

from src.adapters.voice.arbitration.session_probe import should_defer_autonomous_speak
from src.agent.decision.autonomous_check_meta import apply_defer_metadata
from src.agent.action import action_builders as build
from src.agent.core.models import DecisionResult, Event, Intent
from src.agent.llm.client import LLMClient
from src.agent.policy_config import LLMRoutingPolicy, SensorReportPolicy
from src.agent.state.agent_state import AgentState
from src.agent.state.summary_builder import build_sensor_status_summary


class SensorStatusHandler:
    def __init__(
        self,
        *,
        policy: LLMRoutingPolicy | None = None,
        sensor_policy: SensorReportPolicy | None = None,
    ) -> None:
        self.policy = policy or LLMRoutingPolicy()
        self.sensor_policy = sensor_policy or SensorReportPolicy()

    def decide(
        self,
        *,
        state: AgentState,
        event: Event,
        llm_client: LLMClient,
    ) -> DecisionResult:
        del llm_client  # 环境播报为确定性，不调用 LLM，保证数值完整准确。
        has_data = self._has_environment_data(state)
        log = {
            "force_report": True,
            "has_environment_data": has_data,
            "cooldown_result": "pass",
            "guard_result": "pass",
            "final_action_reason": None,
        }

        block_reason, block_outcome, defer_reason = self._deterministic_block(state, event.timestamp)
        if block_reason:
            if block_outcome in {"voice_session_active_deferred"}:
                apply_defer_metadata(
                    log,
                    outcome=block_outcome,
                    defer_reason=defer_reason or "voice_session_active",
                    trigger="sensor_status_report",
                )
            else:
                # away / 无数据：业务 no_op，正常消耗周期，不回退调度。
                log["final_action_reason"] = block_outcome
            return DecisionResult(
                intents=[Intent("no_op", block_reason)],
                source="sensor_status_report",
                reason=block_reason,
                log_fields=log,
            )

        summary = build_sensor_status_summary(state, check_time=event.timestamp)
        reply = self._format_broadcast(summary)
        if not reply:
            log["final_action_reason"] = "no_environment_data"
            return DecisionResult(
                intents=[Intent("no_op", "no sensor readings available")],
                source="sensor_status_report",
                reason="no sensor readings available",
                log_fields=log,
            )

        log["final_action_reason"] = "status_report"
        actions = [
            build.speak(reply, kind="status_report", reason="status_report"),
            build.display(reply, kind="status_report", reason="status_report"),
        ]
        return DecisionResult(
            intents=[Intent("report_sensor_status", "sensor_status_report deterministic")],
            actions=actions,
            source="sensor_status_report",
            reason="sensor status reported",
            reply_text=reply,
            log_fields=log,
        )

    @staticmethod
    def _has_environment_data(state: AgentState) -> bool:
        env = state.environment
        return any(
            v is not None
            for v in (env.light_lux, env.temperature_c, env.humidity_pct, env.noise_db)
        )

    @staticmethod
    def _format_broadcast(summary: dict) -> str:
        """把所有可用读数拼成一句完整、准确的环境播报；无任何读数时返回空串。"""

        readings: list[str] = []
        temp = summary.get("temperature_c")
        if temp is not None:
            readings.append(f"温度 {float(temp):.1f}℃")
        humidity = summary.get("humidity_pct")
        if humidity is not None:
            readings.append(f"湿度 {float(humidity):.0f}%")
        light = summary.get("light_lux")
        if light is not None:
            readings.append(f"光照 {int(round(float(light)))} lux")
        noise = summary.get("noise_db")
        if noise is not None:
            readings.append(f"噪声 {int(round(float(noise)))} 分贝")

        if not readings:
            return ""

        text = "当前环境：" + "，".join(readings) + "。"

        abnormal = summary.get("abnormal_items") or []
        if isinstance(abnormal, list) and abnormal:
            hints = {
                "temperature": "温度偏离舒适区，注意通风或调温",
                "humidity": "湿度不太合适，可适当加湿或除湿",
                "light": "光照不太合适，调整一下会更护眼",
                "noise": "噪声偏高，可能影响专注",
            }
            top = abnormal[0]
            hint = hints.get(str(top.get("type", "")))
            if hint:
                text += hint + "。"
        return text

    def _deterministic_block(
        self, state: AgentState, timestamp: int
    ) -> tuple[str | None, str | None, str | None]:
        if self.sensor_policy.block_when_away and state.user.presence == "away":
            return "user is away; sensor report skipped", "user_away", None
        if should_defer_autonomous_speak(dialogue_state=state.interaction.dialogue_state):
            return (
                "voice session active; sensor report deferred",
                "voice_session_active_deferred",
                "voice_session_active",
            )
        return None, None, None
