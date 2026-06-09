from __future__ import annotations

"""事件分流器。

在状态归约之后，把事件分到五类处理方式之一，本身不修改状态：

- ``speech_llm``：``speech_recognized``，进入单次语音 LLM。
- ``behavior_distraction``：``system_triggered`` 且 trigger=behavior_distraction_check、
  source=agent_autonomy，进入玩手机分心专项检查 LLM。
- ``wellness_care``：``system_triggered`` 且 trigger=wellness_care_check、
  source=agent_autonomy，进入疲劳/情绪/姿态关怀检查（不含环境）。
- ``environment_care``：``system_triggered`` 且 trigger=environment_care_check、
  source=agent_autonomy，进入环境关怀检查（仅光照/温度/湿度/噪声）。
- ``sensor_status``：``system_triggered`` 且 trigger=sensor_status_report、
  source=agent_autonomy，进入每 5 分钟一次的传感器详细强制播报。
- ``rule``：focus_start_requested / focus_stop_requested / timer_finished，走规则。
- ``state_only``：其余事件（传感器 / 环境 / 语音生命周期 / TTS / timer_ticked 等），
  只更新 State / RuntimeHistory，永远不走 LLM。
"""

from dataclasses import dataclass

from src.agent.event.event_model import Event
from src.agent.policy_config import LLMRoutingPolicy

RouteKind = str  # "speech_llm" | "behavior_distraction" | "wellness_care" | "environment_care" | "sensor_status" | "rule" | "state_only"


@dataclass(frozen=True)
class RouteDecision:
    kind: RouteKind
    reason: str

    @property
    def uses_llm(self) -> bool:
        return self.kind in {
            "speech_llm",
            "behavior_distraction",
            "wellness_care",
            "environment_care",
        }


class EventRouter:
    """稳定、确定性的事件分流。"""

    def __init__(self, policy: LLMRoutingPolicy | None = None) -> None:
        self.policy = policy or LLMRoutingPolicy()

    def classify(self, event: Event) -> RouteDecision:
        event_type = str(event.type)

        if event_type == self.policy.speech_event:
            return RouteDecision("speech_llm", f"speech_input:{event_type}")

        if event_type == "system_triggered":
            trigger = str(event.payload.get("trigger", "")).strip()
            source = str(event.payload.get("source", "")).strip()
            if source == self.policy.trusted_source:
                if trigger == self.policy.behavior_distraction_trigger:
                    return RouteDecision(
                        "behavior_distraction",
                        f"behavior_distraction_check:{source}",
                    )
                if trigger == self.policy.wellness_care_trigger:
                    return RouteDecision("wellness_care", f"wellness_care_check:{source}")
                if trigger == self.policy.environment_care_trigger:
                    return RouteDecision("environment_care", f"environment_care_check:{source}")
                if trigger == self.policy.sensor_trigger:
                    return RouteDecision("sensor_status", f"sensor_status_report:{source}")
            return RouteDecision(
                "state_only",
                f"ignored_system_trigger:{trigger or 'unspecified'}",
            )

        if event_type in self.policy.rule_events:
            return RouteDecision("rule", f"structured_control:{event_type}")

        return RouteDecision("state_only", f"state_input_only:{event_type}")
