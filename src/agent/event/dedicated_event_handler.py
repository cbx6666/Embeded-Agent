from __future__ import annotations

"""P4 profile/settings 事件的确定性处理器。"""

from dataclasses import dataclass

from src.agent.config.policy_config import DedicatedEventPolicyConfig
from src.agent.event.event_model import Event
from src.agent.state import AgentState
from src.services.user_profile_service import UserProfileService


@dataclass(frozen=True)
class DedicatedEventResult:
    handled: bool
    reason: str
    updated_field: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "handled": self.handled,
            "reason": self.reason,
            "updated_field": self.updated_field,
        }


class DedicatedEventHandler:
    """处理无需 DecisionPipeline 的显式 profile/settings 事件。"""

    def __init__(
        self,
        profile_service: UserProfileService,
        *,
        policy_config: DedicatedEventPolicyConfig | None = None,
    ) -> None:
        self.profile_service = profile_service
        self.config = policy_config or DedicatedEventPolicyConfig()

    def handle(self, *, event: Event, state: AgentState) -> DedicatedEventResult:
        event_type = str(event.type)
        if event_type == "user_switched":
            user_id = str(event.payload.get("user_id") or state.current_user_id)
            state.current_user_id = self.profile_service.switch_user(
                user_id,
                display_name=_optional_text(event.payload.get("display_name")),
                timestamp=event.timestamp,
            )
            return DedicatedEventResult(True, "user_switched", "current_user_id")

        if event_type in {"user_profile_updated", "user_preference_update_requested"}:
            return self._handle_profile_update(event, state)

        mapping = self.config.voice_preference_fields.get(event_type)
        if mapping is not None:
            preference_key, payload_keys = mapping
            value = _first_value(event, payload_keys)
            if value is None:
                return DedicatedEventResult(False, "settings_value_missing")
            self.profile_service.update_preference(
                state.current_user_id,
                preference_key,
                value,
                timestamp=event.timestamp,
            )
            return DedicatedEventResult(True, "voice_setting_updated", preference_key)
        return DedicatedEventResult(False, "no_dedicated_handler")

    def _handle_profile_update(
        self,
        event: Event,
        state: AgentState,
    ) -> DedicatedEventResult:
        section = str(event.payload.get("section", "preference")).strip().lower()
        key = str(
            event.payload.get("key")
            or event.payload.get("preference_key")
            or event.payload.get("info_key")
            or ""
        ).strip()
        if not key or "value" not in event.payload:
            return DedicatedEventResult(False, "profile_update_fields_missing")
        value = event.payload.get("value")
        try:
            if section == "info" or event.payload.get("info_key") is not None:
                self.profile_service.update_info(
                    state.current_user_id,
                    key,
                    value,
                    timestamp=event.timestamp,
                )
                return DedicatedEventResult(True, "profile_info_updated", key)
            self.profile_service.update_preference(
                state.current_user_id,
                key,
                value,
                timestamp=event.timestamp,
            )
            return DedicatedEventResult(True, "profile_preference_updated", key)
        except (TypeError, ValueError) as exc:
            return DedicatedEventResult(False, f"profile_update_rejected:{exc}")


def _first_value(event: Event, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in event.payload and event.payload.get(key) is not None:
            return event.payload.get(key)
    return None


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
