from __future__ import annotations

"""把意图落地为真实可执行 Action。

只会产生五种动作：speak、display、start_timer、stop_timer、set_tts_volume。
不会再生成任何已删除动作（render_pet_expression / set_light_state /
start_voice_capture / stop_voice_capture / set_tts_voice / set_tts_speed）。
"""

from src.agent.action import action_builders as build
from src.agent.action.action_model import Action
from src.agent.core.models import Intent
from src.agent.policy_config import ActionPolicy

_REMINDER_INTENTS = {
    "suggest_rest": ("rest_reminder", "rest_reminder_text"),
    "offer_emotion_care": ("emotion_reminder", "emotion_care_text"),
    "remind_distraction": ("distraction_reminder", "distraction_reminder_text"),
    "adjust_environment_feedback": ("environment_warning", "environment_warning_text"),
    "suggest_media": ("media_suggestion", "media_suggestion_text"),
}

# reason -> 默认文案字段（用于 intent.payload 覆盖了 reason 的场景，如 posture_reminder）。
_REASON_TEXT_ATTR = {
    "rest_reminder": "rest_reminder_text",
    "joke_reminder": "joke_care_text",
    "emotion_reminder": "emotion_care_text",
    "posture_reminder": "posture_reminder_text",
    "distraction_reminder": "distraction_reminder_text",
    "environment_warning": "environment_warning_text",
    "media_suggestion": "media_suggestion_text",
}


class ActionRealizer:
    def __init__(self, policy: ActionPolicy | None = None) -> None:
        self.policy = policy or ActionPolicy()

    def realize(self, intents: list[Intent], *, reply_text: str = "") -> list[Action]:
        actions: list[Action] = []
        for intent in intents:
            actions.extend(self._realize_one(intent, reply_text))
        return actions

    def _realize_one(self, intent: Intent, reply_text: str) -> list[Action]:
        intent_type = intent.type
        text = (reply_text or "").strip()

        if intent_type == "answer_user":
            if not text:
                return []
            return [build.speak(text), build.display(text)]

        if intent_type == "start_focus":
            duration = self._clamp_duration(intent.payload.get("duration_sec"))
            minutes = max(1, round(duration / 60))
            shown = self.policy.focus_started_template.format(minutes=minutes)
            return [
                build.start_timer(duration),
                build.display(shown, kind="focus_mode"),
                build.speak(text or shown),
            ]

        if intent_type == "stop_focus":
            shown = text or self.policy.focus_stopped_text
            return [build.stop_timer(), build.display(shown, kind="idle"), build.speak(shown)]

        if intent_type == "complete_focus":
            shown = text or self.policy.focus_complete_text
            return [
                build.stop_timer(),
                build.speak(shown),
                build.display(shown, kind="idle"),
            ]

        if intent_type == "set_tts_volume":
            volume = self._clamp_volume(intent.payload.get("volume"))
            actions = [build.set_tts_volume(volume)]
            if text:
                actions.extend([build.speak(text), build.display(text)])
            return actions

        if intent_type == "play_media":
            return [
                build.play_media(
                    track_id=str(intent.payload.get("track_id", "")),
                    path=str(intent.payload.get("path", "")),
                    title=str(intent.payload.get("title", "")),
                    media_type=str(intent.payload.get("media_type", "")),
                    category=str(intent.payload.get("category", "")),
                    source=str(intent.payload.get("source", "user_explicit")),
                )
            ]

        if intent_type == "stop_media":
            return [build.stop_media(str(intent.payload.get("reason", "user")))]

        if intent_type == "pause_media":
            return [build.pause_media()]

        if intent_type == "resume_media":
            return [build.resume_media()]

        if intent_type == "next_media":
            return [build.next_media()]

        if intent_type in _REMINDER_INTENTS:
            default_reason, default_attr = _REMINDER_INTENTS[intent_type]
            # 允许决策层通过 intent.payload["reason"] 覆盖默认 reason，
            # 例如姿态不佳走 suggest_rest 意图、但 reason=posture_reminder。
            reason = str(intent.payload.get("reason") or default_reason)
            text_attr = _REASON_TEXT_ATTR.get(reason, default_attr)
            shown = text or getattr(self.policy, text_attr, getattr(self.policy, default_attr))
            return [
                build.speak(shown, kind="notification", reason=reason),
                build.display(shown, kind="notification", reason=reason),
            ]

        return []

    def _clamp_duration(self, value: object) -> int:
        try:
            duration = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            duration = self.policy.default_focus_duration_sec
        if duration <= 0:
            duration = self.policy.default_focus_duration_sec
        return max(self.policy.min_duration_sec, min(self.policy.max_duration_sec, duration))

    def _clamp_volume(self, value: object) -> int:
        try:
            volume = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            volume = self.policy.max_tts_volume
        return max(self.policy.min_tts_volume, min(self.policy.max_tts_volume, volume))
