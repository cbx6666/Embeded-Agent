from __future__ import annotations

"""集中式个性化策略。

PersonalizationPolicy 把 UserProfile、Preference 和 Insight 转成 Planner/Realizer
可以直接使用的策略对象，避免各处散落 `if profile.xxx`。
"""

from dataclasses import dataclass, field

from src.services.user_profile_service import UserProfileService

DEFAULT_REST_REMINDER_MIN_FOCUS_SEC = 300
NIGHT_USER_REST_REMINDER_MIN_FOCUS_SEC = 600
FATIGUE_SENSITIVE_REST_REMINDER_MIN_FOCUS_SEC = 240


@dataclass(frozen=True)
class PersonalizedPolicy:
    """给 Planner/Realizer 使用的只读个性化策略快照。"""

    user_id: str
    reminder_tone: str | None = None
    user_name_prefix: str = ""
    break_activity: str | None = None
    tts_payload: dict[str, object] = field(default_factory=dict)
    info_context: str = "暂无明确资料"
    preference_context: str = "暂无明确偏好"
    rest_reminder_min_focus_sec: int = DEFAULT_REST_REMINDER_MIN_FOCUS_SEC
    reduce_night_rest_pressure: bool = False
    fatigue_sensitive: bool = False
    explanations: list[str] = field(default_factory=list)


class PersonalizationPolicy:
    """从长期 profile 生成个性化策略快照。"""

    def __init__(self, profile_service: UserProfileService) -> None:
        self.profile_service = profile_service

    def build(self, user_id: str | None) -> PersonalizedPolicy:
        """根据当前用户资料、偏好和画像生成策略。"""
        profile = self.profile_service.get_user(user_id)
        explanations: list[str] = []

        rest_threshold = DEFAULT_REST_REMINDER_MIN_FOCUS_SEC
        reduce_night_rest_pressure = False
        fatigue_sensitive = False

        for insight in profile.insights:
            if insight.confidence < 0.6:
                continue
            if insight.insight_type == "study_time_pattern" and "夜间学习" in insight.content:
                reduce_night_rest_pressure = True
                rest_threshold = max(rest_threshold, NIGHT_USER_REST_REMINDER_MIN_FOCUS_SEC)
                explanations.append("画像显示用户倾向夜间学习，夜间休息提醒更保守")
            if insight.insight_type == "fatigue_sensitivity":
                fatigue_sensitive = True
                rest_threshold = min(rest_threshold, FATIGUE_SENSITIVE_REST_REMINDER_MIN_FOCUS_SEC)
                explanations.append("画像显示用户较容易疲劳，提前触发疲劳提醒")

        return PersonalizedPolicy(
            user_id=profile.info.user_id,
            reminder_tone=profile.preference.reminder_style,
            user_name_prefix=self.profile_service.user_name_prefix(profile.info.user_id),
            break_activity=self.profile_service.preferred_break_activity(profile.info.user_id),
            tts_payload=self.profile_service.tts_payload(profile.info.user_id),
            info_context=self.profile_service.info_context(profile.info.user_id),
            preference_context=self.profile_service.preference_context(profile.info.user_id),
            rest_reminder_min_focus_sec=rest_threshold,
            reduce_night_rest_pressure=reduce_night_rest_pressure,
            fatigue_sensitive=fatigue_sensitive,
            explanations=explanations,
        )
