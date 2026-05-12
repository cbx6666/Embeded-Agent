from __future__ import annotations

"""显式 UserProfile 服务。

它是什么：
UserProfileService 是显式用户画像的唯一业务入口，负责用户创建/切换、资料更新、
偏好更新、展示渲染，以及为 PersonalContextBuilder 提供只读 profile 字典。

它不是什么：
它不是长期记忆管线，不接收 LLM 自由生成内容，不从模糊行为直接写入 profile。

为什么存在：
UserProfile 是 Authoritative Source。所有 display_name、age、hobbies、TTS 设置
和明确偏好，都必须从这里读取，避免同一事实在 LongTermMemory 中出现第二份真相。

边界：
UserProfileService 不依赖 LongTermMemoryPipeline；它只依赖 UserProfileStore。
"""

import time
from collections.abc import Callable
from dataclasses import asdict, fields
from typing import Any

from src.agent.profile.user_profile import UserInfo, UserPreference, UserProfile
from src.storage.user_profile_store import UserProfileStore

DEFAULT_USER_ID = "default"

LIST_INFO_KEYS = {"hobbies"}
INT_INFO_KEYS = {"age"}
STRING_INFO_KEYS = {"display_name", "gender", "identity"}
INFO_KEYS = LIST_INFO_KEYS | INT_INFO_KEYS | STRING_INFO_KEYS
INFO_RENDER_ORDER = ("display_name", "age", "gender", "identity", "hobbies")

LIST_PREFERENCE_KEYS = {"favorite_content_types", "favorite_music_styles", "disliked_topics"}
FLOAT_PREFERENCE_KEYS = {"tts_speed", "tts_volume"}
STRING_PREFERENCE_KEYS = {"reminder_style", "speech_style", "tts_voice"}
PREFERENCE_KEYS = LIST_PREFERENCE_KEYS | FLOAT_PREFERENCE_KEYS | STRING_PREFERENCE_KEYS
PREFERENCE_RENDER_ORDER = (
    "favorite_content_types",
    "favorite_music_styles",
    "reminder_style",
    "speech_style",
    "tts_voice",
    "tts_speed",
    "tts_volume",
    "disliked_topics",
)


class UserProfileService:
    """管理用户明确资料和显式偏好。"""

    def __init__(
        self,
        store: UserProfileStore,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.store = store
        self._now_fn = now_fn or time.time
        self.profiles = _profiles_from_raw(self.store.load_profiles())
        self.ensure_user(DEFAULT_USER_ID, display_name="default")

    def ensure_user(
        self,
        user_id: str | None,
        *,
        display_name: str | None = None,
        timestamp: float | None = None,
    ) -> UserProfile:
        """确保用户存在；只允许显式 display_name 更新。"""

        normalized_user_id = _normalize_user_id(user_id)
        now = self._now(timestamp)
        profile = self.profiles.get(normalized_user_id)
        if profile is None:
            profile = UserProfile(
                info=UserInfo(
                    user_id=normalized_user_id,
                    display_name=display_name or normalized_user_id,
                    created_at=now,
                    updated_at=now,
                    last_seen_at=now,
                )
            )
            self.profiles[normalized_user_id] = profile
            self._save_profiles()
            return profile

        if display_name and profile.info.display_name != display_name:
            profile.info.display_name = display_name
            self._mark_updated(profile, now)
            self._save_profiles()
        return profile

    def get_user(self, user_id: str | None) -> UserProfile:
        """读取用户；空 user_id 归一化为 default。"""

        return self.ensure_user(user_id)

    def ensure_user_id(self, user_id: str | None) -> str:
        """确保用户存在，并返回规范化 user_id。"""

        return self.ensure_user(user_id).info.user_id

    def list_users(self) -> list[UserProfile]:
        """按 user_id 排序返回所有用户。"""

        return [self.profiles[user_id] for user_id in sorted(self.profiles)]

    def touch_user(self, user_id: str | None, *, timestamp: float | None = None) -> str:
        """更新用户最近活跃时间；不写入任何推断信息。"""

        profile = self.ensure_user(user_id)
        self._mark_updated(profile, self._now(timestamp), last_seen=True)
        self._save_profiles()
        return profile.info.user_id

    def switch_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        timestamp: float | None = None,
    ) -> str:
        """切换到指定用户，并返回规范化 user_id。"""

        profile = self.ensure_user(user_id, display_name=display_name, timestamp=timestamp)
        self._mark_updated(profile, self._now(timestamp), last_seen=True)
        self._save_profiles()
        return profile.info.user_id

    def update_preference(
        self,
        user_id: str | None,
        key: str,
        raw_value: object,
        *,
        timestamp: float | None = None,
    ) -> str:
        """更新用户显式偏好字段。"""

        if key not in PREFERENCE_KEYS:
            raise ValueError(f"不支持的偏好字段: {key}")
        profile = self.ensure_user(user_id)
        setattr(profile.preference, key, _parse_preference_value(key, raw_value))
        self._mark_updated(profile, self._now(timestamp))
        self._save_profiles()
        return profile.info.user_id

    def update_info(
        self,
        user_id: str | None,
        key: str,
        raw_value: object,
        *,
        timestamp: float | None = None,
    ) -> str:
        """更新用户明确资料字段。"""

        if key not in INFO_KEYS:
            raise ValueError(f"不支持的用户资料字段: {key}")
        profile = self.ensure_user(user_id)
        setattr(profile.info, key, _parse_info_value(key, raw_value))
        self._mark_updated(profile, self._now(timestamp))
        self._save_profiles()
        return profile.info.user_id

    def profile_context(self, user_id: str | None) -> dict[str, Any]:
        """返回 PersonalContext 使用的权威 profile 字典。"""

        profile = self.ensure_user(user_id)
        return {
            "info": asdict(profile.info),
            "preference": asdict(profile.preference),
            "authoritative_source": "UserProfile",
        }

    def render_profile(self, user_id: str | None) -> str:
        """渲染当前用户显式画像，供 CLI 展示。"""

        profile = self.ensure_user(user_id)
        info = profile.info
        info_lines = _info_lines(info)
        pref_lines = _preference_lines(profile.preference)
        return "\n".join(
            [
                f"用户: {self.user_label(info.user_id)}",
                f"created_at: {info.created_at:.0f}",
                f"updated_at: {info.updated_at:.0f}",
                f"last_seen_at: {info.last_seen_at:.0f}",
                "资料:",
                *(info_lines or ["- 暂无"]),
                "偏好:",
                *(pref_lines or ["- 暂无"]),
            ]
        )

    def render_users(self, *, current_user_id: str | None = None) -> str:
        """渲染用户列表，当前用户用星号标记。"""

        lines = ["用户列表:"]
        for profile in self.list_users():
            marker = "*" if profile.info.user_id == current_user_id else "-"
            lines.append(f"{marker} {self.user_label(profile.info.user_id)}")
        return "\n".join(lines)

    def render_switch_result(self, user_id: str | None) -> str:
        return f"已切换到用户: {self.user_label(user_id)}"

    def render_preference_update_result(self, user_id: str | None, key: str) -> str:
        return f"已更新 {self.user_label(user_id)} 的偏好: {key}"

    def render_info_update_result(self, user_id: str | None, key: str) -> str:
        return f"已更新 {self.user_label(user_id)} 的资料: {key}"

    def info_context(self, user_id: str | None) -> str:
        """生成紧凑基础资料摘要，供 prompt 使用。"""

        lines = _info_lines(self.ensure_user(user_id).info)
        return "；".join(line.removeprefix("- ") for line in lines) if lines else "暂无明确资料"

    def preference_context(self, user_id: str | None) -> str:
        """生成紧凑偏好摘要，供 prompt 使用。"""

        lines = _preference_lines(self.ensure_user(user_id).preference)
        return "；".join(line.removeprefix("- ") for line in lines) if lines else "暂无明确偏好"

    def user_label(self, user_id: str | None) -> str:
        profile = self.ensure_user(user_id)
        display_name = profile.info.display_name or profile.info.user_id
        return f"{display_name} ({profile.info.user_id})"

    def user_name_prefix(self, user_id: str | None) -> str:
        profile = self.ensure_user(user_id)
        name = (profile.info.display_name or profile.info.user_id).strip()
        if not name or profile.info.user_id == DEFAULT_USER_ID:
            return ""
        return f"{name}: "

    def uses_gentle_reminder(self, user_id: str | None) -> bool:
        return self.ensure_user(user_id).preference.reminder_style == "gentle"

    def preferred_break_activity(self, user_id: str | None) -> str | None:
        preference = self.ensure_user(user_id).preference
        content_type = preference.favorite_content_types[0] if preference.favorite_content_types else ""
        music_style = preference.favorite_music_styles[0] if preference.favorite_music_styles else ""
        return music_style or content_type or None

    def tts_payload(self, user_id: str | None) -> dict[str, object]:
        preference = self.ensure_user(user_id).preference
        payload: dict[str, object] = {}
        if preference.tts_voice:
            payload["voice"] = preference.tts_voice
        if preference.tts_speed is not None:
            payload["speed"] = float(preference.tts_speed)
        if preference.tts_volume is not None:
            payload["volume"] = int(preference.tts_volume)
        return payload

    def _mark_updated(self, profile: UserProfile, timestamp: float, *, last_seen: bool = False) -> None:
        profile.info.updated_at = timestamp
        if last_seen:
            profile.info.last_seen_at = timestamp

    def _save_profiles(self) -> None:
        self.store.save_profiles({
            user_id: asdict(profile)
            for user_id, profile in self.profiles.items()
        })

    def _now(self, timestamp: float | None = None) -> float:
        return float(timestamp) if timestamp is not None else float(self._now_fn())


def _profiles_from_raw(raw_profiles: dict[str, dict[str, Any]]) -> dict[str, UserProfile]:
    profiles: dict[str, UserProfile] = {}
    for user_id, raw_profile in raw_profiles.items():
        try:
            profile = _profile_from_raw(raw_profile, fallback_user_id=user_id)
        except (TypeError, ValueError):
            continue
        profiles[profile.info.user_id] = profile
    return profiles


def _profile_from_raw(data: dict[str, Any], *, fallback_user_id: str) -> UserProfile:
    return UserProfile(
        info=_info_from_raw(data.get("info"), fallback_user_id=fallback_user_id),
        preference=_preference_from_raw(data.get("preference")),
    )


def _info_from_raw(data: object, *, fallback_user_id: str) -> UserInfo:
    values = _only_fields(data if isinstance(data, dict) else {}, UserInfo)
    values["user_id"] = _normalize_user_id(str(values.get("user_id") or fallback_user_id))
    values["hobbies"] = _normalize_list(values.get("hobbies"))
    if values.get("age") is not None:
        values["age"] = int(values["age"])
    return UserInfo(**values)


def _preference_from_raw(data: object) -> UserPreference:
    values = _only_fields(data if isinstance(data, dict) else {}, UserPreference)
    for key in LIST_PREFERENCE_KEYS:
        values[key] = _normalize_list(values.get(key))
    for key in FLOAT_PREFERENCE_KEYS:
        if values.get(key) is not None:
            values[key] = float(values[key])
    return UserPreference(**values)


def _normalize_user_id(user_id: str | None) -> str:
    normalized = str(user_id or "").strip()
    return normalized or DEFAULT_USER_ID


def _parse_preference_value(key: str, raw_value: object) -> object:
    if key in LIST_PREFERENCE_KEYS:
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        return [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if key in FLOAT_PREFERENCE_KEYS:
        return float(raw_value)
    return str(raw_value).strip() or None


def _parse_info_value(key: str, raw_value: object) -> object:
    if key in LIST_INFO_KEYS:
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        return [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if key in INT_INFO_KEYS:
        return int(raw_value)
    return str(raw_value).strip() or None


def _info_lines(info: UserInfo) -> list[str]:
    lines: list[str] = []
    for key in INFO_RENDER_ORDER:
        value = getattr(info, key)
        if value is None or value == []:
            continue
        rendered = ", ".join(value) if isinstance(value, list) else str(value)
        lines.append(f"- {key}: {rendered}")
    return lines


def _preference_lines(preference: UserPreference) -> list[str]:
    lines: list[str] = []
    for key in PREFERENCE_RENDER_ORDER:
        value = getattr(preference, key)
        if value is None or value == []:
            continue
        rendered = ", ".join(value) if isinstance(value, list) else str(value)
        lines.append(f"- {key}: {rendered}")
    return lines


def _only_fields(data: dict[str, Any], dc: type) -> dict[str, Any]:
    names = {field.name for field in fields(dc)}
    return {key: value for key, value in data.items() if key in names}


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    rendered = str(value).strip()
    return [rendered] if rendered else []
