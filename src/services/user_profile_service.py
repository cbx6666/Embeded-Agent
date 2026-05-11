from __future__ import annotations

"""长期用户画像与偏好服务。

UserProfileService 是 profile 子系统唯一业务入口：
- 负责默认用户创建和用户切换；
- 负责把 JSON 原始数据转换为 profile dataclass；
- 负责显式偏好的解析、更新和展示；
- 负责画像 insight 的合并更新；
- 负责给决策/落地层提供只读的个性化查询。

它不参与 Agent 的实时状态归约，也不让 LLM 直接写入长期 profile。
"""

import time
from collections.abc import Callable
from dataclasses import asdict, fields
from typing import Any

from src.agent.state.user_profile_state import (
    UserInfo,
    UserPreference,
    UserProfile,
    UserProfileInsight,
)
from src.storage.profile_store import ProfileStore

DEFAULT_USER_ID = "default"

# 基础资料字段：表达“用户是谁”，和“偏好某种服务方式”的 preference 分开管理。
LIST_INFO_KEYS = {"hobbies"}
INT_INFO_KEYS = {"age"}
STRING_INFO_KEYS = {"display_name", "gender", "identity"}
INFO_KEYS = LIST_INFO_KEYS | INT_INFO_KEYS | STRING_INFO_KEYS

INFO_RENDER_ORDER = (
    "display_name",
    "age",
    "gender",
    "identity",
    "hobbies",
)

# 列表型偏好：命令行里用逗号分隔，保存时统一转成 list[str]。
LIST_PREFERENCE_KEYS = {
    "favorite_content_types",
    "favorite_music_styles",
    "disliked_topics",
}

# 数值型偏好：目前主要用于 TTS 参数。
FLOAT_PREFERENCE_KEYS = {"tts_speed", "tts_volume"}

# 文本型偏好：用于提醒话术、说话风格和音色选择。
STRING_PREFERENCE_KEYS = {"reminder_style", "speech_style", "tts_voice"}
PREFERENCE_KEYS = LIST_PREFERENCE_KEYS | FLOAT_PREFERENCE_KEYS | STRING_PREFERENCE_KEYS

# 显式渲染顺序比 fields() 更稳定，也更适合新人快速理解哪些偏好是支持的。
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
    """管理长期用户、显式偏好和画像 insight。"""

    def __init__(
        self,
        store: ProfileStore,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        # now_fn 主要用于测试，避免断言依赖真实时间。
        self.store = store
        self._now_fn = now_fn or time.time
        self.profiles = _profiles_from_raw(self.store.load_profiles())
        # 系统启动时保证 default 用户存在，让旧流程不需要先显式建用户。
        self.ensure_user(DEFAULT_USER_ID, display_name="default")

    def ensure_user(
        self,
        user_id: str | None,
        *,
        display_name: str | None = None,
        timestamp: float | None = None,
    ) -> UserProfile:
        """确保用户存在；不存在就创建，存在则按需更新展示名。"""
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

        # 展示名允许由显式切换命令后补；其他字段不在这里做隐式推断。
        if display_name and profile.info.display_name != display_name:
            profile.info.display_name = display_name
            self._mark_updated(profile, now)
            self._save_profiles()
        return profile

    def get_user(self, user_id: str | None) -> UserProfile:
        """读取用户；如果 user_id 为空，则回退到 default。"""
        return self.ensure_user(user_id)

    def ensure_user_id(self, user_id: str | None) -> str:
        """确保用户存在并返回规范化 ID。

        这个方法专门给 Core 这类上层调度模块使用：Core 只需要保存
        current_user_id，不应该读取 UserProfile.info 里的内部字段。
        """
        return self.ensure_user(user_id).info.user_id

    def list_users(self) -> list[UserProfile]:
        """按 user_id 排序返回所有用户，保证展示稳定。"""
        return [self.profiles[user_id] for user_id in sorted(self.profiles)]

    def touch_user(self, user_id: str | None, *, timestamp: float | None = None) -> str:
        """更新用户最近活跃时间，并返回规范化后的 user_id。"""
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
        """切换到指定用户，并返回规范化后的 user_id。

        Core 只需要知道当前用户 ID，不需要理解 UserProfile 内部结构。
        """
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
        """更新一个用户显式偏好字段，并返回规范化后的 user_id。"""
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
        """更新一个用户基础资料字段，并返回规范化后的 user_id。

        基础资料必须来自显式命令或明确业务入口；LLM 不能直接写入这些长期信息。
        """
        if key not in INFO_KEYS:
            raise ValueError(f"不支持的用户资料字段: {key}")

        profile = self.ensure_user(user_id)
        setattr(profile.info, key, _parse_info_value(key, raw_value))
        self._mark_updated(profile, self._now(timestamp))
        self._save_profiles()
        return profile.info.user_id

    def upsert_insight(
        self,
        user_id: str | None,
        *,
        insight_type: str,
        content: str,
        confidence: float,
        evidence_count: int,
        timestamp: float | None = None,
    ) -> UserProfileInsight:
        """添加或更新一条画像 insight。

        insight 表达长期倾向，不是绝对事实，所以必须带置信度和证据数。
        """
        profile = self.ensure_user(user_id)
        normalized_type = insight_type.strip()
        normalized_content = content.strip()
        now = self._now(timestamp)

        for insight in profile.insights:
            # 相同类型 + 相同内容视作同一条 insight，更新置信度和证据数。
            if insight.insight_type == normalized_type and insight.content == normalized_content:
                insight.confidence = _clamp_confidence(confidence)
                insight.evidence_count = int(evidence_count)
                insight.updated_at = now
                self._mark_updated(profile, now)
                self._save_profiles()
                return insight

        insight = UserProfileInsight(
            insight_type=normalized_type,
            content=normalized_content,
            confidence=_clamp_confidence(confidence),
            evidence_count=int(evidence_count),
            updated_at=now,
        )
        profile.insights.append(insight)
        self._mark_updated(profile, now)
        self._save_profiles()
        return insight

    def replace_insights(
        self,
        user_id: str | None,
        insights: list[UserProfileInsight],
        *,
        timestamp: float | None = None,
    ) -> str:
        """替换用户画像 insight 列表。

        这个入口供 MemoryPolicy 执行衰减/废弃/矛盾处理后使用；具体策略仍在
        agent.memory.policy 中，service 只负责安全写回 profile。
        """
        profile = self.ensure_user(user_id)
        profile.insights = list(insights)
        self._mark_updated(profile, self._now(timestamp))
        self._save_profiles()
        return profile.info.user_id

    def render_profile(self, user_id: str | None) -> str:
        """渲染当前用户画像，供 /profile 展示。"""
        profile = self.ensure_user(user_id)
        info = profile.info
        info_lines = _info_lines(info)
        pref_lines = _preference_lines(profile.preference)
        insight_lines = [
            f"- {item.insight_type}: {item.content} "
            f"(confidence={item.confidence:.2f}, evidence={item.evidence_count})"
            for item in profile.insights
        ]
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
                "画像:",
                *(insight_lines or ["- 暂无"]),
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
        """渲染切换用户后的提示文本，让 Core 不拼 profile 内部字段。"""
        return f"已切换到用户：{self.user_label(user_id)}"

    def render_preference_update_result(self, user_id: str | None, key: str) -> str:
        """渲染偏好更新结果，让 CLI/Core 不关心用户展示格式。"""
        return f"已更新 {self.user_label(user_id)} 的偏好：{key}"

    def render_info_update_result(self, user_id: str | None, key: str) -> str:
        """渲染基础资料更新结果，让 CLI/Core 不关心用户展示格式。"""
        return f"已更新 {self.user_label(user_id)} 的资料：{key}"

    def info_context(self, user_id: str | None) -> str:
        """生成紧凑基础资料摘要，供 LLM prompt 读取。"""
        profile = self.ensure_user(user_id)
        lines = _info_lines(profile.info)
        return "；".join(line.removeprefix("- ") for line in lines) if lines else "暂无明确资料"

    def preference_context(self, user_id: str | None) -> str:
        """生成紧凑偏好摘要，供 LLM prompt 和个性化文案使用。"""
        profile = self.ensure_user(user_id)
        lines = _preference_lines(profile.preference)
        return "；".join(line.removeprefix("- ") for line in lines) if lines else "暂无明确偏好"

    def user_label(self, user_id: str | None) -> str:
        """返回稳定的用户展示名，格式为 display_name(user_id)。"""
        profile = self.ensure_user(user_id)
        display_name = profile.info.display_name or profile.info.user_id
        return f"{display_name} ({profile.info.user_id})"

    def user_name_prefix(self, user_id: str | None) -> str:
        """返回用于对话文案的用户名前缀；default 用户不加前缀。"""
        profile = self.ensure_user(user_id)
        name = (profile.info.display_name or profile.info.user_id).strip()
        if not name or profile.info.user_id == DEFAULT_USER_ID:
            return ""
        return f"{name}，"

    def uses_gentle_reminder(self, user_id: str | None) -> bool:
        """判断当前用户是否偏好温和提醒。"""
        return self.ensure_user(user_id).preference.reminder_style == "温和"

    def preferred_break_activity(self, user_id: str | None) -> str | None:
        """返回休息建议中可提到的内容偏好。

        内容大类优先；当用户选择“音乐”且有音乐风格时，使用音乐风格。
        这个规则属于 profile 偏好解释，因此集中放在 service。
        """
        preference = self.ensure_user(user_id).preference
        content_type = preference.favorite_content_types[0] if preference.favorite_content_types else ""
        music_style = preference.favorite_music_styles[0] if preference.favorite_music_styles else ""
        if content_type == "音乐" and music_style:
            return music_style
        if content_type:
            return content_type
        if music_style:
            return music_style
        return None

    def tts_payload(self, user_id: str | None) -> dict[str, object]:
        """把长期 TTS 偏好转换为 speak action 可直接使用的 payload。"""
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
        """统一维护更新时间，避免多个方法重复写时间字段。"""
        profile.info.updated_at = timestamp
        if last_seen:
            profile.info.last_seen_at = timestamp

    def _save_profiles(self) -> None:
        """统一持久化入口，避免业务方法直接操作 store 的结构。"""
        self.store.save_profiles({
            user_id: _profile_to_raw(profile)
            for user_id, profile in self.profiles.items()
        })

    def _now(self, timestamp: float | None = None) -> float:
        """统一获取时间，调用方传入 timestamp 时优先使用。"""
        if timestamp is not None:
            return float(timestamp)
        return float(self._now_fn())


def _profiles_from_raw(raw_profiles: dict[str, dict[str, Any]]) -> dict[str, UserProfile]:
    """把 store 返回的原始字典恢复为 profile dataclass。"""
    profiles: dict[str, UserProfile] = {}
    for user_id, raw_profile in raw_profiles.items():
        try:
            profile = _profile_from_raw(raw_profile, fallback_user_id=user_id)
        except (TypeError, ValueError):
            continue
        profiles[profile.info.user_id] = profile
    return profiles


def _profile_from_raw(data: dict[str, Any], *, fallback_user_id: str) -> UserProfile:
    """从单个原始 profile 字典恢复 UserProfile。"""
    info = _info_from_raw(data.get("info"), fallback_user_id=fallback_user_id)
    preference = _preference_from_raw(data.get("preference"))
    insights = [
        _insight_from_raw(item)
        for item in data.get("insights", [])
        if isinstance(item, dict)
    ]
    return UserProfile(info=info, preference=preference, insights=insights)


def _profile_to_raw(profile: UserProfile) -> dict[str, Any]:
    """把 UserProfile 转为可 JSON 序列化的普通字典。"""
    return asdict(profile)


def _info_from_raw(data: object, *, fallback_user_id: str) -> UserInfo:
    """恢复 UserInfo，并用 fallback_user_id 修复缺失主键。"""
    values = _only_fields(data if isinstance(data, dict) else {}, UserInfo)
    values["user_id"] = _normalize_user_id(str(values.get("user_id") or fallback_user_id))
    values["hobbies"] = _normalize_list(values.get("hobbies"))
    if values.get("age") is not None:
        values["age"] = int(values["age"])
    return UserInfo(**values)


def _preference_from_raw(data: object) -> UserPreference:
    """恢复 UserPreference，并集中处理列表/数值默认值。"""
    values = _only_fields(data if isinstance(data, dict) else {}, UserPreference)
    for key in LIST_PREFERENCE_KEYS:
        values[key] = _normalize_list(values.get(key))
    for key in FLOAT_PREFERENCE_KEYS:
        if values.get(key) is not None:
            values[key] = float(values[key])
    return UserPreference(**values)


def _insight_from_raw(data: dict[str, Any]) -> UserProfileInsight:
    """恢复单条 insight，并裁剪置信度。"""
    values = _only_fields(data, UserProfileInsight)
    values["confidence"] = _clamp_confidence(values.get("confidence", 0.0))
    values["evidence_count"] = int(values.get("evidence_count", 0))
    return UserProfileInsight(**values)


def _normalize_user_id(user_id: str | None) -> str:
    """规范化用户 ID，空值统一回退到 default。"""
    normalized = str(user_id or "").strip()
    return normalized or DEFAULT_USER_ID


def _parse_preference_value(key: str, raw_value: object) -> object:
    """按偏好字段类型解析外部输入值。"""
    if key in LIST_PREFERENCE_KEYS:
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        return [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if key in FLOAT_PREFERENCE_KEYS:
        return float(raw_value)
    return str(raw_value).strip() or None


def _parse_info_value(key: str, raw_value: object) -> object:
    """按基础资料字段类型解析外部输入值。"""
    if key in LIST_INFO_KEYS:
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        return [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if key in INT_INFO_KEYS:
        return int(raw_value)
    return str(raw_value).strip() or None


def _info_lines(info: UserInfo) -> list[str]:
    """把非空基础资料字段渲染成稳定顺序的文本行。"""
    lines: list[str] = []
    for key in INFO_RENDER_ORDER:
        value = getattr(info, key)
        if value is None or value == []:
            continue
        rendered = ", ".join(value) if isinstance(value, list) else str(value)
        lines.append(f"- {key}: {rendered}")
    return lines


def _preference_lines(preference: UserPreference) -> list[str]:
    """把非空偏好字段渲染成稳定顺序的文本行。"""
    lines: list[str] = []
    for key in PREFERENCE_RENDER_ORDER:
        value = getattr(preference, key)
        if value is None or value == []:
            continue
        rendered = ", ".join(value) if isinstance(value, list) else str(value)
        lines.append(f"- {key}: {rendered}")
    return lines


def _only_fields(data: dict[str, Any], dc: type) -> dict[str, Any]:
    """只保留目标 dataclass 定义过的字段，忽略未知字段。"""
    names = {field.name for field in fields(dc)}
    return {key: value for key, value in data.items() if key in names}


def _normalize_list(value: object) -> list[str]:
    """把外部输入或 JSON 值规范化为 list[str]。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    rendered = str(value).strip()
    return [rendered] if rendered else []


def _clamp_confidence(value: object) -> float:
    """把 insight 置信度裁剪到 0 到 1。"""
    return max(0.0, min(1.0, float(value)))
