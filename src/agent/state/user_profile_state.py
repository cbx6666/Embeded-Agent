from __future__ import annotations

"""长期用户画像状态模型。

本模块只定义数据结构，不做序列化、默认值修复、偏好解析或业务判断。
这些逻辑统一放在 UserProfileService / ProfileStore，避免 state 层和服务层职责混在一起。
"""

from dataclasses import dataclass, field


@dataclass
class UserInfo:
    """单个用户的基础身份信息。

    user_id 是长期 profile 的主键；display_name 只用于展示。
    age、gender、identity、hobbies 来自用户显式提供，属于基础资料，
    不放进 preference，避免把“用户是谁”和“用户喜欢什么服务方式”混在一起。
    三个时间字段由 UserProfileService 维护，state 层不决定何时更新。
    """

    user_id: str
    display_name: str | None = None
    age: int | None = None
    gender: str | None = None
    identity: str | None = None
    hobbies: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    last_seen_at: float = 0.0


@dataclass
class UserPreference:
    """用户显式设置的长期偏好。

    favorite_content_types 表达休息/陪伴内容大类，例如音乐、相声、脱口秀、戏曲。
    favorite_music_styles 只表达音乐内部风格，例如轻音乐、古风、爵士。
    """

    favorite_content_types: list[str] = field(default_factory=list)
    favorite_music_styles: list[str] = field(default_factory=list)
    disliked_topics: list[str] = field(default_factory=list)
    reminder_style: str | None = None
    speech_style: str | None = None
    tts_voice: str | None = None
    tts_speed: float | None = None
    tts_volume: float | None = None


@dataclass
class UserProfileInsight:
    """从长期行为中沉淀出的软画像结论。

    insight 不是事实库，因此必须带 confidence 和 evidence_count，
    后续策略层使用它时也应把它当作倾向而非绝对判断。
    """

    insight_type: str
    content: str
    confidence: float = 0.0
    evidence_count: int = 0
    updated_at: float = 0.0


@dataclass
class UserProfile:
    """单个用户的长期画像聚合根。

    info 保存身份和时间信息，preference 保存用户显式偏好，
    insights 保存系统从长期行为中总结出的画像倾向。
    """

    info: UserInfo
    preference: UserPreference = field(default_factory=UserPreference)
    insights: list[UserProfileInsight] = field(default_factory=list)
