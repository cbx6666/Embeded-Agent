from __future__ import annotations

"""显式用户画像数据模型。

它是什么：
UserProfile 是用户明确声明或系统明确配置的资料，包括展示名、年龄、爱好、
显式偏好、提醒风格、TTS 风格和语速等。

它不是什么：
它不是 LongTermMemory，不承载从模糊行为里推断出的“系统学习结论”，也不是
RuntimeHistory 的归档位置。

为什么存在：
显式资料具有权威性和可编辑性。把它与系统学习的长期记忆分开，能避免“用户说过的”
和“系统猜到的”互相污染。

边界：
UserProfile 只能由 UserProfileService 通过显式命令、配置或可靠业务入口写入。
LLM 不允许直接生成或修改 UserProfile。
"""

from dataclasses import dataclass, field


@dataclass
class UserInfo:
    """用户明确提供的基础资料。"""

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
    """用户明确设置的服务偏好。"""

    favorite_content_types: list[str] = field(default_factory=list)
    favorite_music_styles: list[str] = field(default_factory=list)
    disliked_topics: list[str] = field(default_factory=list)
    reminder_style: str | None = None
    speech_style: str | None = None
    tts_voice: str | None = None
    tts_speed: float | None = None
    tts_volume: float | None = None


@dataclass
class UserProfile:
    """单个用户的权威显式画像。"""

    info: UserInfo
    preference: UserPreference = field(default_factory=UserPreference)
