from __future__ import annotations

"""决策落地阶段使用的轻量个性化文案工具。

这里放的是“如何把已存在的用户偏好应用到文案/动作参数上”的规则。
它不创建用户、不修改 profile、不持久化数据，也不推断画像；这些都属于
UserProfileService 的职责。Realizer 只调用这里拿文本或 payload，保持
Intent -> Action 落地层尽量干净。
"""

from src.agent.state import AgentState
from src.services.user_profile_service import UserProfileService

DEFAULT_REST_REMINDER = "你已经专注了一段时间，而且看起来有点疲劳，建议休息一下。"
DEFAULT_DISTRACTION_REMINDER = "检测到你有些分心，建议回到当前专注任务。"
NO_PREFERENCE_CONTEXT = "暂无明确偏好"
NO_INFO_CONTEXT = "暂无明确资料"


def build_rest_reminder_text(
    state: AgentState,
    profile_service: UserProfileService | None,
) -> str:
    """根据当前用户偏好生成休息提醒文案。

    个性化只消费 service 暴露的只读查询：提醒风格、用户名前缀、休息内容偏好。
    这样 Realizer 不需要知道 UserProfile 的字段结构，也不会承担 profile 管理职责。
    """
    if profile_service is None:
        return DEFAULT_REST_REMINDER

    user_id = state.current_user_id
    prefix = profile_service.user_name_prefix(user_id)
    if profile_service.uses_gentle_reminder(user_id):
        text = f"{prefix}你有点累啦。要不要先休息一下？"
    else:
        text = f"{prefix}{DEFAULT_REST_REMINDER}"

    activity = profile_service.preferred_break_activity(user_id)
    if activity:
        return f"{text}我可以陪你听一小段{activity}，放松一下再继续。"
    return text


def build_distraction_reminder_text(
    state: AgentState,
    profile_service: UserProfileService | None,
) -> str:
    """根据提醒风格生成分心提醒文案。"""
    if profile_service is None:
        return DEFAULT_DISTRACTION_REMINDER

    user_id = state.current_user_id
    prefix = profile_service.user_name_prefix(user_id)
    if profile_service.uses_gentle_reminder(user_id):
        return f"{prefix}稍微有点走神啦，我们慢慢把注意力拉回当前任务。"
    return f"{prefix}{DEFAULT_DISTRACTION_REMINDER}"


def build_llm_prompt(
    text: str,
    state: AgentState,
    profile_service: UserProfileService | None,
) -> str:
    """构造自然语言回复阶段使用的上下文提示词。

    LLM 只能读取偏好摘要来辅助回复，不能直接写 profile 或产出 Action。
    """
    recent_messages = state.memory.recent_messages[-5:]
    latest_emotion = state.memory.emotion_summaries[-1] if state.memory.emotion_summaries else {}
    preference_context = (
        profile_service.preference_context(state.current_user_id)
        if profile_service is not None
        else NO_PREFERENCE_CONTEXT
    )
    info_context = (
        profile_service.info_context(state.current_user_id)
        if profile_service is not None
        else NO_INFO_CONTEXT
    )
    return (
        f"用户输入：{text}\n"
        f"当前用户：{state.current_user_id}\n"
        f"用户资料：{info_context}\n"
        f"用户偏好：{preference_context}\n"
        f"专注状态：active={state.focus.active}, remaining={state.focus.remaining_sec}\n"
        f"用户状态：presence={state.user.presence}, attention={state.user.attention}, "
        f"emotion={state.user.emotion}, fatigue={state.user.fatigue_level}\n"
        f"最近消息：{recent_messages}\n"
        f"最近情绪摘要：{latest_emotion}"
    )


def tts_payload_for_user(
    state: AgentState,
    profile_service: UserProfileService | None,
) -> dict[str, object]:
    """读取当前用户的 TTS 偏好，并转成 speak action 的 payload。"""
    if profile_service is None:
        return {}
    return profile_service.tts_payload(state.current_user_id)
