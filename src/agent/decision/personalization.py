from __future__ import annotations

"""决策落地阶段使用的轻量个性化文案工具。

这里放的是“如何把已存在的用户偏好应用到文案/动作参数上”的规则。
它不创建用户、不修改 profile、不持久化数据，也不推断画像；这些都属于
UserProfileService 的职责。Realizer 只调用这里拿文本或 payload，保持
Intent -> Action 落地层尽量干净。
"""

from src.agent.state import AgentState
from src.agent.memory.policy.personalization_policy import PersonalizedPolicy
from src.services.user_profile_service import UserProfileService

DEFAULT_REST_REMINDER = "你已经专注了一段时间，而且看起来有点疲劳，建议休息一下。"
DEFAULT_DISTRACTION_REMINDER = "检测到你有些分心，建议回到当前专注任务。"
NO_PREFERENCE_CONTEXT = "暂无明确偏好"
NO_INFO_CONTEXT = "暂无明确资料"


def build_rest_reminder_text(
    state: AgentState,
    personalized_policy: PersonalizedPolicy | None,
    profile_service: UserProfileService | None = None,
) -> str:
    """根据当前用户偏好生成休息提醒文案。

    优先消费 MemoryPipeline 生成的 PersonalizedPolicy；profile_service 仅作为旧调用路径
    的兜底输入，避免 Realizer 直接理解 UserProfile 内部结构。
    """
    policy = personalized_policy or _build_policy_from_service(state, profile_service)
    if policy is None:
        return DEFAULT_REST_REMINDER

    if policy.reminder_tone == "温和":
        text = f"{policy.user_name_prefix}你有点累啦。要不要先休息一下？"
    else:
        text = f"{policy.user_name_prefix}{DEFAULT_REST_REMINDER}"

    if policy.break_activity:
        return f"{text}我可以陪你听一小段{policy.break_activity}，放松一下再继续。"
    return text


def build_distraction_reminder_text(
    state: AgentState,
    personalized_policy: PersonalizedPolicy | None,
    profile_service: UserProfileService | None = None,
) -> str:
    """根据个性化策略生成分心提醒文案。"""
    policy = personalized_policy or _build_policy_from_service(state, profile_service)
    if policy is None:
        return DEFAULT_DISTRACTION_REMINDER

    if policy.reminder_tone == "温和":
        return f"{policy.user_name_prefix}稍微有点走神啦，我们慢慢把注意力拉回当前任务。"
    return f"{policy.user_name_prefix}{DEFAULT_DISTRACTION_REMINDER}"


def build_llm_prompt(
    text: str,
    state: AgentState,
    personalized_policy: PersonalizedPolicy | None,
    profile_service: UserProfileService | None = None,
) -> str:
    """构造自然语言回复阶段使用的上下文提示词。

    LLM 只能读取资料/偏好/画像策略摘要来辅助回复，不能直接写 profile 或产出 Action。
    """
    recent_messages = state.memory.recent_messages[-5:]
    latest_emotion = state.memory.emotion_summaries[-1] if state.memory.emotion_summaries else {}
    policy = personalized_policy or _build_policy_from_service(state, profile_service)
    preference_context = policy.preference_context if policy is not None else NO_PREFERENCE_CONTEXT
    info_context = policy.info_context if policy is not None else NO_INFO_CONTEXT
    policy_explanations = policy.explanations if policy is not None else []

    return (
        f"用户输入：{text}\n"
        f"当前用户：{state.current_user_id}\n"
        f"用户资料：{info_context}\n"
        f"用户偏好：{preference_context}\n"
        f"个性化策略依据：{policy_explanations}\n"
        f"专注状态：active={state.focus.active}, remaining={state.focus.remaining_sec}\n"
        f"用户状态：presence={state.user.presence}, attention={state.user.attention}, "
        f"emotion={state.user.emotion}, fatigue={state.user.fatigue_level}\n"
        f"最近消息：{recent_messages}\n"
        f"最近情绪摘要：{latest_emotion}"
    )


def tts_payload_for_user(
    state: AgentState,
    personalized_policy: PersonalizedPolicy | None,
    profile_service: UserProfileService | None = None,
) -> dict[str, object]:
    """读取当前用户的 TTS 个性化策略，并转成 speak action 的 payload。"""
    policy = personalized_policy or _build_policy_from_service(state, profile_service)
    if policy is None:
        return {}
    return dict(policy.tts_payload)


def _build_policy_from_service(
    state: AgentState,
    profile_service: UserProfileService | None,
) -> PersonalizedPolicy | None:
    """兼容没有 MemoryPipeline 的测试/旧路径，集中从 service 构造策略快照。"""
    if profile_service is None:
        return None
    user_id = state.current_user_id
    reminder_tone = "温和" if profile_service.uses_gentle_reminder(user_id) else None
    return PersonalizedPolicy(
        user_id=profile_service.ensure_user_id(user_id),
        reminder_tone=reminder_tone,
        user_name_prefix=profile_service.user_name_prefix(user_id),
        break_activity=profile_service.preferred_break_activity(user_id),
        tts_payload=profile_service.tts_payload(user_id),
        info_context=profile_service.info_context(user_id),
        preference_context=profile_service.preference_context(user_id),
    )
