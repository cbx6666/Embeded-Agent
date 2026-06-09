from __future__ import annotations

"""LLM 决策入口的 prompt 构建。

- ``build_speech_prompt``：speech_recognized
- ``build_behavior_distraction_prompt``：behavior_distraction_check
- ``build_wellness_prompt``：wellness_care_check
- ``build_environment_care_prompt``：environment_care_check

prompt 文本来自 ``src/agent/prompts/*.md``；均带上结构化 ``user_context``
（含 ``memory_usage_hints``）。
"""

import json
from pathlib import Path
from typing import Any

from src.agent.prompt_io import prompt_path, read_prompt
from src.agent.state.agent_state import AgentState

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_SPEECH_PROMPT = prompt_path(_PROMPTS_DIR, "speech_recognized.md")
_BEHAVIOR_DISTRACTION_PROMPT = prompt_path(_PROMPTS_DIR, "behavior_distraction_check.md")
_WELLNESS_PROMPT = prompt_path(_PROMPTS_DIR, "wellness_care_check.md")
_ENVIRONMENT_CARE_PROMPT = prompt_path(_PROMPTS_DIR, "environment_care_check.md")
_FOCUS_COMPLETE_PROMPT = prompt_path(_PROMPTS_DIR, "focus_complete_care.md")
_TTS_REPLY_QUALITY = prompt_path(_PROMPTS_DIR, "tts_reply_quality.md")


def _with_tts_quality(prompt_body: str) -> str:
    return f"{prompt_body}\n\n{read_prompt(_TTS_REPLY_QUALITY)}"


def state_brief(state: AgentState) -> dict[str, Any]:
    """供 prompt 使用的紧凑状态视图。"""

    return {
        "interaction": {
            "mode": state.interaction.mode,
            "dialogue_state": state.interaction.dialogue_state,
            "in_conversation": state.interaction.in_conversation,
        },
        "focus": {
            "active": state.focus.active,
            "elapsed_sec": state.focus.elapsed_sec,
            "remaining_sec": state.focus.remaining_sec,
            "target_duration_sec": state.focus.target_duration_sec,
        },
        "user": {
            "presence": state.user.presence,
            "attention": state.user.attention,
            "emotion": state.user.emotion,
            "fatigue_level": state.user.fatigue_level,
            "posture": state.user.posture,
            "current_activity": state.user.current_activity,
        },
        "environment": {
            "light_level": state.environment.light_level,
            "noise_level": state.environment.noise_level,
            "temperature_level": state.environment.temperature_level,
            "humidity_level": state.environment.humidity_level,
        },
    }


def speech_state_brief(state: AgentState) -> dict[str, Any]:
    """语音回复专用状态：不含视觉疲劳/情绪/姿态，避免与自主关怀检查抢话。"""

    return {
        "interaction": {
            "mode": state.interaction.mode,
            "dialogue_state": state.interaction.dialogue_state,
            "in_conversation": state.interaction.in_conversation,
        },
        "focus": {
            "active": state.focus.active,
            "elapsed_sec": state.focus.elapsed_sec,
            "remaining_sec": state.focus.remaining_sec,
            "target_duration_sec": state.focus.target_duration_sec,
        },
        "user": {
            "presence": state.user.presence,
            "attention": state.user.attention,
            "current_activity": state.user.current_activity,
        },
    }


def _format_user_context(user_context: dict[str, Any] | None) -> str:
    """把结构化 user_context 渲染为 JSON 块；空时给出占位。"""

    if not user_context:
        return "{}"
    return json.dumps(user_context, ensure_ascii=False, indent=2)


def build_speech_prompt(
    *,
    state: AgentState,
    user_context: dict[str, Any] | None,
    user_text: str,
    media_context: dict[str, Any] | None = None,
) -> str:
    media_block = ""
    if media_context:
        media_block = (
            f"\n\n## 媒体播放状态 media_context\n"
            f"{json.dumps(media_context, ensure_ascii=False, indent=2)}\n"
            f"library.tracks 列出全部曲目及 folder/media_type 属性，选曲只能从中取 id。\n"
            f"仅用户明确点播或 pending_suggestion 已批准时才可 play_media；"
            f"必须先写好 reply（将先播报再播放）；用户拒绝则 answer_user。"
        )
    return (
        f"{_with_tts_quality(read_prompt(_SPEECH_PROMPT))}\n\n"
        f"## 当前状态\n{json.dumps(speech_state_brief(state), ensure_ascii=False, indent=2)}\n\n"
        f"## 用户画像与记忆 user_context\n{_format_user_context(user_context)}\n\n"
        f"## 用户本轮语音\n{user_text.strip()}"
        f"{media_block}"
    )


def build_behavior_distraction_prompt(
    *,
    distraction_summary: dict[str, Any],
    user_context: dict[str, Any] | None,
) -> str:
    return (
        f"{_with_tts_quality(read_prompt(_BEHAVIOR_DISTRACTION_PROMPT))}\n\n"
        f"## 分心检查汇总\n{json.dumps(distraction_summary, ensure_ascii=False, indent=2)}\n\n"
        f"## 用户画像与记忆 user_context\n{_format_user_context(user_context)}"
    )


def build_wellness_prompt(
    *,
    wellness_summary: dict[str, Any],
    selected_intent: str,
    care_focus: str,
    user_context: dict[str, Any] | None,
    wellness_reply_context: dict[str, Any] | None = None,
    media_suggestion: dict[str, Any] | None = None,
    media_ask_allowed: bool = True,
) -> str:
    focus_summary = wellness_summary.get("focus_summary") or {"active": False}
    media_block = ""
    if media_suggestion:
        media_block = (
            f"\n\n## 本轮媒体建议（系统已选定，勿自行改类型）\n"
            f"{json.dumps(media_suggestion, ensure_ascii=False, indent=2)}\n"
            f"intent=suggest_media 时：你**必须**结合 care_topic 与 user_context，用一句口语**询问**用户"
            f"是否愿意听本轮 media_type/media_category（如轻音乐、相声），**禁止**直接说已开始播放；"
            f"**禁止**编造库里不存在的曲目名。"
        )
    elif not media_ask_allowed:
        media_block = (
            "\n\n## 媒体询问限制（系统硬性约束）\n"
            "本轮 intent **不是** suggest_media，系统**禁止**你询问是否听歌/放音乐/听相声/播放媒体。\n"
            "你**不得**在 reply 中出现「要不要听」「放首歌」「听歌放松」「来段相声」等邀请播放的措辞。\n"
            "记忆或兴趣里的音乐/相声内容**只能**用于非媒体类关怀（如休息、坐姿、情绪安抚），"
            "**不得**转化为播放邀请。"
        )
    reply_ctx_block = ""
    if wellness_reply_context:
        reply_ctx_block = (
            f"\n\n## 本轮关怀个性化上下文 wellness_reply_context\n"
            f"{json.dumps(wellness_reply_context, ensure_ascii=False, indent=2)}"
        )
    return (
        f"{_with_tts_quality(read_prompt(_WELLNESS_PROMPT))}\n\n"
        f"## 已确定的关怀方向\n"
        f"trigger_focus={care_focus}（fatigue/emotion/posture，仅表示**本轮触发原因**，不是文案模板）; "
        f"intent={selected_intent}\n\n"
        f"## 专注计时状态 focus_summary\n"
        f"{json.dumps(focus_summary, ensure_ascii=False, indent=2)}\n\n"
        f"## 疲劳/情绪/姿态关怀汇总\n{json.dumps(wellness_summary, ensure_ascii=False, indent=2)}"
        f"{reply_ctx_block}\n\n"
        f"## 用户画像与记忆 user_context\n{_format_user_context(user_context)}"
        f"{media_block}"
    )


def build_environment_care_prompt(
    *,
    environment_summary: dict[str, Any],
    user_context: dict[str, Any] | None,
) -> str:
    return (
        f"{_with_tts_quality(read_prompt(_ENVIRONMENT_CARE_PROMPT))}\n\n"
        f"## 环境关怀汇总\n{json.dumps(environment_summary, ensure_ascii=False, indent=2)}\n\n"
        f"## 用户画像与记忆 user_context\n{_format_user_context(user_context)}"
    )


def build_focus_complete_prompt(
    *,
    state: AgentState,
    user_context: dict[str, Any] | None,
) -> str:
    focus_info = {
        "target_duration_sec": state.focus.target_duration_sec,
        "elapsed_sec": state.focus.elapsed_sec,
    }
    return (
        f"{_with_tts_quality(read_prompt(_FOCUS_COMPLETE_PROMPT))}\n\n"
        f"## 专注信息\n{json.dumps(focus_info, ensure_ascii=False, indent=2)}\n\n"
        f"## 用户画像与记忆 user_context\n{_format_user_context(user_context)}"
    )

