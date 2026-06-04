from __future__ import annotations

"""Fast dialogue prompt builder.

将四角色链路的关键输入压缩为一次 LLM 调用：状态摘要、最近对话、个性化与结构化上下文。
"""

import json
from pathlib import Path
from typing import Any

from src.agent.decision.agent_context_builder import AgentContext
from src.agent.prompt_io import prompt_path, read_prompt

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_FAST_DIALOGUE_PROMPT = prompt_path(_PROMPTS_DIR, "fast_dialogue.md")
_ROLE_NAME = "fast_dialogue"


def build_fast_dialogue_prompt(context: AgentContext) -> str:
    """构建 fast 模式单次调用的完整 prompt。"""

    prompt_dict = context.to_prompt_dict()
    structured = _compact_context(prompt_dict)
    return (
        f"{read_prompt(_FAST_DIALOGUE_PROMPT)}\n\n"
        f"## 当前状态摘要\n{_format_state_brief(context.state_summary)}\n\n"
        f"## 最近对话\n{_format_recent_messages(context.recent_messages)}\n\n"
        f"## 用户偏好与相关记忆\n"
        f"{_format_personalization_brief(prompt_dict.get('personalization_guidance', {}))}\n\n"
        f"## 结构化上下文 JSON\n"
        f"{json.dumps(structured, ensure_ascii=False, indent=2)}\n\n"
        f"## 用户本轮输入\n{context.user_text.strip()}"
    )


def fast_dialogue_role_name() -> str:
    return _ROLE_NAME


def _compact_context(prompt_dict: dict[str, Any]) -> dict[str, Any]:
    """保留决策相关字段，去掉过大的 personal_context 原始块。"""

    return {
        "event": prompt_dict.get("event", {}),
        "state": prompt_dict.get("state", {}),
        "previous_state": prompt_dict.get("previous_state", {}),
        "personalization_guidance": prompt_dict.get("personalization_guidance", {}),
        "recent_messages": prompt_dict.get("recent_messages", []),
        "relevant_memories": prompt_dict.get("relevant_memories", []),
    }


def _format_state_brief(state: dict[str, Any]) -> str:
    if not state:
        return "- （暂无运行时状态）"

    lines: list[str] = []
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    if focus.get("active"):
        lines.append(
            "- 专注：进行中；"
            f"目标 {_minutes(focus.get('target_duration_sec'))}；"
            f"已进行 {_duration(focus.get('elapsed_sec'))}；"
            f"剩余 {_duration(focus.get('remaining_sec'))}"
        )
    else:
        lines.append("- 专注：当前未在计时")

    interaction = state.get("interaction") if isinstance(state.get("interaction"), dict) else {}
    lines.append(
        "- 交互："
        f"mode={interaction.get('mode', 'unknown')}，"
        f"对话={interaction.get('dialogue_state', 'unknown')}，"
        f"会话中={_yes_no(interaction.get('in_conversation'))}"
    )

    user = state.get("user") if isinstance(state.get("user"), dict) else {}
    lines.append(
        "- 用户："
        f"在场={user.get('presence', 'unknown')}，"
        f"注意力={user.get('attention', 'unknown')}，"
        f"疲劳={user.get('fatigue_level', 'unknown')}，"
        f"情绪={user.get('emotion', 'unknown')}，"
        f"姿态={user.get('posture', 'unknown')}，"
        f"活动={user.get('current_activity', 'unknown')}"
    )

    environment = state.get("environment") if isinstance(state.get("environment"), dict) else {}
    lines.append(
        "- 环境："
        f"光线={environment.get('light_level', 'unknown')}，"
        f"噪音={environment.get('noise_level', 'unknown')}，"
        f"温度={environment.get('temperature_level', 'unknown')}，"
        f"湿度={environment.get('humidity_level', 'unknown')}"
    )

    cooldowns = state.get("cooldowns") if isinstance(state.get("cooldowns"), dict) else {}
    if cooldowns:
        lines.append(f"- 提醒冷却记录：{len(cooldowns)} 项")
    return "\n".join(lines)


def _format_recent_messages(messages: list[dict[str, Any]], *, limit: int = 4) -> str:
    if not messages:
        return "- （无最近对话）"
    lines: list[str] = []
    for item in messages[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("speaker") or "unknown")
        text = str(item.get("text") or item.get("content") or "").strip()
        if text:
            lines.append(f"- {role}: {text}")
    return "\n".join(lines) if lines else "- （无最近对话）"


def _format_personalization_brief(guidance: dict[str, Any]) -> str:
    if not guidance:
        return "- （无显式偏好或相关长期记忆）"

    lines: list[str] = []
    preferences = guidance.get("explicit_user_preferences")
    if isinstance(preferences, list):
        for item in preferences[:4]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if content:
                lines.append(f"- 画像偏好：{content}")

    memories = guidance.get("relevant_long_term_memory")
    if isinstance(memories, list):
        for item in memories[:4]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if content:
                conf = item.get("effective_confidence", item.get("confidence", ""))
                lines.append(f"- 相关记忆（confidence={conf}）：{content}")

    conflicts = guidance.get("profile_memory_conflicts")
    if isinstance(conflicts, list) and conflicts:
        lines.append(f"- 注意：存在 {len(conflicts)} 条画像/记忆冲突，以 UserProfile 为准")

    return "\n".join(lines) if lines else "- （无显式偏好或相关长期记忆）"


def _duration(seconds: object) -> str:
    try:
        total = max(0, int(seconds or 0))
    except (TypeError, ValueError):
        return "未知"
    minutes, secs = divmod(total, 60)
    if minutes and secs:
        return f"{minutes} 分 {secs} 秒"
    if minutes:
        return f"{minutes} 分钟"
    return f"{secs} 秒"


def _minutes(seconds: object) -> str:
    try:
        total = max(0, int(seconds or 0))
    except (TypeError, ValueError):
        return "未知"
    if total >= 60:
        return f"{total // 60} 分钟"
    return f"{total} 秒"


def _yes_no(value: object) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value or "unknown")
