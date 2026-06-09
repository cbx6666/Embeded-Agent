"""桌宠表情映射（对齐 Embeded-Agent-asr-test 的 render_pet_expression 闭集）。"""

from __future__ import annotations

# asr-test 文档与测试用例中的表情闭集
PET_EXPRESSIONS = frozenset({"idle", "happy", "sleepy", "angry", "neutral", "alert"})

EXPRESSION_LABELS: dict[str, str] = {
    "idle": "空闲",
    "happy": "开心",
    "sleepy": "困倦",
    "angry": "烦躁",
    "neutral": "平静",
    "alert": "提醒",
}

_EMOTION_TO_EXPRESSION = {
    "happy": "happy",
    "neutral": "idle",
    "tired": "sleepy",
    "stressed": "angry",
}

_AGENT_TO_EXPRESSION = {
    "idle": "idle",
    "listening": "happy",
    "thinking": "neutral",
    "speaking": "happy",
    "focus_mode": "sleepy",
}


def resolve_expression(*, agent_state: str, user_emotion: str = "neutral") -> str:
    """Agent 对话状态优先；待命时再反映用户情绪。"""
    mapped = _AGENT_TO_EXPRESSION.get(agent_state, "idle")
    if agent_state == "idle":
        return _EMOTION_TO_EXPRESSION.get(user_emotion.strip().lower(), "idle")
    return mapped


def expression_label(expression: str) -> str:
    return EXPRESSION_LABELS.get(expression, expression)
