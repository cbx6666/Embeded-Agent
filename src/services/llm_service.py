from __future__ import annotations

"""大模型服务占位模块。"""

from src.agent.state import AgentState


class LLMService:
    """简化版 LLM 服务。"""

    def generate_reply(self, text: str, state: AgentState) -> str:
        lowered = text.strip().lower()
        if any(word in lowered for word in ("你好", "hello", "hi")):
            return "你好，我已在线。你可以让我开始专注、结束专注，或用 /mock 更新状态。"
        if "谢谢" in lowered or "thanks" in lowered:
            return "收到。当前版本可以帮助你管理专注、记录状态并给出简单提醒。"
        if state.focus.active:
            return "收到。当前你处于专注模式，我会尽量保持简洁。"
        return "收到。当前 MVP 版本支持专注计时、mock 状态更新、基础记忆和简单提醒。"
