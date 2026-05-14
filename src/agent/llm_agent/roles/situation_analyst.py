from __future__ import annotations

"""
SituationAnalyst 角色模块。

本模块负责调用 LLM 分析当前发生了什么、用户可能意图、风险和不确定性。
上游输入是 AgentContext，下游输出是 SituationFrame。它不允许生成 Action、
不允许修改状态，也不负责长期记忆写入。
"""

from pathlib import Path

from src.agent.decision.agent_context_builder import AgentContext
from src.agent.prompt_io import prompt_path, read_prompt
from src.agent.llm_agent.schemas import SituationFrame, fallback_situation, parse_json_object
from src.services.llm_service import LLMService

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class SituationAnalyst:
    """场景理解 LLM 角色。

    输入紧凑上下文，输出只描述事实和风险的 SituationFrame。失败时返回可解释
    fallback frame，让后续 Planner 仍能安全降级。
    """

    role_name = "situation_analyst"

    def __init__(self, prompt_path_override: Path | None = None) -> None:
        self.prompt_path = prompt_path_override or prompt_path(_PROMPTS_DIR, "situation_analyst.md")

    def analyze(self, context: AgentContext, llm_service: LLMService) -> tuple[SituationFrame, dict[str, object]]:
        """调用 LLM 生成 SituationFrame，并拒绝空摘要或非法 JSON。"""

        prompt = f"{read_prompt(self.prompt_path)}\n\nContext JSON:\n{context.to_prompt_json()}"
        try:
            raw = llm_service.complete_json(self.role_name, prompt)
            frame = SituationFrame.from_dict(parse_json_object(raw))
            if not frame.summary:
                raise ValueError("situation summary is empty")
            return frame, {"raw": raw, "fallback": False}
        except Exception as exc:
            frame = fallback_situation(context.event_type, context.user_text)
            return frame, {"fallback": True, "error": str(exc)}
