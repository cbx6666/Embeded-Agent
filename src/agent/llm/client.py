from __future__ import annotations

"""唯一的 LLM 调用封装。

``LLMClient`` 包装底层 LLM 服务（生产环境的 DeepSeek ``LLMService`` 或测试 fake），
对外只暴露 ``complete_json``：发送 prompt、解析为 JSON 对象。失败时抛错，由上层
决策处理器决定 fallback。

这里没有四角色编排、没有流式 sink、没有本地语义兜底；语义理解全部交给 LLM。
"""

import json
import re
from typing import Any, Protocol


class _RawLLM(Protocol):
    def complete_json(self, role: str, prompt: str) -> str: ...


class LLMClient:
    """对底层 LLM 服务的薄封装。"""

    def __init__(self, llm_service: _RawLLM) -> None:
        self._llm = llm_service

    @property
    def service(self) -> _RawLLM:
        return self._llm

    def complete_json(
        self, role: str, prompt: str, *, temperature: float | None = None
    ) -> dict[str, Any]:
        """调用一次 LLM 并解析为 JSON 对象；失败抛出 ValueError/RuntimeError。

        ``temperature`` 为 None 时沿用底层服务默认（稳定 JSON）；面向用户的回复类
        决策可显式传入更高温度让文案更多变。底层若不支持该参数则自动回退。
        """

        if temperature is None:
            raw = self._llm.complete_json(role, prompt)
        else:
            try:
                raw = self._llm.complete_json(role, prompt, temperature=temperature)
            except TypeError:
                raw = self._llm.complete_json(role, prompt)
        return parse_json_object(raw)


def parse_json_object(text: str) -> dict[str, Any]:
    """解析单个 JSON 对象，兼容常见的 ```json 包裹输出。"""

    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON from LLM: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    return data
