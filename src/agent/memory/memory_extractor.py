from __future__ import annotations

"""单次 LLM 用户记忆抽取器。

``MemoryExtractor`` 只做一件事：把用户一句话（可带最近对话上下文）交给一次
``memory_extract`` LLM 调用，解析出结构化 ``MemoryItem`` 列表。

这里**没有**旧版的 critic / consolidator / validator 多阶段链路，也不做四角色编排。
只有一次 LLM 调用 + schema 校验；调用失败由调用方（后台线程）吞掉，绝不影响主链路。
"""

from pathlib import Path
from typing import Any, Protocol

from src.agent.memory.memory_model import MEMORY_TYPES, MemoryItem, make_memory_item
from src.agent.prompt_io import prompt_path, read_prompt

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_EXTRACT_PROMPT = prompt_path(_PROMPTS_DIR, "memory_extract.md")

# 单条记忆 content 上限，避免 LLM 写长段落。
_MAX_CONTENT_LEN = 200
# 单次抽取最多接收的 memory_items，防御异常输出。
_MAX_ITEMS_PER_CALL = 8


class _MemoryLLM(Protocol):
    def complete_json(self, role: str, prompt: str) -> dict[str, Any]: ...


class MemoryExtractor:
    """通过一次 LLM 调用从用户言谈抽取结构化记忆。"""

    def __init__(self, llm_client: _MemoryLLM, *, role: str = "memory_extract") -> None:
        self._llm = llm_client
        self._role = role

    def extract(
        self,
        *,
        user_id: str,
        text: str,
        timestamp: int,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> list[MemoryItem]:
        """抽取记忆；任何异常向上抛出，由调用方决定吞掉（后台异步）。"""

        user_text = str(text or "").strip()
        if not user_text:
            return []

        prompt = build_memory_extract_prompt(
            user_text=user_text,
            recent_messages=recent_messages or [],
        )
        data = self._llm.complete_json(self._role, prompt)
        return self._parse(user_id=user_id, data=data, timestamp=timestamp)

    @staticmethod
    def _parse(*, user_id: str, data: dict[str, Any], timestamp: int) -> list[MemoryItem]:
        if not isinstance(data, dict):
            return []
        raw_items = data.get("memory_items")
        if not isinstance(raw_items, list):
            return []

        items: list[MemoryItem] = []
        for raw in raw_items[:_MAX_ITEMS_PER_CALL]:
            if not isinstance(raw, dict):
                continue
            item_type = str(raw.get("type", "")).strip().lower()
            if item_type not in MEMORY_TYPES:
                continue
            content = str(raw.get("content", "")).strip()
            if not content:
                continue
            if len(content) > _MAX_CONTENT_LEN:
                content = content[:_MAX_CONTENT_LEN].rstrip()
            items.append(
                make_memory_item(
                    user_id=user_id,
                    type=item_type,
                    content=content,
                    evidence=str(raw.get("evidence", "")).strip(),
                    confidence=raw.get("confidence", 0.0),
                    tags=raw.get("tags"),
                    source_event="speech_recognized",
                    timestamp=timestamp,
                )
            )
        return items


def build_memory_extract_prompt(
    *,
    user_text: str,
    recent_messages: list[dict[str, Any]],
) -> str:
    """拼接 memory_extract prompt：系统指令 + 最近对话 + 用户本轮语音。"""

    lines: list[str] = []
    for item in list(recent_messages)[-6:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        text = str(item.get("text") or "").strip()
        if text:
            lines.append(f"- {role}: {text}")
    recent_block = "\n".join(lines) if lines else "-（无最近对话）"

    return (
        f"{read_prompt(_EXTRACT_PROMPT)}\n\n"
        f"## 最近对话（仅供理解上下文，不要直接保存）\n{recent_block}\n\n"
        f"## 用户本轮语音（从这里抽取）\n{user_text}"
    )
