from __future__ import annotations

"""Prompt 文件读取工具。

只提供两个极轻量函数，不引入 PromptManager 或任何抽象层。
"""

from pathlib import Path


def prompt_path(prompts_dir: Path, name: str) -> Path:
    """在给定 prompts 目录下构造 prompt 文件路径。"""
    return prompts_dir / name


def read_prompt(path: Path) -> str:
    """读取 prompt 文件；缺失时返回最小 JSON-only 兜底提示。"""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return "Return strict JSON for the requested role."
