from __future__ import annotations

"""
DeepSeek LLM 服务适配模块。

它是什么：
本模块是生产链路唯一的真实 LLM 访问入口，负责把各个 LLM 角色的 prompt 发送到
DeepSeek Chat Completions 接口，并返回 JSON 字符串或自然语言文本。

它不是什么：
它不是 mock，不做离线语义兜底，不做关键词判断，不生成本地 intent，不提取本地 memory，
也不在 API 未配置或调用失败时假装智能。

为什么存在：
当前 Agent 是 LLM-first 架构。语义理解、规划、总结和长期记忆候选提取都必须交给
DeepSeek；代码只负责 schema validation、deterministic safety boundary、持久化和执行。
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LLMService:
    """生产 DeepSeek LLM 客户端。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float = 15.0,
        env_path: str | Path | None = None,
    ) -> None:
        env_values = _load_env_file(env_path)
        self.api_key = api_key or env_values.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (
            base_url
            or env_values.get("DEEPSEEK_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com/v1"
        ).rstrip("/")
        self.model = model or env_values.get("DEEPSEEK_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout_sec = float(timeout_sec)
        self._is_configured()

    def complete_json(self, role: str, prompt: str) -> str:
        """为指定 LLM 角色生成 JSON 字符串；失败时抛错，由上层边界处理。"""

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {role} in an embedded LLM-centered agent. "
                    "Return exactly one JSON object. Do not include markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.chat_completion(messages, temperature=0.1)

    def generate_reply(self, text: str, state: object | None = None) -> str:
        """生成用户可见文本；只调用 DeepSeek，不做本地回复 fallback。"""

        del state
        messages = [
            {
                "role": "system",
                "content": (
                    "You write short, natural text for an embedded focus assistant. "
                    "Do not invent device actions or claim state changes."
                ),
            },
            {"role": "user", "content": text},
        ]
        return self.chat_completion(messages, temperature=0.4)

    def chat_completion(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        """调用 DeepSeek Chat Completions 并提取首条文本。"""

        self._is_configured()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        response_data = self._post_json("/chat/completions", payload)
        return self._extract_message_content(response_data)

    def _is_configured(self) -> bool:
        if not self.api_key or not self.base_url or not self.model:
            raise RuntimeError("DeepSeek API is not configured.")
        return True

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek API connection failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("DeepSeek API returned invalid JSON") from exc

    def _extract_message_content(self, response_data: dict[str, Any]) -> str:
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("DeepSeek API response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("DeepSeek API response missing message")
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            merged = "".join(parts).strip()
            if merged:
                return merged
        raise RuntimeError("DeepSeek API response missing text content")


def _load_env_file(env_path: str | Path | None) -> dict[str, str]:
    candidates = [Path(env_path)] if env_path is not None else _default_env_candidates()
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        return _parse_env_text(path.read_text(encoding="utf-8"))
    return {}


def _default_env_candidates() -> list[Path]:
    """按启动目录向上查找 `.env`，支持从项目根目录或 `src/` 启动。"""

    candidates: list[Path] = []
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidates.append(parent / ".env")
    candidates.append(Path(__file__).resolve().parents[2] / ".env")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
            value = value[1:-1]
        values[key.strip()] = value
    return values
