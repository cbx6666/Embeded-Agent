from __future__ import annotations

"""大模型服务模块。"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.state import AgentState


class LLMService:
    """提供回复生成与意图选择的 LLM 服务。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        env_path: str | Path | None = None,
    ) -> None:
        """初始化 LLM 配置；未显式传入时优先从 .env 和环境变量读取。"""
        env_values = _load_env_file(env_path)

        self.api_key = (
            api_key
            or env_values.get("EMBEDED_AGENT_LLM_API_KEY")
            or env_values.get("DEEPSEEK_API_KEY")
            or env_values.get("OPENAI_API_KEY")
            or os.environ.get("EMBEDED_AGENT_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.base_url = (
            base_url
            or env_values.get("EMBEDED_AGENT_LLM_BASE_URL")
            or env_values.get("DEEPSEEK_BASE_URL")
            or env_values.get("OPENAI_BASE_URL")
            or os.environ.get("EMBEDED_AGENT_LLM_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.deepseek.com/v1"
        ).rstrip("/")
        self.model = (
            model
            or env_values.get("EMBEDED_AGENT_LLM_MODEL")
            or env_values.get("DEEPSEEK_MODEL")
            or env_values.get("OPENAI_MODEL")
            or os.environ.get("EMBEDED_AGENT_LLM_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "deepseek-chat"
        )
        raw_timeout = (
            timeout_sec
            or env_values.get("EMBEDED_AGENT_LLM_TIMEOUT_SEC")
            or os.environ.get("EMBEDED_AGENT_LLM_TIMEOUT_SEC")
            or "20"
        )
        self.timeout_sec = float(raw_timeout)

    def generate_reply(self, text: str, state: AgentState) -> str:
        """根据输入文本和状态生成自然语言回复。"""
        if not self._is_configured():
            return self._mock_generate_reply(text, state)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个嵌入式专注辅助 agent 的语言回复模块。"
                    "请保持简洁、自然、礼貌，不要编造硬件动作，不要修改状态，"
                    "只输出给用户看的回复文本。"
                ),
            },
            {"role": "user", "content": text},
        ]

        try:
            return self._chat_completion(messages, temperature=0.4)
        except Exception:
            return self._mock_generate_reply(text, state)

    def choose_intents(
        self,
        prompt: str,
        allowed_intent_types: list[str],
    ) -> str:
        """在允许的意图范围内返回一个 JSON 格式的选择结果。"""
        if not self._is_configured():
            return self._mock_choose_intents(prompt, allowed_intent_types)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个受限的 Intent 选择器。"
                    "你只能从允许的 intent type 中选择，不能创造新类型，"
                    "不能输出动作，不能修改状态。"
                    "请严格返回 JSON，格式为 "
                    '{"intents":[{"type":"answer_user","priority":10,"reason":"...","payload":{},"requires_llm":true}]}。'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"允许的 intent type: {allowed_intent_types}\n"
                    f"{prompt}\n"
                    "请只返回 JSON，不要输出解释。"
                ),
            },
        ]

        try:
            return self._chat_completion(messages, temperature=0.1)
        except Exception:
            return self._mock_choose_intents(prompt, allowed_intent_types)

    def _is_configured(self) -> bool:
        """判断当前是否具备真实 API 调用所需配置。"""
        return bool(
            getattr(self, "api_key", "")
            and getattr(self, "base_url", "")
            and getattr(self, "model", "")
        )

    def _chat_completion(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        """调用兼容 OpenAI Chat Completions 的接口并返回文本内容。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        response_data = self._post_json("/chat/completions", payload)
        return self._extract_message_content(response_data)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向 LLM 接口发送 JSON 请求并解析 JSON 响应。"""
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
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:  # pragma: no cover - 依赖外部网络
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - 依赖外部网络
            raise RuntimeError(f"LLM API 连接失败: {exc.reason}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - 依赖外部网络
            raise RuntimeError("LLM API 返回了非法 JSON。") from exc

    def _extract_message_content(self, response_data: dict[str, Any]) -> str:
        """从 Chat Completions 响应里提取第一条文本内容。"""
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM API 返回中缺少 choices。")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM API 返回中缺少 message。")

        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            merged = "".join(text_parts).strip()
            if merged:
                return merged

        raise RuntimeError("LLM API 返回中缺少可用文本内容。")

    def _mock_generate_reply(self, text: str, state: AgentState) -> str:
        """在未配置真实 API 时提供本地回退回复。"""
        lowered = text.strip().lower()
        if any(word in lowered for word in ("你好", "hello", "hi")):
            return "你好，我已在线。你可以让我开始专注、结束专注，或用 /mock 更新状态。"
        if "谢谢" in lowered or "thanks" in lowered:
            return "收到。当前版本可以帮助你管理专注、记录状态并给出简单提醒。"
        if state.focus.active:
            return "收到。当前你处于专注模式，我会尽量保持简洁。"
        return "收到。当前 MVP 版本支持专注计时、mock 状态更新、基础记忆和简单提醒。"

    def _mock_choose_intents(
        self,
        prompt: str,
        allowed_intent_types: list[str],
    ) -> str:
        """在未配置真实 API 时提供本地回退意图选择。"""
        lowered = prompt.strip().lower()

        selected_type = "answer_user"
        requires_llm = selected_type == "answer_user"
        reason = "用户输入是开放式对话，需要自然语言回复"

        if any(word in lowered for word in ("累", "疲惫", "困", "休息")) and "suggest_rest" in allowed_intent_types:
            selected_type = "suggest_rest"
            requires_llm = False
            reason = "用户表达了疲劳或休息需求"
        elif any(word in lowered for word in ("继续学", "开始专注", "start focus")) and "start_focus" in allowed_intent_types:
            selected_type = "start_focus"
            requires_llm = False
            reason = "用户表达了继续学习或开始专注的意图"
        elif "answer_user" not in allowed_intent_types and allowed_intent_types:
            selected_type = allowed_intent_types[0]
            requires_llm = False
            reason = "在允许范围内选择首个可用意图"

        return json.dumps(
            {
                "intents": [
                    {
                        "type": selected_type,
                        "priority": 10,
                        "reason": reason,
                        "payload": {"llm_selected": True},
                        "requires_llm": requires_llm,
                    }
                ]
            },
            ensure_ascii=False,
        )


def _load_env_file(env_path: str | Path | None) -> dict[str, str]:
    """从 .env 文件读取键值对，不依赖第三方库。"""
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(Path(env_path))
    else:
        candidates.append(Path.cwd() / ".env")

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        return _parse_env_text(path.read_text(encoding="utf-8"))
    return {}


def _parse_env_text(text: str) -> dict[str, str]:
    """解析 .env 文本内容。"""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values
