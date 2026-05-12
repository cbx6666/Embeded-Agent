from __future__ import annotations

"""
LLM 服务适配模块。

本文件位于所有 LLM 角色的最底层，负责把角色 prompt 发送给
OpenAI 风格 Chat Completions 接口，并在离线或失败时提供可解释的本地
mock 输出。它的上游是 `llm_agent/roles/*` 和 `memory/llm_memory_manager.py`，
下游是外部 LLM API。

本模块不理解业务流程、不修改 AgentState、不写 MemoryStore，也不生成 Action。
它只返回文本或 JSON 字符串，所有语义约束都必须由上层 schema validator 和
deterministic boundary 再次校验。
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LLMService:
    """角色级 LLM 调用服务。

    输入是角色名和 prompt，输出是模型返回的 JSON 字符串或自然语言文本。
    它不负责判断模型输出是否可信；调用方必须继续做 schema validation、
    intent/action 白名单校验和安全过滤。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        env_path: str | Path | None = None,
    ) -> None:
        """读取 LLM 配置。

        显式参数优先，其次读取项目 `.env` 和系统环境变量。未配置 API key
        时不会访问网络，而是进入本地 mock，保证嵌入式开发和测试可离线运行。
        """

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
            or "5"
        )
        self.timeout_sec = float(raw_timeout)

    def complete_json(self, role: str, prompt: str) -> str:
        """为指定 LLM 角色生成 JSON 字符串。

        外部 API 失败时返回本地 mock JSON，而不是抛出到主循环；这样主链路可以
        继续通过 validator/guard 形成安全 fallback。
        """

        if not self._is_configured():
            return self._mock_complete_json(role, prompt)

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
        try:
            return self._chat_completion(messages, temperature=0.1)
        except Exception:
            return self._mock_complete_json(role, prompt)

    def generate_reply(self, text: str, state: object | None = None) -> str:
        """生成用户可见文本。

        该方法只作为 ResponseWriter 的表达 fallback 使用，不参与行为决策。即使
        外部 LLM 不可用，也会返回简短本地文本，避免对话链路空响应。
        """

        del state
        if not self._is_configured():
            return self._mock_generate_reply(text)

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
        try:
            return self._chat_completion(messages, temperature=0.4)
        except Exception:
            return self._mock_generate_reply(text)

    def _is_configured(self) -> bool:
        """判断是否具备真实 LLM API 调用条件；否则走本地 mock。"""

        return bool(self.api_key and self.base_url and self.model)

    def _chat_completion(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        """调用 OpenAI 风格 chat/completions，并提取首条文本。"""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        response_data = self._post_json("/chat/completions", payload)
        return self._extract_message_content(response_data)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 JSON 请求并解析 JSON 响应。

        网络层错误会被包装成 RuntimeError，由上层 `complete_json` 捕获并降级到
        本地 mock，避免外部服务问题拖垮嵌入式主循环。
        """

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
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API connection failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM API returned invalid JSON") from exc

    def _extract_message_content(self, response_data: dict[str, Any]) -> str:
        """兼容常见 Chat Completions 响应格式，提取可用文本。"""

        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM API response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM API response missing message")
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
        raise RuntimeError("LLM API response missing text content")

    def _mock_complete_json(self, role: str, prompt: str) -> str:
        """本地离线 mock。

        mock 只保证链路可运行，不承担真实智能。真实部署时应由外部 LLM 完成语义
        判断，mock 输出仍会经过同一套 validator/guard 边界。
        """

        if role == "situation_analyst":
            return json.dumps(
                {
                    "summary": _mock_summary(prompt),
                    "user_intent": "interpreted by local mock LLM",
                    "current_state": "compact state was inspected",
                    "risks": [],
                    "uncertainties": [],
                    "should_respond": _contains_any(prompt, ["user_text_input", "speech_recognized", "timer_finished"]),
                    "risk_level": "low",
                },
                ensure_ascii=False,
            )

        if role == "intent_planner":
            intent = _mock_intent_from_prompt(prompt)
            return json.dumps(
                {
                    "intents": [intent],
                    "reasoning": "Local mock LLM selected a registered intent from the event context.",
                    "risk_level": "low",
                    "interrupt_user": intent["type"] in {"suggest_rest", "remind_distraction"},
                    "response_requirements": {},
                },
                ensure_ascii=False,
            )

        if role == "safety_critic":
            return json.dumps({"decision": "approve", "reason": "No safety conflict found.", "revised_plan": None})

        if role == "response_writer":
            text = "I understand. I will keep it simple."
            if "timer_finished" in prompt:
                text = "Focus time is complete."
            elif "suggest_rest" in prompt:
                text = "A short rest may help now."
            elif "start_focus" in prompt:
                text = "Focus timer started."
            return json.dumps({"speak_text": text, "display_text": text, "tone": "calm"}, ensure_ascii=False)

        if role == "memory_observer":
            worth = _contains_any(prompt, ["user_text_input", "speech_recognized", "break_suggestion"])
            return json.dumps({"worth_remembering": worth, "reason": "mock memory observation"}, ensure_ascii=False)

        if role == "memory_extractor":
            candidate = _mock_memory_item(prompt)
            return json.dumps({"candidates": [candidate] if candidate else []}, ensure_ascii=False)

        if role == "memory_critic":
            return json.dumps({"approved_indexes": [0], "rejected_reasons": []}, ensure_ascii=False)

        if role == "memory_consolidator":
            try:
                data = json.loads(_extract_last_json(prompt))
                return json.dumps({"candidates": data.get("new", [])}, ensure_ascii=False)
            except Exception:
                return json.dumps({"candidates": []}, ensure_ascii=False)

        return "{}"

    def _mock_generate_reply(self, text: str) -> str:
        """离线文本 fallback；只生成表达，不生成动作承诺。"""

        text = text.strip()
        if not text:
            return "I am here."
        return "I understand. " + text[:80]


def _load_env_file(env_path: str | Path | None) -> dict[str, str]:
    """读取一个 `.env` 文件中的 KEY=VALUE 配置；读取失败时返回空配置。"""

    candidates = [Path(env_path)] if env_path is not None else [Path.cwd() / ".env"]
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        return _parse_env_text(path.read_text(encoding="utf-8"))
    return {}


def _parse_env_text(text: str) -> dict[str, str]:
    """解析最小 `.env` 语法，避免为配置读取引入额外依赖。"""

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


def _contains_any(text: str, needles: list[str]) -> bool:
    """检查 mock prompt 中是否包含任一触发词；仅用于离线兜底。"""

    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _mock_summary(prompt: str) -> str:
    """为 SituationAnalyst mock 生成最小情况摘要。"""

    if "user_text_input" in prompt or "speech_recognized" in prompt:
        return "The user is directly interacting with the assistant."
    if "timer_finished" in prompt:
        return "A focus timer finished."
    return "An agent event occurred."


def _mock_intent_from_prompt(prompt: str) -> dict[str, Any]:
    """为 IntentPlanner mock 生成注册 intent。

    这不是主要智能来源，只是测试和离线运行的安全 fallback；真实语义判断由
    LLM 完成，输出仍要经过 IntentPlanValidator 和 DeterministicGuard。
    """

    if "focus_start_requested" in prompt:
        return {"type": "start_focus", "priority": 80, "reason": "explicit focus event", "payload": {}, "requires_llm": False}
    if "focus_stop_requested" in prompt:
        return {"type": "stop_focus", "priority": 80, "reason": "explicit stop event", "payload": {}, "requires_llm": False}
    if "timer_finished" in prompt:
        return {"type": "complete_focus", "priority": 80, "reason": "timer complete", "payload": {}, "requires_llm": False}
    lowered = prompt.lower()
    if "focus_health_check" in lowered and '"active": true' in lowered and '"fatigue_level": "high"' in lowered:
        return {"type": "suggest_rest", "priority": 60, "reason": "fatigue during focus", "payload": {}, "requires_llm": False}
    if "user_fatigue_updated" in prompt:
        return {"type": "suggest_rest", "priority": 60, "reason": "fatigue event", "payload": {}, "requires_llm": False}
    if "user_text_input" in prompt or "speech_recognized" in prompt:
        return {
            "type": "answer_user",
            "priority": 50,
            "reason": "direct user message",
            "payload": {"response_mode": "dialogue"},
            "requires_llm": True,
        }
    return {"type": "no_op", "priority": 0, "reason": "no useful action", "payload": {}, "requires_llm": False}


def _mock_memory_item(prompt: str) -> dict[str, Any] | None:
    """为 MemoryExtractor mock 生成候选记忆；真实记忆仍需 evidence 校验。"""

    lowered = prompt.lower()
    if not _contains_any(lowered, ["prefer", "like", "dislike", "don't", "do not", "remind", "style"]):
        return None
    return {
        "memory_type": "explicit_preference",
        "content": "User expressed a preference in the latest interaction.",
        "confidence": 0.6,
        "evidence": [{"source": "mock_llm", "snippet": "latest interaction"}],
        "source": "llm",
        "metadata": {},
    }


def _extract_last_json(text: str) -> str:
    """从 prompt 尾部取出 JSON 片段，供 memory_consolidator mock 使用。"""

    start = text.rfind("{")
    if start == -1:
        return "{}"
    return text[start:]
