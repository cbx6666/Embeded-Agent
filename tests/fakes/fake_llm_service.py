from __future__ import annotations

import json
from typing import Any


class FakeLLMService:
    """测试专用 LLM fake。

    这个类只存在于 tests/，用于把旧的离线 mock 行为从生产 LLMService 中移走。
    """

    def __init__(
        self,
        responses: dict[str, list[str] | str] | None = None,
        *,
        reply_text: str | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.reply_text = reply_text
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []
        self.reply_calls = 0

    def complete_json(self, role: str, prompt: str) -> str:
        self.calls.append(role)
        self.prompts.append((role, prompt))
        value = self.responses.get(role)
        if isinstance(value, list):
            if value:
                return value.pop(0)
        elif isinstance(value, str):
            return value
        return _fake_complete_json(role, prompt)

    def generate_reply(self, text: str, state: object | None = None) -> str:
        del state
        self.reply_calls += 1
        if self.reply_text is not None:
            return self.reply_text
        return f"fallback reply: {text[:20]}"


class CapturingFakeLLMService(FakeLLMService):
    """语义化别名：FakeLLMService 本身已经捕获 prompts。"""


def _fake_complete_json(role: str, prompt: str) -> str:
    if role == "situation_analyst":
        context = _context_from_prompt(prompt)
        event = context.get("event", {}) if isinstance(context, dict) else {}
        event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
        return json.dumps(
            {
                "summary": _summary_for_event(event_type),
                "user_intent": "interpreted by test fake LLM",
                "current_state": "compact state was inspected",
                "risks": [],
                "uncertainties": [],
                "should_respond": event_type in {"user_text_input", "speech_recognized", "timer_finished"},
                "risk_level": "low",
            },
            ensure_ascii=False,
        )

    if role == "intent_planner":
        intent = _intent_from_prompt(prompt)
        return json.dumps(
            {
                "intents": [intent],
                "reasoning": "Test fake selected an intent from event context.",
                "risk_level": "low",
                "interrupt_user": intent["type"] in {"suggest_rest", "remind_distraction"},
                "response_requirements": {},
            },
            ensure_ascii=False,
        )

    if role == "safety_critic":
        return json.dumps({"decision": "approve", "reason": "No safety conflict found.", "revised_plan": None})

    if role == "response_writer":
        text = "我明白了，会尽量用简单、温和的方式回应。"
        if "timer_finished" in prompt:
            text = "这轮专注完成了。"
        elif "suggest_rest" in prompt:
            text = "你已经专注一会儿了，要不要稍微休息一下？"
        elif "start_focus" in prompt:
            text = "已开始专注。"
        return json.dumps({"speak_text": text, "display_text": text, "tone": "calm"}, ensure_ascii=False)

    if role == "fast_dialogue":
        text = _fast_dialogue_reply_from_prompt(prompt)
        return json.dumps({"speak_text": text, "display_text": text, "tone": "calm"}, ensure_ascii=False)

    if role == "unified_planner":
        context = _context_from_prompt(prompt)
        event = context.get("event", {}) if isinstance(context, dict) else {}
        event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
        intent = _intent_from_prompt(prompt)
        response_text = ""
        if intent["type"] == "answer_user":
            response_text = _fast_dialogue_reply_from_prompt(prompt)
        return json.dumps(
            {
                "situation": {
                    "summary": _summary_for_event(event_type),
                    "user_intent": "interpreted by unified test planner",
                    "current_state": "compact state was inspected",
                    "risks": [],
                    "uncertainties": [],
                    "should_respond": bool(response_text),
                    "risk_level": "low",
                },
                "plan": {
                    "intents": [intent],
                    "reasoning": "Unified test planner selected an intent.",
                    "risk_level": "low",
                    "interrupt_user": intent["type"] in {"suggest_rest", "remind_distraction"},
                    "response_requirements": {},
                },
                "response": {
                    "speak_text": response_text,
                    "display_text": response_text,
                    "tone": "calm",
                },
            },
            ensure_ascii=False,
        )

    if role == "memory_observer":
        context = _context_from_prompt(prompt)
        event = context.get("event", {}) if isinstance(context, dict) else {}
        event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
        worth = event_type in {"user_text_input", "speech_recognized"} or bool(context.get("outcome"))
        return json.dumps({"worth_remembering": worth, "reason": "test fake memory observation"}, ensure_ascii=False)

    if role == "memory_extractor":
        candidate = _memory_item_from_prompt(prompt)
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


def _fast_dialogue_reply_from_prompt(prompt: str) -> str:
    context = _context_from_prompt(prompt)
    state = context.get("state", {}) if isinstance(context, dict) else {}
    focus = state.get("focus", {}) if isinstance(state, dict) else {}
    if isinstance(focus, dict) and focus.get("active"):
        remaining = focus.get("remaining_sec", 0)
        return f"你还在专注中，大约还剩 {remaining} 秒。"
    return "好的，我在。"


def _context_from_prompt(prompt: str) -> dict[str, Any]:
    for marker in ("Context JSON:\n", "## 结构化上下文 JSON\n"):
        start = prompt.rfind(marker)
        if start == -1:
            continue
        chunk = prompt[start + len(marker) :]
        if marker.startswith("##"):
            chunk = chunk.split("\n\n## 用户本轮输入", 1)[0].strip()
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return _extract_context_json(prompt)


def _extract_context_json(prompt: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidate: dict[str, Any] | None = None
    for index, char in enumerate(prompt):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(prompt[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "event" not in data:
            continue
        if "user_id" in data or "state" in data:
            candidate = data
    return candidate or {}


def _summary_for_event(event_type: str) -> str:
    if event_type in {"user_text_input", "speech_recognized"}:
        return "The user is directly interacting with the assistant."
    if event_type == "timer_finished":
        return "A focus timer finished."
    return "An agent event occurred."


def _intent_from_prompt(prompt: str) -> dict[str, Any]:
    context = _context_from_prompt(prompt)
    event = context.get("event", {}) if isinstance(context, dict) else {}
    state = context.get("state", {}) if isinstance(context, dict) else {}
    event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
    event_payload = event.get("payload", {}) if isinstance(event, dict) else {}
    trigger = str(event_payload.get("trigger", "")) if isinstance(event_payload, dict) else ""

    if event_type == "system_triggered":
        if trigger == "focus_health_check" and _focus_needs_rest(state):
            return {"type": "suggest_rest", "priority": 60, "reason": "fatigue during focus", "payload": {}, "requires_llm": False}
        if trigger == "environment_check":
            return {"type": "adjust_environment_feedback", "priority": 60, "reason": "environment check", "payload": {}, "requires_llm": False}
        return {"type": "no_op", "priority": 0, "reason": "system trigger has no fake action", "payload": {}, "requires_llm": False}
    if event_type == "timer_finished":
        return {"type": "complete_focus", "priority": 80, "reason": "timer complete", "payload": {}, "requires_llm": False}
    if event_type == "focus_start_requested":
        return {"type": "start_focus", "priority": 80, "reason": "explicit focus event", "payload": {}, "requires_llm": False}
    if event_type == "focus_stop_requested":
        return {"type": "stop_focus", "priority": 80, "reason": "explicit stop event", "payload": {}, "requires_llm": False}
    if event_type == "user_fatigue_updated":
        return {"type": "suggest_rest", "priority": 60, "reason": "fatigue event", "payload": {}, "requires_llm": False}
    if event_type in {"user_text_input", "speech_recognized"}:
        return {
            "type": "answer_user",
            "priority": 50,
            "reason": "direct user message",
            "payload": {"response_mode": "dialogue"},
            "requires_llm": True,
        }
    return {"type": "no_op", "priority": 0, "reason": "no useful action", "payload": {}, "requires_llm": False}


def _focus_needs_rest(state: object) -> bool:
    if not isinstance(state, dict):
        return False
    focus = state.get("focus", {})
    user = state.get("user", {})
    if not isinstance(focus, dict) or not isinstance(user, dict):
        return False
    return bool(focus.get("active")) and str(user.get("fatigue_level")) == "high"


def _memory_item_from_prompt(prompt: str) -> dict[str, Any] | None:
    context = _context_from_prompt(prompt)
    event = context.get("event", {}) if isinstance(context, dict) else {}
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type", ""))
    payload = event.get("payload", {})
    text = str(payload.get("text", "") if isinstance(payload, dict) else "")
    if not _looks_like_preference(text):
        return None
    return {
        "memory_type": "behavior_preference",
        "content": "User expressed a durable preference in the latest interaction.",
        "confidence": 0.7,
        "evidence": [
            {
                "source_event_type": event_type,
                "timestamp": event.get("timestamp"),
                "user_text": text,
                "source": "dialogue",
            }
        ],
        "source": "llm",
        "metadata": {"preference_key": "interaction_style", "preference_value": "user_stated_preference"},
    }


def _looks_like_preference(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in [
            "prefer",
            "like",
            "dislike",
            "don't",
            "do not",
            "remind",
            "style",
            "喜欢",
            "不喜欢",
            "提醒",
            "温和",
        ]
    )


def _extract_last_json(text: str) -> str:
    decoder = json.JSONDecoder()
    latest = "{}"
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "new" in data:
            return text[index : index + end]
        latest = text[index : index + end]
    return latest
