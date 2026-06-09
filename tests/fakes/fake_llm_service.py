from __future__ import annotations

"""测试专用 LLM fake。

生产 ``LLMClient`` 通过 ``complete_json(role, prompt) -> str`` 调用底层服务。
决策 role：`speech_recognized`、`behavior_distraction_check`、`wellness_care_check`、
`environment_care_check`；另有后台 `memory_extract`。
`sensor_status_report` 不走 LLM。
按 role 返回预设 JSON 并记录调用。
"""

import json
from collections.abc import Callable
from typing import Any, Union

Responder = Union[str, Callable[[str], str]]


class FakeLLMService:
    def __init__(self, responses: dict[str, Responder] | None = None) -> None:
        self.responses: dict[str, Responder] = dict(responses or {})
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    def set_response(self, role: str, payload: dict[str, Any] | str) -> None:
        self.responses[role] = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    def complete_json(self, role: str, prompt: str, *, temperature: float | None = None) -> str:
        del temperature
        self.calls.append(role)
        self.prompts.append((role, prompt))
        responder = self.responses.get(role)
        if callable(responder):
            return responder(prompt)
        if isinstance(responder, str):
            return responder
        if role == "speech_recognized":
            return json.dumps({"intent": "answer_user", "reply": "好的，我在。"}, ensure_ascii=False)
        if role == "memory_extract":
            return json.dumps({"memory_items": []}, ensure_ascii=False)
        return json.dumps({"intent": "no_op", "reply": ""}, ensure_ascii=False)
