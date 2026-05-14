from __future__ import annotations

"""Shared utilities for single-machine runtime experiments."""

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.state import AgentState
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import UserProfileService
from src.storage.json_store import JsonStore
from src.storage.long_term_memory_store import LongTermMemoryStore
from src.storage.user_profile_store import UserProfileStore


def prepare_output_dir(experiment_name: str, output_root: str | Path | None = None) -> Path:
    base = Path(output_root) if output_root else ROOT / "scripts" / "runtime_experiments" / "output"
    output_dir = base / experiment_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    return output_dir


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def ev(event_type: str, timestamp: int, **payload: Any) -> Event:
    return Event(type=event_type, timestamp=timestamp, payload=dict(payload))


class ExperimentLLM:
    """Deterministic local LLM double for runtime experiments."""

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

    def complete_json(self, role: str, prompt: str) -> str:
        self.calls.append(role)
        self.prompts.append((role, prompt))
        scripted = self.responses.get(role)
        if isinstance(scripted, list):
            if scripted:
                return scripted.pop(0)
        elif isinstance(scripted, str):
            return scripted
        return _default_complete_json(role, prompt)

    def generate_reply(self, text: str, state: object | None = None) -> str:
        del state
        if self.reply_text is not None:
            return self.reply_text
        return f"我会保持稳定处理：{text[:40]}"


@dataclass
class ExperimentEnvironment:
    output_dir: Path
    llm: ExperimentLLM
    core: AgentCore
    memory_store: LongTermMemoryStore
    profile_service: UserProfileService

    def shutdown(self) -> None:
        self.core.shutdown()


def build_environment(
    experiment_name: str,
    *,
    llm: ExperimentLLM | None = None,
    user_id: str = "default",
    output_root: str | Path | None = None,
) -> ExperimentEnvironment:
    output_dir = prepare_output_dir(experiment_name, output_root)
    memory_store = LongTermMemoryStore(output_dir / "data" / "long_term_memory.json")
    profile_service = UserProfileService(
        UserProfileStore(output_dir / "data" / "user_profiles.json"),
        now_fn=lambda: 0,
    )
    core = AgentCore(
        output=ConsoleOutput(silent=True),
        timer_service=TimerService(background=False),
        runtime_history_service=RuntimeHistoryService(),
        llm_service=llm or ExperimentLLM(),
        store=JsonStore(output_dir / "data" / "runtime_store.json"),
        long_term_memory_pipeline=LongTermMemoryPipeline(memory_store),
        personal_context_builder=PersonalContextBuilder(
            long_term_memory_store=memory_store,
            user_profile_service=profile_service,
        ),
    )
    if user_id != core.state.current_user_id:
        core.switch_user(user_id, timestamp=0)
    return ExperimentEnvironment(
        output_dir=output_dir,
        llm=core.llm_service,
        core=core,
        memory_store=memory_store,
        profile_service=profile_service,
    )


@dataclass
class ExperimentRecorder:
    name: str
    env: ExperimentEnvironment
    events: list[dict[str, Any]] = field(default_factory=list)
    action_timeline: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    memory_snapshots: list[dict[str, Any]] = field(default_factory=list)
    personalization_snapshots: list[dict[str, Any]] = field(default_factory=list)
    blocked_intents: int = 0
    rejected_memory_candidates: int = 0

    def run_event(self, event: Event, *, label: str | None = None) -> list[object]:
        actions, _ = self.env.core.handle_event(event)
        self.events.append(_event_to_dict(event, label=label))
        self._record_after_event(event, actions, label=label or event.type)
        return actions

    def record_snapshot(
        self,
        *,
        label: str,
        timestamp: int,
        user_id: str | None = None,
        event_type: str = "maintenance_snapshot",
        text: str = "",
    ) -> None:
        synthetic = Event(type=event_type, timestamp=timestamp, payload={"text": text, "label": label})
        self._record_memory_snapshot(label=label, timestamp=timestamp)
        self._record_personalization_snapshot(
            event=synthetic,
            label=label,
            user_id=user_id or self.env.core.state.current_user_id,
        )

    def _record_after_event(self, event: Event, actions: list[object], *, label: str) -> None:
        trace = self.env.core.last_runtime_trace.to_dict() if self.env.core.last_runtime_trace else {"events": []}
        decision = self.env.core.last_decision_result
        self.blocked_intents += len(decision.blocked_intents) if decision else 0
        self.rejected_memory_candidates += _trace_rejected_memory_count(trace)

        index = len(self.action_timeline) + 1
        self.action_timeline.append(
            {
                "index": index,
                "label": label,
                "event": _event_to_dict(event),
                "current_user_id": self.env.core.state.current_user_id,
                "actions": [_action_to_dict(action) for action in actions],
            }
        )
        self.traces.append({"index": index, "label": label, "trace": trace})
        self._record_memory_snapshot(label=label, timestamp=event.timestamp)
        self._record_personalization_snapshot(
            event=event,
            label=label,
            user_id=self.env.core.state.current_user_id,
        )

    def _record_memory_snapshot(self, *, label: str, timestamp: int) -> None:
        memories = [item.to_dict() for item in self.env.memory_store.list(include_inactive=True)]
        self.memory_snapshots.append(
            {
                "label": label,
                "timestamp": timestamp,
                "memory_count": len(memories),
                "active_count": sum(1 for item in memories if item.get("status") == "active"),
                "contradicted_count": sum(1 for item in memories if item.get("status") == "contradicted"),
                "memories": memories,
            }
        )

    def _record_personalization_snapshot(self, *, event: Event, label: str, user_id: str) -> None:
        personal_context = self.env.core.personal_context_builder.build(
            user_id=user_id,
            state=self.env.core.state,
            event=event,
        )
        text = str(event.payload.get("text", ""))
        retrieved = personal_context.retrieve_relevant(event_type=str(event.type), text=text, limit=8)
        self.personalization_snapshots.append(
            {
                "label": label,
                "timestamp": event.timestamp,
                "user_id": user_id,
                "profile": personal_context.user_profile,
                "personal_context": personal_context.to_dict(),
                "retrieval_query": {"event_type": event.type, "text": text},
                "retrieved": retrieved,
                "retrieval_size": len(retrieved),
            }
        )

    def metrics(self) -> dict[str, Any]:
        retrieval_sizes = [item["retrieval_size"] for item in self.personalization_snapshots]
        memories = self.memory_snapshots[-1]["memories"] if self.memory_snapshots else []
        return {
            "event_count": len(self.events),
            "memory_count": len(memories),
            "active_memory_count": sum(1 for item in memories if item.get("status") == "active"),
            "contradicted_memory_count": sum(1 for item in memories if item.get("status") == "contradicted"),
            "average_retrieval_size": round(sum(retrieval_sizes) / len(retrieval_sizes), 3)
            if retrieval_sizes
            else 0.0,
            "blocked_intents": self.blocked_intents,
            "rejected_memory_candidates": self.rejected_memory_candidates,
            "action_count": sum(len(item["actions"]) for item in self.action_timeline),
        }

    def write_outputs(self, *, title: str, notes: list[str] | None = None) -> Path:
        out = self.env.output_dir
        write_json(out / "events.json", self.events)
        write_json(out / "action_timeline.json", self.action_timeline)
        write_json(out / "trace_logs.json", self.traces)
        write_json(out / "memory_snapshots.json", self.memory_snapshots)
        write_json(out / "personalization_snapshots.json", self.personalization_snapshots)
        write_json(out / "metrics.json", self.metrics())
        report = _render_report(
            title=title,
            notes=notes or [],
            metrics=self.metrics(),
            action_timeline=self.action_timeline,
            memory_snapshots=self.memory_snapshots,
            personalization_snapshots=self.personalization_snapshots,
            traces=self.traces,
        )
        (out / "report.md").write_text(report, encoding="utf-8")
        return out


def _default_complete_json(role: str, prompt: str) -> str:
    if role == "situation_analyst":
        context = _context_from_prompt(prompt)
        event = context.get("event", {}) if isinstance(context, dict) else {}
        event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
        return json.dumps(
            {
                "summary": f"Experiment observed {event_type}.",
                "user_intent": "deterministic experiment interpretation",
                "current_state": "state summary inspected",
                "risks": [],
                "uncertainties": [],
                "should_respond": event_type in {"user_text_input", "speech_recognized", "timer_finished"},
                "risk_level": "low",
            },
            ensure_ascii=False,
        )

    if role == "intent_planner":
        context = _context_from_prompt(prompt)
        return json.dumps(_intent_plan_from_context(context), ensure_ascii=False)

    if role == "safety_critic":
        return json.dumps({"decision": "approve", "reason": "experiment deterministic approval", "revised_plan": None})

    if role == "response_writer":
        return json.dumps(_response_from_prompt(prompt), ensure_ascii=False)

    if role == "memory_observer":
        context = _memory_context_from_prompt(prompt)
        return json.dumps(
            {
                "worth_remembering": bool(_memory_candidates_from_context(context)),
                "reason": "experiment durable preference detector",
            },
            ensure_ascii=False,
        )

    if role == "memory_extractor":
        context = _memory_context_from_prompt(prompt)
        return json.dumps({"candidates": _memory_candidates_from_context(context)}, ensure_ascii=False)

    if role == "memory_critic":
        candidates = _candidates_from_critic_prompt(prompt)
        return json.dumps(
            {
                "approved_indexes": list(range(len(candidates))),
                "rejected_reasons": [],
            },
            ensure_ascii=False,
        )

    if role == "memory_consolidator":
        data = _json_after_marker(prompt, "Existing and New Candidates JSON:\n")
        new_candidates = data.get("new", []) if isinstance(data, dict) else []
        return json.dumps({"candidates": new_candidates}, ensure_ascii=False)

    return "{}"


def _intent_plan_from_context(context: dict[str, Any]) -> dict[str, Any]:
    event = context.get("event", {}) if isinstance(context, dict) else {}
    state = context.get("state", {}) if isinstance(context, dict) else {}
    event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
    payload = event.get("payload", {}) if isinstance(event, dict) else {}
    trigger = str(payload.get("trigger", "")) if isinstance(payload, dict) else ""
    user_text = str(event.get("user_text", "")).lower() if isinstance(event, dict) else ""

    intent = {"type": "no_op", "priority": 0, "reason": "no deterministic action", "payload": {}, "requires_llm": False}
    if event_type in {"user_text_input", "speech_recognized"} and user_text:
        intent = {
            "type": "answer_user",
            "priority": 50,
            "reason": "direct user interaction",
            "payload": {"response_mode": "dialogue"},
            "requires_llm": True,
        }
    elif event_type == "focus_start_requested":
        intent = {"type": "start_focus", "priority": 90, "reason": "user started study session", "payload": {}, "requires_llm": False}
    elif event_type == "focus_stop_requested":
        intent = {"type": "stop_focus", "priority": 90, "reason": "user requested rest", "payload": {}, "requires_llm": False}
    elif event_type == "timer_finished":
        intent = {"type": "complete_focus", "priority": 90, "reason": "study block completed", "payload": {}, "requires_llm": False}
    elif event_type == "user_fatigue_updated":
        intent = {"type": "suggest_rest", "priority": 60, "reason": "fatigue update", "payload": {}, "requires_llm": False}
    elif event_type == "system_triggered" and trigger == "environment_check":
        intent = {"type": "adjust_environment_feedback", "priority": 50, "reason": "environment check", "payload": {}, "requires_llm": False}
    elif event_type == "system_triggered" and trigger == "focus_health_check":
        focus = state.get("focus", {}) if isinstance(state, dict) else {}
        user = state.get("user", {}) if isinstance(state, dict) else {}
        if isinstance(user, dict) and user.get("attention") == "distracted":
            intent = {"type": "remind_distraction", "priority": 70, "reason": "attention drift", "payload": {}, "requires_llm": False}
        if isinstance(focus, dict) and focus.get("active") and isinstance(user, dict) and user.get("fatigue_level") == "high":
            intent = {"type": "suggest_rest", "priority": 80, "reason": "fatigue during focus", "payload": {}, "requires_llm": False}

    return {
        "intents": [intent],
        "reasoning": "Deterministic experiment planner.",
        "risk_level": "low",
        "interrupt_user": intent["type"] in {"suggest_rest", "remind_distraction", "adjust_environment_feedback"},
        "response_requirements": {},
    }


def _response_from_prompt(prompt: str) -> dict[str, str]:
    lowered = prompt.lower()
    text = "我会稳定执行，并保持记录。"
    if "suggest_rest" in lowered:
        text = "你已经专注一会儿了，我轻轻提醒你可以休息一下。"
    elif "remind_distraction" in lowered:
        text = "我们慢慢把注意力带回当前任务。"
    elif "start_focus" in lowered:
        text = "已开始专注。"
    elif "complete_focus" in lowered:
        text = "这轮专注完成了，可以休息一下。"
    elif "direct" in lowered:
        text = "收到，我会更直接地提醒你。"
    elif "gentle" in lowered or "温和" in prompt:
        text = "收到，我会用更温和、低打扰的方式提醒你。"
    return {"speak_text": text, "display_text": text, "tone": "calm"}


def _memory_candidates_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    event = context.get("event", {}) if isinstance(context, dict) else {}
    if not isinstance(event, dict):
        return []
    event_type = str(event.get("type", ""))
    payload = event.get("payload", {})
    text = str(payload.get("text", "") if isinstance(payload, dict) else "")
    lowered = text.lower()

    if "[weak_evidence]" in lowered:
        return [
            {
                "memory_type": "behavior_preference",
                "content": "Weak evidence claims the user likes loud reminders.",
                "confidence": 0.72,
                "evidence": [{"source": "llm"}],
                "metadata": {"preference_key": "reminder_style", "preference_value": "loud"},
            }
        ]

    key = ""
    value = ""
    content = ""
    if "温和" in text or "gentle" in lowered or "轻" in text:
        key, value = "reminder_style", "gentle"
        content = "User prefers gentle reminders."
    if "直接" in text or "direct" in lowered:
        key, value = "reminder_style", "direct"
        content = "User prefers direct reminders."
    if "少提醒" in text or "不喜欢频繁" in text or "less" in lowered or "low frequency" in lowered:
        key, value = "reminder_frequency", "low"
        content = "User prefers low-frequency reminders."
    if "主动提醒" in text or "more proactive" in lowered or "frequent reminders" in lowered:
        key, value = "reminder_frequency", "high"
        content = "User now prefers more proactive reminders."

    if not key:
        return []
    return [
        {
            "memory_type": "behavior_preference",
            "content": content,
            "confidence": 0.78,
            "evidence": [
                {
                    "source_event_type": event_type,
                    "timestamp": event.get("timestamp"),
                    "source": "dialogue",
                    "user_text": text,
                }
            ],
            "metadata": {"preference_key": key, "preference_value": value},
        }
    ]


def _context_from_prompt(prompt: str) -> dict[str, Any]:
    data = _json_after_marker(prompt, "Context JSON:\n")
    return data if isinstance(data, dict) else {}


def _memory_context_from_prompt(prompt: str) -> dict[str, Any]:
    data = _json_after_marker(prompt, "MemoryContext JSON:\n")
    return data if isinstance(data, dict) else {}


def _candidates_from_critic_prompt(prompt: str) -> list[dict[str, Any]]:
    data = _json_after_marker(prompt, "Candidates JSON:\n")
    return data if isinstance(data, list) else []


def _json_after_marker(prompt: str, marker: str) -> object:
    start = prompt.rfind(marker)
    if start == -1:
        return _extract_last_json(prompt)
    text = prompt[start + len(marker) :].strip()
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(text)
        return data
    except json.JSONDecodeError:
        return {}


def _extract_last_json(text: str) -> object:
    decoder = json.JSONDecoder()
    latest: object = {}
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        latest = data
    return latest


def _event_to_dict(event: Event, *, label: str | None = None) -> dict[str, Any]:
    data = {"type": event.type, "timestamp": event.timestamp, "payload": dict(event.payload)}
    if label:
        data["label"] = label
    return data


def _action_to_dict(action: object) -> dict[str, Any]:
    return {"type": getattr(action, "type", ""), "payload": dict(getattr(action, "payload", {}) or {})}


def _trace_rejected_memory_count(trace: dict[str, Any]) -> int:
    count = 0
    for item in trace.get("events", []):
        if not isinstance(item, dict) or item.get("stage") != "memory_pipeline":
            continue
        payload = item.get("payload", {})
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        rejected = result.get("rejected", []) if isinstance(result, dict) else []
        count += len(rejected) if isinstance(rejected, list) else 0
    return count


def _render_report(
    *,
    title: str,
    notes: list[str],
    metrics: dict[str, Any],
    action_timeline: list[dict[str, Any]],
    memory_snapshots: list[dict[str, Any]],
    personalization_snapshots: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> str:
    lines = [f"# {title}", ""]
    if notes:
        lines.extend(["## Notes", *[f"- {note}" for note in notes], ""])

    lines.extend(["## Runtime Metrics"])
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Action Timeline"])
    for item in action_timeline:
        action_types = [action["type"] for action in item["actions"]]
        lines.append(
            f"- #{item['index']} {item['event']['timestamp']} {item['event']['type']} "
            f"user={item['current_user_id']} actions={action_types}"
        )

    lines.extend(["", "## Memory Evolution"])
    seen: set[str] = set()
    for snapshot in memory_snapshots:
        memories = snapshot["memories"]
        new_ids = [item["id"] for item in memories if item["id"] not in seen]
        seen.update(item["id"] for item in memories)
        top = memories[-3:]
        lines.append(
            f"- {snapshot['label']} active={snapshot['active_count']} "
            f"contradicted={snapshot['contradicted_count']} new={new_ids}"
        )
        for memory in top:
            lines.append(
                f"  - {memory['status']} {memory['memory_type']} conf={memory['confidence']:.3f} "
                f"decay={memory['decay']:.3f} content={memory['content']}"
            )

    lines.extend(["", "## Personalization Evolution"])
    for snapshot in personalization_snapshots[-10:]:
        retrieved = [item.get("content", "") for item in snapshot["retrieved"][:3]]
        compression = snapshot["personal_context"].get("compression", {})
        lines.append(
            f"- {snapshot['label']} user={snapshot['user_id']} retrieval_size={snapshot['retrieval_size']} "
            f"compression={compression.get('output_counts')} selected={retrieved}"
        )

    lines.extend(["", "## Trace Samples"])
    for item in traces[-5:]:
        summary = _trace_summary(item["trace"])
        lines.append(f"- #{item['index']} {item['label']}: {summary}")
    lines.append("")
    return "\n".join(lines)


def _trace_summary(trace: dict[str, Any]) -> str:
    parts: list[str] = []
    for event in trace.get("events", []):
        if not isinstance(event, dict):
            continue
        stage = event.get("stage")
        label = event.get("label")
        if stage in {"event", "prompt", "llm_output", "validator", "guard", "action"}:
            parts.append(f"{stage}:{label}")
    return " -> ".join(parts)


def state_from_store(path: str | Path | None, *, user_id: str = "default") -> AgentState:
    if path is None:
        return AgentState(current_user_id=user_id)
    return AgentState.from_dict(JsonStore(path).load_state_dict())
