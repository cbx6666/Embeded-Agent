from __future__ import annotations

"""Agent 自主化验证终端。"""

import argparse
import json
import time
from pathlib import Path

from src.adapters.cli_input import CLIInputAdapter, parse_cli_event
from src.adapters.console_output import ConsoleOutput
from src.adapters.mock_input import parse_mock_command
from src.adapters.profile_cli import handle_profile_command
from src.agent.core import AgentCore, build_default_core
from src.agent.event import Event
from src.agent.runtime.autonomy import build_autonomous_check_event
from src.agent.runtime.loop import AgentLoop

LAB_HELP_TEXT = """可用命令：
  普通文本：
    你好
    开始专注 25 分钟
    结束专注
    现在状态如何

  mock 命令：
    /mock presence present
    /mock presence away
    /mock attention focused
    /mock attention distracted
    /mock attention idle
    /mock behavior working
    /mock behavior phone_use
    /mock behavior staring
    /mock behavior desk_rest
    /mock behavior away
    /mock emotion neutral
    /mock emotion tired
    /mock emotion stressed
    /mock emotion happy
    /mock fatigue none
    /mock fatigue mild
    /mock fatigue moderate
    /mock fatigue high

  自主检查：
    /auto periodic_check
    /auto focus_health_check
    /auto user_idle_check
    /auto environment_check

  内置验证场景：
    /scenarios
    /scenario focus_fatigue_rest
    /scenario focus_distraction
    /scenario away_periodic
    /scenario environment_warning

  调试命令：
    /state
    /history
    /profile
    /users
    /switch_user xiaoli
    /switch_user xiaoli 小李
    /set_info age 12
    /set_info gender 女
    /set_info identity 小学生
    /set_info hobbies 画画,足球
    /set_pref favorite_content_types 音乐,相声,脱口秀
    /set_pref reminder_style 温和
    /set_pref favorite_music_styles 轻音乐,古风
    /trace
    /trace 5
    /last
    /reset
    /help
    /exit
"""

SCENARIO_DESCRIPTIONS = {
    "focus_fatigue_rest": "专注中出现疲劳，再触发自主健康检查，验证是否建议休息。",
    "focus_distraction": "专注中出现分心，再触发自主健康检查，验证是否提醒回到任务。",
    "away_periodic": "用户离场后触发周期检查，验证系统不会主动 speak。",
    "environment_warning": "环境亮度异常后触发环境检查，验证系统优先 display 反馈。",
}


def build_scenario_events(name: str, start_ts: int | None = None) -> list[tuple[str, Event]]:
    """构造内置验证场景对应的一组事件序列。"""
    base_ts = int(start_ts or time.time())
    if name == "focus_fatigue_rest":
        return [
            ("开始专注", Event(type="focus_start_requested", timestamp=base_ts, payload={"duration_sec": 1500, "source": "lab"})),
            ("用户在场", Event(type="user_presence_updated", timestamp=base_ts + 1, payload={"presence": "present", "source": "lab"})),
            (
                "注意力稳定",
                Event(
                    type="user_attention_updated",
                    timestamp=base_ts + 2,
                    payload={"attention": "focused", "behavior": "working", "source": "lab"},
                ),
            ),
            ("疲劳升高", Event(type="user_fatigue_updated", timestamp=base_ts + 3, payload={"fatigue_level": "high", "source": "lab"})),
            (
                "触发专注健康检查",
                Event(
                    type="system_triggered",
                    timestamp=base_ts + 601,
                    payload={"trigger": "focus_health_check", "source": "agent_autonomy"},
                ),
            ),
        ]
    if name == "focus_distraction":
        return [
            ("开始专注", Event(type="focus_start_requested", timestamp=base_ts, payload={"duration_sec": 1200, "source": "lab"})),
            ("用户在场", Event(type="user_presence_updated", timestamp=base_ts + 1, payload={"presence": "present", "source": "lab"})),
            (
                "用户分心",
                Event(
                    type="user_attention_updated",
                    timestamp=base_ts + 2,
                    payload={"attention": "distracted", "behavior": "phone_use", "source": "lab"},
                ),
            ),
            (
                "触发专注健康检查",
                Event(
                    type="system_triggered",
                    timestamp=base_ts + 3,
                    payload={"trigger": "focus_health_check", "source": "agent_autonomy"},
                ),
            ),
        ]
    if name == "away_periodic":
        return [
            ("用户离场", Event(type="user_presence_updated", timestamp=base_ts, payload={"presence": "away", "source": "lab"})),
            (
                "触发周期检查",
                Event(
                    type="system_triggered",
                    timestamp=base_ts + 1,
                    payload={"trigger": "periodic_check", "source": "agent_autonomy"},
                ),
            ),
        ]
    if name == "environment_warning":
        return [
            (
                "环境光变暗",
                Event(
                    type="light_level_updated",
                    timestamp=base_ts,
                    payload={"light_lux": 10, "level": "dark", "source": "lab"},
                ),
            ),
            (
                "触发环境检查",
                Event(
                    type="system_triggered",
                    timestamp=base_ts + 1,
                    payload={"trigger": "environment_check", "source": "agent_autonomy"},
                ),
            ),
        ]
    raise ValueError(f"不支持的场景: {name}")


def create_runtime(store_path: str | Path, max_steps: int, output: ConsoleOutput) -> tuple[AgentCore, AgentLoop]:
    """创建一套用于终端验证的 AgentCore 与 AgentLoop。"""
    profile_store_path = Path(store_path).with_name("user_profiles_lab.json")
    core = build_default_core(
        store_path=store_path,
        profile_store_path=profile_store_path,
        output=output,
    )
    loop = AgentLoop(core, max_steps=max_steps)
    return core, loop


def main() -> None:
    """启动交互式自主化验证终端。"""
    parser = argparse.ArgumentParser(description="Embeded-Agent 验证终端")
    parser.add_argument("--store-path", type=str, default="data/runtime_lab_store.json", help="验证终端使用的状态存储路径")
    parser.add_argument("--max-steps", type=int, default=5, help="单次闭环最大步数")
    args = parser.parse_args()

    output = ConsoleOutput()
    cli = CLIInputAdapter(prompt="lab> ")
    store_path = Path(args.store_path)
    core, loop = create_runtime(store_path, args.max_steps, output)

    output.show_text("Agent 验证终端已启动，输入 /help 查看命令。")
    try:
        while True:
            line = cli.readline()
            if line is None:
                break
            command = line.strip()
            if not command:
                continue

            if command == "/exit":
                break
            if command == "/help":
                output.show_text(LAB_HELP_TEXT.rstrip())
                continue
            if command == "/state":
                output.show_text(core.render_state())
                continue
            if command == "/history":
                output.show_text(core.render_history())
                continue
            if handle_profile_command(core, output, command):
                continue
            if command == "/scenarios":
                _show_scenarios(output)
                continue
            if command.startswith("/trace"):
                _show_recent_traces(output, loop, command)
                continue
            if command == "/last":
                _show_last_decision(output, core)
                continue
            if command == "/reset":
                core.shutdown()
                if store_path.exists():
                    store_path.unlink()
                core, loop = create_runtime(store_path, args.max_steps, output)
                output.show_text("验证终端状态已重置。")
                continue

            if command.startswith("/auto "):
                reason = command.split(maxsplit=1)[1].strip()
                event = build_autonomous_check_event(core.state, now_ts=int(time.time()), reason=reason)
                _run_event(loop, output, event, title=f"自主检查: {reason}")
                continue

            if command.startswith("/scenario "):
                scenario_name = command.split(maxsplit=1)[1].strip()
                _run_scenario(loop, output, scenario_name)
                continue

            try:
                mock_event = parse_mock_command(command)
            except ValueError as exc:
                output.show_text(f"[Error] {exc}")
                continue

            if mock_event is not None:
                _run_event(loop, output, mock_event, title=f"Mock 事件: {mock_event.type}")
                continue

            _run_event(loop, output, parse_cli_event(command), title=f"用户输入: {command}")
    finally:
        core.shutdown()
        output.show_text("Agent 验证终端已退出。")


def _run_scenario(loop: AgentLoop, output: ConsoleOutput, scenario_name: str) -> None:
    """顺序执行一个内置验证场景。"""
    try:
        events = build_scenario_events(scenario_name)
    except ValueError as exc:
        output.show_text(f"[Error] {exc}")
        return

    output.show_text(f"开始场景验证: {scenario_name}")
    for index, (label, event) in enumerate(events, start=1):
        _run_event(loop, output, event, title=f"场景步骤 {index}: {label}")
    output.show_text(f"场景验证结束: {scenario_name}")


def _run_event(loop: AgentLoop, output: ConsoleOutput, event: Event, *, title: str) -> None:
    """执行单个事件并输出本轮闭环的关键验证信息。"""
    output.show_text(f"\n=== {title} ===")
    output.show_text(f"[Event] type={event.type} payload={json.dumps(event.payload, ensure_ascii=False)}")
    start_trace_index = len(loop.recent_traces)
    actions = loop.run_once(event)
    new_traces = loop.recent_traces[start_trace_index:]
    output.show_text(f"[Summary] action_count={len(actions)}")
    for trace in new_traces:
        output.show_text(_format_trace(trace))


def _format_trace(trace: object) -> str:
    """将 trace 对象格式化为便于终端查看的摘要文本。"""
    event_type = getattr(trace, "event_type", "unknown")
    loop_step = getattr(trace, "loop_step", 0)
    intents = getattr(trace, "intents", [])
    actions = getattr(trace, "actions", [])
    results = getattr(trace, "results", [])
    intent_summary = _summarize_intents_for_trace(intents)
    action_types = [item.get("type") for item in actions if isinstance(item, dict)]
    result_summary = [f"{item.get('action_type')}={'ok' if item.get('success') else 'fail'}" for item in results if isinstance(item, dict)]
    return (
        f"[Trace step={loop_step}] event={event_type} "
        f"intents={intent_summary or ['<none>']} "
        f"actions={action_types or ['<none>']} "
        f"results={result_summary or ['<none>']}"
    )


def _summarize_intents_for_trace(intents: list[object]) -> list[str]:
    """把 intents 格式化为更适合验证 LLM 参与度的摘要。"""
    summary: list[str] = []
    for item in intents:
        if not isinstance(item, dict):
            continue
        label = str(item.get("type", "unknown"))
        if item.get("payload", {}).get("llm_selected"):
            label += "[llm_selected]"
        if item.get("requires_llm"):
            label += "[requires_llm]"
        reason = str(item.get("reason", "")).strip()
        if reason:
            label += f"@{reason}"
        summary.append(label)
    return summary


def _show_recent_traces(output: ConsoleOutput, loop: AgentLoop, command: str) -> None:
    """按数量输出最近的决策 trace。"""
    parts = command.split()
    limit = 5
    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
        except ValueError:
            output.show_text("[Error] /trace 的数量参数必须是整数。")
            return

    traces = loop.recent_traces[-limit:]
    if not traces:
        output.show_text("当前还没有 trace。")
        return
    for trace in traces:
        output.show_text(_format_trace(trace))


def _show_last_decision(output: ConsoleOutput, core: AgentCore) -> None:
    """输出最近一轮的 intents 和 action results。"""
    if not core.last_intents and not core.last_action_results:
        output.show_text("当前还没有最近决策记录。")
        return
    output.show_text(
        "[Last intents] "
        + json.dumps(
            [
                {
                    "type": intent.type,
                    "priority": intent.priority,
                    "reason": intent.reason,
                    "payload": intent.payload,
                    "requires_llm": intent.requires_llm,
                }
                for intent in core.last_intents
            ],
            ensure_ascii=False,
        )
    )
    output.show_text(
        "[Last results] "
        + json.dumps(
            [
                {
                    "action_type": result.action_type,
                    "success": result.success,
                    "reason": result.reason,
                    "payload": result.payload,
                }
                for result in core.last_action_results
            ],
            ensure_ascii=False,
        )
    )


def _show_scenarios(output: ConsoleOutput) -> None:
    """列出内置验证场景。"""
    output.show_text("可用场景：")
    for name, description in SCENARIO_DESCRIPTIONS.items():
        output.show_text(f"  {name}: {description}")


if __name__ == "__main__":
    main()
