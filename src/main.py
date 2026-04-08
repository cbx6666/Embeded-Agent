from __future__ import annotations

from src.adapters.cli_input import CLIInputAdapter, HELP_TEXT, parse_cli_event
from src.adapters.console_output import ConsoleOutput
from src.adapters.mock_input import parse_mock_command
from src.agent.core import build_default_core



def main() -> None:
    output = ConsoleOutput()
    cli = CLIInputAdapter()
    core = build_default_core(output=output)

    output.show_text("Agent MVP 已启动，输入 /help 查看可用命令。")
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
                output.show_text(HELP_TEXT.rstrip())
                continue
            if command == "/state":
                output.show_text(core.render_state())
                continue
            if command == "/history":
                output.show_text(core.render_history())
                continue

            try:
                mock_event = parse_mock_command(command)
            except ValueError as exc:
                output.show_text(f"[Error] {exc}")
                continue

            if mock_event is not None:
                core.handle_event(mock_event)
                continue

            core.handle_event(parse_cli_event(command))
    finally:
        core.shutdown()
        output.show_text("Agent MVP 已退出。")


if __name__ == "__main__":
    main()
