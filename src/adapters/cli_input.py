"""命令行输入适配器模块。"""

from __future__ import annotations

import re
import time

from src.agent.event import Event

HELP_TEXT = """可用命令：
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
    /mock posture sitting
    /mock posture standing
    /mock posture leaning
    /mock posture lying
    /mock activity studying
    /mock activity working
    /mock activity resting

  系统命令：
    /state
    /history
    /profile
    /voice_once
    /voice_replay
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
    /help
    /exit
"""

FOCUS_START_PATTERNS = (
    re.compile(r"开始专注\s*(\d+)\s*分钟?"),
    re.compile(r"专注\s*(\d+)\s*分钟?"),
    re.compile(r"start focus\s*(\d+)", re.IGNORECASE),
)
FOCUS_END_KEYWORDS = ("结束专注", "停止专注", "结束focus", "stop focus")


class CLIInputAdapter:
    """负责从终端读取一行输入。"""

    def __init__(self, prompt: str = "agent> ") -> None:
        self.prompt = prompt

    def readline(self) -> str | None:
        """读取一行终端输入。"""
        try:
            return input(self.prompt)
        except EOFError:
            return None
        except KeyboardInterrupt:
            return "/exit"



def parse_cli_event(command: str, timestamp: int | None = None) -> Event | None:
    """将 CLI 文本翻译成结构化控制事件。

    只识别开始/结束专注这类结构化控制指令；不再产生 user_text_input。
    自然语言对话应通过语音链路（speech_recognized）进入。无法识别时返回 None。
    """
    ts = timestamp or int(time.time())
    text = command.strip()

    minutes = _parse_focus_start_minutes(text)
    if minutes is not None:
        return Event(
            type="focus_start_requested",
            timestamp=ts,
            payload={"duration_sec": minutes * 60, "source": "cli", "raw_text": text},
        )

    if _is_focus_stop_command(text):
        return Event(
            type="focus_stop_requested",
            timestamp=ts,
            payload={"source": "cli", "raw_text": text},
        )

    return None



def _parse_focus_start_minutes(text: str) -> int | None:
    for pattern in FOCUS_START_PATTERNS:
        match = pattern.search(text.strip())
        if match:
            return int(match.group(1))
    return None



def _is_focus_stop_command(text: str) -> bool:
    lowered = text.strip().lower()
    return any(keyword in lowered for keyword in FOCUS_END_KEYWORDS)
