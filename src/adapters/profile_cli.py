from __future__ import annotations

"""Profile 相关 CLI 命令解析。

这个模块只负责把文本命令转成 Core 的公开 API 调用，不理解 profile 内部字段，
也不直接操作 UserProfileService。这样 main.py 和 agent_lab.py 可以复用同一份
命令处理逻辑，避免后续新增 profile 命令时两边忘记同步。
"""

from typing import Protocol


class ProfileCommandCore(Protocol):
    """Profile CLI 需要的 Core 最小接口。"""

    def render_profile(self) -> str:
        """返回当前用户画像文本。"""

    def render_users(self) -> str:
        """返回用户列表文本。"""

    def switch_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
    ) -> str:
        """切换当前用户。"""

    def set_user_preference(self, key: str, value: object) -> str:
        """更新当前用户偏好。"""

    def set_user_info(self, key: str, value: object) -> str:
        """更新当前用户基础资料。"""


class TextOutput(Protocol):
    """Profile CLI 需要的输出最小接口。"""

    def show_text(self, text: str) -> None:
        """输出一段文本。"""


def handle_profile_command(core: ProfileCommandCore, output: TextOutput, command: str) -> bool:
    """处理 profile 相关命令，命中时返回 True。

    这里只做命令解析和错误展示；用户创建、偏好校验、渲染格式都委托给 Core/Service，
    避免适配器层泄漏业务规则。
    """
    if command == "/profile":
        output.show_text(core.render_profile())
        return True
    if command == "/users":
        output.show_text(core.render_users())
        return True
    if _matches_command(command, "/switch_user"):
        _handle_switch_user(core, output, command)
        return True
    if _matches_command(command, "/set_pref"):
        _handle_set_pref(core, output, command)
        return True
    if _matches_command(command, "/set_info"):
        _handle_set_info(core, output, command)
        return True
    return False


def _matches_command(command: str, name: str) -> bool:
    """避免 /switch_user_x 被误判为 /switch_user。"""
    return command == name or command.startswith(f"{name} ")


def _handle_switch_user(core: ProfileCommandCore, output: TextOutput, command: str) -> None:
    """处理 `/switch_user <user_id> [display_name]`。"""
    parts = command.split(maxsplit=2)
    if len(parts) < 2:
        output.show_text("[Error] 用法: /switch_user <user_id> [display_name]")
        return
    display_name = parts[2].strip() if len(parts) >= 3 else None
    output.show_text(core.switch_user(parts[1], display_name=display_name))


def _handle_set_pref(core: ProfileCommandCore, output: TextOutput, command: str) -> None:
    """处理 `/set_pref <key> <value>`。"""
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        output.show_text("[Error] 用法: /set_pref <key> <value>")
        return
    try:
        output.show_text(core.set_user_preference(parts[1], parts[2]))
    except ValueError as exc:
        output.show_text(f"[Error] {exc}")


def _handle_set_info(core: ProfileCommandCore, output: TextOutput, command: str) -> None:
    """处理 `/set_info <key> <value>`。"""
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        output.show_text("[Error] 用法: /set_info <key> <value>")
        return
    try:
        output.show_text(core.set_user_info(parts[1], parts[2]))
    except ValueError as exc:
        output.show_text(f"[Error] {exc}")
