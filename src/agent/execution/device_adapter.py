"""
设备执行边界模块。

本模块位于 ActionRealizer 之后，负责把注册 Action 交给 TimerService 或输出
适配器执行。上游输入是 Action，下游输出是 ActionResult。

本模块不理解用户语义、不调用 LLM、不生成新的 intent；它只执行已经通过
validator、guard 和 action realizer 的动作，并把异常转换为失败结果。
"""

from __future__ import annotations

from collections.abc import Callable

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.runtime.action_result import ActionResult
from src.services.timer_service import TimerService


class DeviceAdapter:
    """确定性设备适配器。

    输入标准 Action，输出 ActionResult。计时器动作由 TimerService 执行，其余
    输出动作交给 ConsoleOutput 或真实设备适配器。
    """

    def __init__(
        self,
        *,
        output: ConsoleOutput,
        timer_service: TimerService,
        timer_callback: Callable[[int], None],
    ) -> None:
        self.output = output
        self.timer_service = timer_service
        self.timer_callback = timer_callback

    def execute(self, action: Action, timestamp: int) -> ActionResult:
        """执行单个动作并捕获设备层异常。

        设备失败不应中断 Agent 主循环，而是作为 ActionResult 反馈给 runtime。
        """

        try:
            if action.type == "start_timer":
                duration_sec = int(action.payload.get("duration_sec", 0))
                self.timer_service.start(duration_sec, self.timer_callback)
            elif action.type == "stop_timer":
                self.timer_service.stop()
            else:
                self.output.execute(action)
            return ActionResult(
                action_type=action.type,
                success=True,
                timestamp=timestamp,
                payload=dict(action.payload),
            )
        except Exception as exc:
            return ActionResult(
                action_type=action.type,
                success=False,
                timestamp=timestamp,
                reason=str(exc),
                payload=dict(action.payload),
            )
