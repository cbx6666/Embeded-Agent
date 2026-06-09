from __future__ import annotations

"""设备执行边界。

把五个注册动作交给 TimerService 或输出适配器执行：

- start_timer / stop_timer → TimerService
- speak / display / set_tts_volume → 输出适配器（ConsoleOutput / 真实设备）

对未知动作必须返回 ``success=False, reason="unsupported_action"``，不允许假成功。
设备异常被转换为失败结果，不中断主循环。
"""

from collections.abc import Callable

from src.agent.action.action_model import Action
from src.agent.action.types import ACTION_TYPE_SET
from src.agent.core.models import ActionResult

_TIMER_ACTIONS = {"start_timer", "stop_timer"}
_OUTPUT_ACTIONS = {"speak", "display", "set_tts_volume"}
_MEDIA_ACTIONS = {"play_media", "stop_media", "pause_media", "resume_media", "next_media"}


class DeviceAdapter:
    def __init__(
        self,
        *,
        output: object,
        timer_service: object,
        timer_callback: Callable[[int], None],
        media_controller: object | None = None,
        voice_runtime: object | None = None,
    ) -> None:
        self.output = output
        self.timer_service = timer_service
        self.timer_callback = timer_callback
        self.media_controller = media_controller
        self.voice_runtime = voice_runtime

    def execute(self, action: Action, timestamp: int) -> ActionResult:
        if action.type not in ACTION_TYPE_SET:
            return ActionResult(
                action_type=str(action.type),
                success=False,
                timestamp=timestamp,
                reason="unsupported_action",
                payload=dict(action.payload),
            )

        try:
            if action.type == "start_timer":
                duration_sec = int(action.payload.get("duration_sec", 0))
                self.timer_service.start(duration_sec, self.timer_callback)
            elif action.type == "stop_timer":
                self.timer_service.stop()
            elif action.type in _OUTPUT_ACTIONS:
                self.output.execute(action)
            elif action.type in _MEDIA_ACTIONS:
                self._execute_media(action)
                if action.type == "play_media":
                    self._log_media_to_console(action)
            else:  # pragma: no cover - 注册集合与处理分支保持同步
                return ActionResult(
                    action_type=str(action.type),
                    success=False,
                    timestamp=timestamp,
                    reason="unsupported_action",
                    payload=dict(action.payload),
                )
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

    def _execute_media(self, action: Action) -> None:
        """媒体动作统一经 media_controller 执行，禁止直接操作播放器。"""
        mc = self.media_controller
        if mc is None:
            raise RuntimeError("media_controller not configured")

        from src.agent.media.media_models import MediaRequest, MediaTrack

        payload = dict(action.payload)
        if action.type == "play_media":
            if payload.get("defer_after_speak"):
                runtime = self.voice_runtime
                schedule = getattr(runtime, "schedule_deferred_play_media", None)
                if callable(schedule):
                    schedule(action)
                    return
            track = MediaTrack(
                id=str(payload.get("track_id", "")),
                title=str(payload.get("title", "")),
                path=str(payload.get("path", "")),
                media_type=str(payload.get("media_type", "unknown")),
                category=str(payload.get("category", "default")),
            )
            from src.agent.media.media_models import MediaSource

            source_raw = str(payload.get("source", "user_explicit"))
            source = (
                MediaSource.AGENT_SUGGESTION
                if source_raw == MediaSource.AGENT_SUGGESTION.value
                else MediaSource.USER_EXPLICIT
            )
            mc.play_track(track, source=source)
            try:
                from src.adapters.voice.runtime.logger import voice_log

                backend = getattr(getattr(mc, "_player", None), "_backend", None)
                alsa = getattr(backend, "_resolved_alsa_device", lambda: None)()
                voice_log(f"媒体开始播放：{track.title}｜设备={alsa or 'auto'}｜{track.path}")
            except Exception:
                pass
            return

        if action.type == "stop_media":
            reason = str(payload.get("reason", "user"))
            if reason == "wake_word":
                mc.stop_for_wake_word()
            else:
                mc.stop_by_user()
            return

        if action.type == "pause_media":
            mc.pause()
            return

        if action.type == "resume_media":
            mc.resume()
            return

        if action.type == "next_media":
            mc.next_track()
            return

    def _log_media_to_console(self, action: Action) -> None:
        """把媒体播放信息打到控制台（play_media 不经 MultiOutput 语音链路）。"""
        target = getattr(self.output, "console", self.output)
        execute = getattr(target, "execute", None)
        if callable(execute):
            try:
                execute(action)
            except Exception:
                pass
