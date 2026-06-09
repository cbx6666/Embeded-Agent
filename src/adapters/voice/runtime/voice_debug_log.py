from __future__ import annotations

"""语音链路调试日志：仅保留最近一次唤醒的 voice.log / pipeline_config.json。"""

import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_VOICE_DEBUG_DIR = Path("data/voice_debug")
LATEST_SESSION_DIRNAME = "latest"


def cleanup_legacy_voice_debug(root_dir: str | Path) -> None:
    """删除旧版 wake_* 子目录及根目录 latest.log，只保留 latest/。"""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("wake_"):
            shutil.rmtree(child, ignore_errors=True)
    legacy_latest = root / "latest.log"
    if legacy_latest.is_file():
        try:
            legacy_latest.unlink()
        except OSError:
            pass


class VoiceSessionDebugLog:
    """单次唤醒/录音会话的调试日志（覆盖写入 data/voice_debug/latest/）。"""

    def __init__(
        self,
        *,
        root_dir: Path,
        session_id: str,
        enabled: bool = True,
        verbose: bool = False,
        console: bool = False,
    ) -> None:
        self.session_id = session_id
        self.enabled = bool(enabled)
        self.verbose = bool(verbose)
        self.console = bool(console)
        self._root_dir = root_dir.resolve()
        self.dir = (self._root_dir / LATEST_SESSION_DIRNAME).resolve()
        self.log_path = self.dir / "voice.log"
        if self.enabled:
            cleanup_legacy_voice_debug(self._root_dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
            self._write("INFO", "session_start", session_id=session_id, dir=str(self.dir))

    def _write(self, level: str, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        payload = " ".join(f"{key}={value!r}" for key, value in fields.items())
        line = f"[{ts}] [{level}] {event}" + (f" {payload}" if payload else "")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.console or level in {"INFO", "WARN", "ERROR"} or event in {
            "asr_result",
            "wake_hit",
            "persistent_vad_ok",
            "persistent_vad_failed",
            "vad_capture_ok",
            "vad_capture_failed",
            "session_end",
        }:
            print(f"[VoiceDebug] {line}", flush=True)

    def info(self, event: str, **fields: Any) -> None:
        self._write("INFO", event, **fields)

    def step(self, event: str, **fields: Any) -> None:
        if self.verbose or fields.get("important"):
            fields.pop("important", None)
            self._write("STEP", event, **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self._write("WARN", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._write("ERROR", event, **fields)

    def exception(self, event: str, exc: BaseException, **fields: Any) -> None:
        fields["exc_type"] = type(exc).__name__
        fields["exc_msg"] = str(exc)
        fields["traceback"] = traceback.format_exc()
        self._write("ERROR", event, **fields)

    def save_json(self, name: str, data: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        path = self.dir / name
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.info("artifact_saved", path=str(path))
            return path
        except OSError as exc:
            self.error("artifact_save_failed", name=name, error=str(exc))
            return None

    def save_bytes(self, name: str, data: bytes) -> Path | None:
        if not self.enabled:
            return None
        path = self.dir / name
        try:
            path.write_bytes(data)
            self.info("artifact_saved", path=str(path), bytes=len(data))
            return path
        except OSError as exc:
            self.error("artifact_save_failed", name=name, error=str(exc))
            return None

    def finish(self, *, status: str, **fields: Any) -> None:
        self.info("session_end", status=status, **fields)


class VoiceDebugLogManager:
    """进程内语音调试日志管理。"""

    def __init__(
        self,
        *,
        log_dir: str | Path = DEFAULT_VOICE_DEBUG_DIR,
        enabled: bool = True,
        verbose: bool = False,
        console: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.enabled = bool(enabled)
        self.verbose = bool(verbose)
        self.console = bool(console)
        env_dir = os.environ.get("EMBED_VOICE_DEBUG_DIR", "").strip()
        if env_dir:
            self.log_dir = Path(env_dir).expanduser()
        env_on = os.environ.get("EMBED_VOICE_DEBUG", "").strip().lower()
        if env_on in {"1", "true", "yes", "on"}:
            self.enabled = True
        if os.environ.get("EMBED_VOICE_DEBUG_VERBOSE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self.verbose = True
        if self.enabled:
            cleanup_legacy_voice_debug(self.log_dir)

    def open_session(self, session_id: str) -> VoiceSessionDebugLog:
        return VoiceSessionDebugLog(
            root_dir=self.log_dir,
            session_id=session_id,
            enabled=self.enabled,
            verbose=self.verbose,
            console=self.console,
        )

    def log_line(self, message: str, *, event: str = "runtime") -> None:
        """写入 latest/voice.log（无唤醒会话时的运行时里程碑日志）。"""
        if not self.enabled:
            return
        log_path = (self.log_dir / LATEST_SESSION_DIRNAME / "voice.log").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{ts}] [INFO] {event} message={message!r}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.console:
            print(f"[VoiceDebug] {line}", flush=True)


_manager: VoiceDebugLogManager | None = None


def configure_voice_debug(
    *,
    log_dir: str | Path = DEFAULT_VOICE_DEBUG_DIR,
    enabled: bool = True,
    verbose: bool = False,
    console: bool = False,
) -> VoiceDebugLogManager:
    global _manager
    _manager = VoiceDebugLogManager(
        log_dir=log_dir,
        enabled=enabled,
        verbose=verbose,
        console=console,
    )
    return _manager


def get_voice_debug_manager() -> VoiceDebugLogManager:
    global _manager
    if _manager is None:
        _manager = VoiceDebugLogManager(
            enabled=True,
            verbose=False,
        )
    return _manager
