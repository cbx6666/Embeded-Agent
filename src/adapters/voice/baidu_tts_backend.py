from __future__ import annotations

"""百度文本转语音 backend。

当前目标：
- 消费 Agent 的 `speak` / `set_tts_*` 动作；
- 调用百度 TTS REST 接口生成音频；
- 在板端用 `aplay` 播放，形成最小语音输出闭环。
"""

import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.adapters.voice.baidu_asr_backend import _load_simple_env_file


class BaiduTTSBackend:
    """百度语音合成实现。"""

    TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
    TTS_URL = "https://tsn.baidu.com/text2audio"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        cuid: str | None = None,
        output_path: str | Path = "data/tts_output.wav",
        default_voice: str = "0",
        default_volume: int = 10,
        default_speed: int = 5,
        default_pitch: int = 5,
        timeout_sec: float = 20.0,
        player_command: str = "aplay",
        env_path: str | Path | None = None,
    ) -> None:
        env_values = _load_simple_env_file(env_path)
        self.api_key = (
            api_key
            or env_values.get("BAIDU_TTS_API_KEY")
            or env_values.get("BAIDU_ASR_API_KEY")
            or os.environ.get("BAIDU_TTS_API_KEY")
            or os.environ.get("BAIDU_ASR_API_KEY")
            or "Sl7dJK1tRlfKQw66riYlafQs"  # hardcoded default
        )
        self.secret_key = (
            secret_key
            or env_values.get("BAIDU_TTS_SECRET_KEY")
            or env_values.get("BAIDU_ASR_SECRET_KEY")
            or os.environ.get("BAIDU_TTS_SECRET_KEY")
            or os.environ.get("BAIDU_ASR_SECRET_KEY")
            or "iyEUSwjoznkQoBXCU4vFKpY2bBtpZ8Gt"  # hardcoded default
        )
        self.cuid = (
            cuid
            or env_values.get("BAIDU_TTS_CUID")
            or os.environ.get("BAIDU_TTS_CUID")
            or "EmbededAgent_Windows_PC"
        )
        self.output_path = Path(output_path)
        self.voice_id = str(default_voice)
        self.volume = int(default_volume)
        self.speed = int(default_speed)
        self.pitch = int(default_pitch)
        self.timeout_sec = float(timeout_sec)
        # player_command="auto" 时根据平台选择默认值，Windows 不依赖外部 aplay
        self.player_command = player_command

    def speak(self, text: str, *, voice: str | None, volume: int | None, speed: float | None) -> None:
        if not self.is_configured():
            raise RuntimeError("百度 TTS 未配置，请先设置 BAIDU_TTS_API_KEY / BAIDU_TTS_SECRET_KEY。")

        final_voice = str(voice or self.voice_id)
        final_volume = _normalize_volume(volume if volume is not None else self.volume)
        final_speed = _normalize_speed(speed if speed is not None else self.speed)

        token = self._fetch_access_token()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        query = urllib.parse.urlencode(
            {
                "tex": text,
                "tok": token,
                "cuid": self.cuid,
                "ctp": 1,
                "lan": "zh",
                "spd": final_speed,
                "pit": self.pitch,
                "vol": final_volume,
                "per": final_voice,
                "aue": 6,
            }
        )
        request = urllib.request.Request(url=f"{self.TTS_URL}?{query}", method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                content_type = str(response.headers.get("Content-Type", ""))
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"百度 TTS 请求失败 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"百度 TTS 连接失败: {exc.reason}") from exc

        if "audio" not in content_type:
            detail = body.decode("utf-8", errors="ignore")
            raise RuntimeError(f"百度 TTS 未返回音频: {detail}")

        self.output_path.write_bytes(body)
        self._play_audio()

    def _play_audio(self) -> None:
        """跨平台播放音频文件。

        播放前先用 amixer（Linux）将 PCM 音量调到最大，解决板级设备音量偏小的问题。
        优先级：aplay（Linux）> sounddevice > winsound（Windows）> afplay（macOS）。
        """
        import platform
        system = platform.system()
        player_cmd = self.player_command

        # ---- 1. sounddevice（跨平台） ----
        try:
            import sounddevice as sd
            import wave
            with wave.open(str(self.output_path), "rb") as wf:
                rate = wf.getframerate()
                n_frames = wf.getnframes()
                data = wf.readframes(n_frames)
            import numpy as np
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, samplerate=rate)
            sd.wait()
            return
        except Exception:
            pass

        # ---- 2. Windows winsound ----
        if system == "Windows":
            try:
                import winsound
                winsound.PlaySound(str(self.output_path), winsound.SND_FILENAME)
                return
            except Exception:
                pass

        # ---- 3. macOS afplay ----
        if system == "Darwin":
            try:
                subprocess.run(["afplay", str(self.output_path)], check=True, capture_output=True)
                return
            except Exception:
                pass

        # ---- 4. Linux / 通用 aplay ----
        if player_cmd and player_cmd != "auto":
            subprocess.run([player_cmd, str(self.output_path)], check=True, capture_output=True)
        else:
            for cmd in ("aplay", "paplay", "ffplay"):
                try:
                    subprocess.run([cmd, str(self.output_path)], check=True, capture_output=True)
                    return
                except Exception:
                    pass
            raise RuntimeError(
                f"无法播放音频文件 {self.output_path}，请安装音频播放器 "
                "（Linux: alsa-utils/paprefs, Windows: 已内置 winsound, macOS: afplay）"
            )

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = str(voice_id)

    def set_volume(self, volume: int) -> None:
        self.volume = _normalize_volume(volume)

    def set_speed(self, speed: float) -> None:
        self.speed = _normalize_speed(speed)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def _fetch_access_token(self) -> str:
        query = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            }
        )
        request = urllib.request.Request(url=f"{self.TOKEN_URL}?{query}", method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"百度 TTS 获取 token 失败 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"百度 TTS 获取 token 连接失败: {exc.reason}") from exc

        marker = '"access_token":"'
        start = payload.find(marker)
        if start == -1:
            raise RuntimeError(f"百度 TTS token 响应缺少 access_token: {payload}")
        start += len(marker)
        end = payload.find('"', start)
        if end == -1:
            raise RuntimeError(f"百度 TTS token 响应格式异常: {payload}")
        return payload[start:end]


def _normalize_volume(volume: int) -> int:
    return max(0, min(15, int(volume)))


def _normalize_speed(speed: float | int) -> int:
    numeric = int(round(float(speed)))
    return max(0, min(15, numeric))
