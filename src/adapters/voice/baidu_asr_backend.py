"""百度短语音识别（ASR）后端。

将音频文件送入百度短语音识别 REST API，返回识别文本。
支持从环境变量或本地 `.env` 文件读取 API 凭证。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 凭证加载
# ---------------------------------------------------------------------------

def _load_simple_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    """从 .env 文件中加载键值对（不依赖 python-dotenv）。"""
    if env_path is None:
        candidates = [
            Path("data/.env"),
            Path(".env"),
            Path(__file__).parent.parent.parent / ".env",
        ]
    else:
        candidates = [Path(env_path)]

    result: dict[str, str] = {}
    for path in candidates:
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    result[key.strip()] = value.strip()
            break
    return result


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class BaiduShortASRBackend:
    """百度短语音识别实现。"""

    ASR_URL = "http://vop.baidu.com/server_api"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        dev_pid: int = 1537,  # 1537 = 识别中文
        sample_rate: int = 16000,
        timeout_sec: float = 20.0,
        env_path: str | Path | None = None,
    ) -> None:
        env_values = _load_simple_env_file(env_path)
        self.api_key = (
            api_key
            or env_values.get("BAIDU_ASR_API_KEY")
            or os.environ.get("BAIDU_ASR_API_KEY")
            or "Sl7dJK1tRlfKQw66riYlafQs"  # hardcoded default
        )
        self.secret_key = (
            secret_key
            or env_values.get("BAIDU_ASR_SECRET_KEY")
            or os.environ.get("BAIDU_ASR_SECRET_KEY")
            or "iyEUSwjoznkQoBXCU4vFKpY2bBtpZ8Gt"  # hardcoded default
        )
        self.app_id = (
            os.environ.get("BAIDU_ASR_APP_ID")
            or "7734053"  # hardcoded default
        )
        self.dev_pid = int(dev_pid)
        self.sample_rate = int(sample_rate)
        self.timeout_sec = float(timeout_sec)

    def is_configured(self) -> bool:
        """检查是否已配置有效的 API 凭证。"""
        return bool(self.api_key and self.secret_key)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def recognize_file(self, audio_path: str | Path) -> str:
        """识别音频文件内容，返回识别的文本（空串表示未识别到）。"""
        if not self.is_configured():
            raise RuntimeError(
                "百度 ASR 未配置凭证，请设置 BAIDU_ASR_API_KEY / BAIDU_ASR_SECRET_KEY。"
            )

        token = self._fetch_access_token()
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"ASR 音频文件不存在：{audio_path}")

        wav_data = _convert_to_pcm16k(audio_path)
        speech_b64 = _base64_encode(wav_data)

        # 用 POST body 发送语音数据，避免 URL 太长导致 414
        post_body = json.dumps(
            {
                "dev_pid": self.dev_pid,
                "cuid": self.app_id,
                "token": token,
                "format": "pcm",
                "rate": self.sample_rate,
                "channel": 1,
                "len": len(wav_data),
                "speech": speech_b64,
                "speech_type": "audio/pcm",
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url=self.ASR_URL,
            data=post_body,
            method="POST",
        )
        request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"百度 ASR 请求失败 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"百度 ASR 连接失败: {exc.reason}") from exc

        err_no = payload.get("err_no")
        err_msg = payload.get("err_msg", "")
        if err_no != 0:
            raise RuntimeError(f"百度 ASR 识别失败 err_no={err_no}：{err_msg}")

        result: list[dict[str, Any]] | None = payload.get("result")
        if not result:
            return ""
        return str(result[0]).strip()

    def recognize_file_with_confidence(self, audio_path: str | Path) -> tuple[str, float]:
        """识别音频文件内容，返回 (识别文本, 置信度) 元组。"""
        text = self.recognize_file(audio_path)
        return text, 0.9  # 百度短语音 API 不返回置信度，用固定值

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _fetch_access_token(self) -> str:
        """向百度 Access Token 端点获取凭证。"""
        token_url = "https://aip.baidubce.com/oauth/2.0/token"
        query = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            }
        )
        request = urllib.request.Request(url=f"{token_url}?{query}", method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"获取百度 Access Token 失败 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"获取百度 Access Token 连接失败: {exc.reason}") from exc

        marker = '"access_token":"'
        start = payload.find(marker)
        if start == -1:
            raise RuntimeError(f"Token 响应缺少 access_token: {payload}")
        start += len(marker)
        end = payload.find('"', start)
        if end == -1:
            raise RuntimeError(f"Token 响应格式异常: {payload}")
        return payload[start:end]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _convert_to_pcm16k(audio_path: Path) -> bytes:
    """将任意音频文件转换为 16kHz PCM 原始字节。

    优先级：
    1. ffmpeg（跨平台，需要安装）
    2. Windows 原生 WAV（直接读取 16-bit 16kHz 单声道 WAV）
    3. avconv（macOS/Linux）
    """
    import subprocess

    audio_path = Path(audio_path)
    pcm_path = audio_path.with_suffix(".pcm")

    # 尝试 ffmpeg（最通用）
    for player in ("ffmpeg", "ffmpeg.exe"):
        try:
            subprocess.run(
                [
                    player,
                    "-y",
                    "-loglevel", "quiet",
                    "-i", str(audio_path),
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "s16le",
                    str(pcm_path),
                ],
                check=True,
                capture_output=True,
            )
            data = pcm_path.read_bytes()
            pcm_path.unlink(missing_ok=True)
            return data
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # 如果已是 PCM/RAW 格式，直接读取
    if audio_path.suffix.lower() in {".pcm", ".raw"}:
        return audio_path.read_bytes()

    # 尝试读取 WAV（Python 标准库 wave，支持 Windows/macOS/Linux）
    try:
        import wave as wave_module
        with wave_module.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)

            # 如果是 16-bit 16kHz 单声道，直接返回
            if sample_width == 2 and frame_rate == 16000 and n_channels == 1:
                return data

            # 其他格式尝试转换
            pcm_path = audio_path.with_suffix(".pcm")
            # 用 wave 重写为标准格式
            with wave_module.open(str(pcm_path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                if sample_width != 2:
                    # 8-bit 转 16-bit
                    import struct
                    frames_8bit = data
                    data_16 = b"".join(
                        struct.pack("<h", (b - 128) * 256) for b in frames_8bit
                    )
                    out.writeframes(data_16)
                elif frame_rate != 16000:
                    # 重采样：用线性插值简化
                    import struct
                    frames = struct.unpack(f"<{len(data)//sample_width}h", data)
                    ratio = 16000 / frame_rate
                    new_len = int(len(frames) * ratio)
                    data_16 = struct.pack(f"<{new_len}h", *[int(frames[int(i/ratio)]) for i in range(new_len)])
                    out.writeframes(data_16)
                elif n_channels != 1:
                    # 立体声转单声道
                    import struct
                    frames = struct.unpack(f"<{len(data)//sample_width}h", data)
                    data_16 = b"".join(
                        struct.pack("<h", frames[i * n_channels])
                        for i in range(len(frames) // n_channels)
                    )
                    out.writeframes(data_16)
                else:
                    out.writeframes(data)
            data = pcm_path.read_bytes()
            pcm_path.unlink(missing_ok=True)
            return data
    except Exception:
        pass

    raise RuntimeError(
        f"无法将 {audio_path} 转换为 16kHz PCM，请安装 ffmpeg（https://ffmpeg.org）\n"
        "或提供已转换为 16kHz 16-bit 单声道 WAV 格式的音频文件。"
    )


# ---------------------------------------------------------------------------
# 录音工具函数
# ---------------------------------------------------------------------------

def record_audio_wav(
    output_path: str | Path,
    duration_sec: int = 10,
    sample_rate: int = 16000,
    device: int | str | None = None,
    *,
    alsa_device: str | None = None,
    prepare_device: bool = True,
    debug: Any | None = None,
) -> Path | None:
    """跨平台录音，保存为 WAV 文件。

    优先级：sounddevice > pyaudio > arecord（Linux）。
    失败时会打印具体错误原因。
    """
    import platform
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 释放麦克风设备（可能被 wake-word detector 的 arecord 占用）----
    if platform.system() == "Linux":
        resolved = alsa_device or (device if isinstance(device, str) else None) or "plughw:0,0"
        if prepare_device:
            _release_alsa_device(resolved, settle_ms=80)
        # 板载 ALSA 设备优先 arecord，避免 sounddevice 抢到错误默认麦
        if alsa_device or isinstance(device, str):
            recorded = _record_arecord_wav(
                output_path,
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                alsa_device=alsa_device
                or (device if isinstance(device, str) else None)
                or "plughw:0,0",
                debug=debug,
            )
            if recorded is not None:
                return recorded

    # ---- 1. sounddevice（跨平台） ----
    try:
        import sounddevice as _sd
        audio = _sd.rec(
            frames=duration_sec * sample_rate,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device,
        )
        _sd.wait()
        import wave as _wave
        with _wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return output_path
    except Exception as exc:
        pass  # sounddevice 未安装，静默降级

    # ---- 2. pyaudio（跨平台） ----
    try:
        import pyaudio as _pa
        import wave as _wave
        CHUNK = 1024
        p = _pa.PyAudio()
        stream = p.open(
            format=_pa.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device,
            frames_per_buffer=CHUNK,
        )
        frames = []
        for _ in range(int(sample_rate / CHUNK * duration_sec)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        with _wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return output_path
    except Exception as exc:
        pass  # pyaudio 未安装，静默降级

    # ---- 3. arecord（Linux） ----
    if platform.system() == "Linux":
        if alsa_device:
            alsa_dev = alsa_device
        elif isinstance(device, str) and device.strip():
            alsa_dev = device.strip()
        elif device is not None:
            alsa_dev = f"plughw:{device},0"
        else:
            alsa_dev = "plughw:0,0"
        recorded = _record_arecord_wav(
            output_path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            alsa_device=alsa_dev,
        )
        if recorded is not None:
            return recorded

    print("[record_audio] ⚠️ 所有录音后端均不可用。", flush=True)
    return None


def _record_arecord_wav(
    output_path: Path,
    *,
    duration_sec: int,
    sample_rate: int,
    alsa_device: str,
    debug: Any | None = None,
) -> Path | None:
    import subprocess as _sp
    import time

    from src.adapters.voice.alsa_audio_devices import (
        capture_device_node_exists,
        resolve_capture_device,
        wait_for_capture_device,
    )

    def _log(level: str, event: str, **fields: Any) -> None:
        if debug is None:
            return
        fn = getattr(debug, level, None)
        if callable(fn):
            fn(event, **fields)

    devices_to_try: list[str] = []
    primary = (alsa_device or "").strip()
    if primary:
        devices_to_try.append(primary)
    fallback = resolve_capture_device(explicit=primary or None)
    if fallback and fallback not in devices_to_try:
        devices_to_try.append(fallback)

    last_stderr = ""
    for device in devices_to_try:
        for attempt in range(1, 4):
            if not capture_device_node_exists(device):
                _log(
                    "warn",
                    "capture_node_missing",
                    device=device,
                    attempt=attempt,
                )
                wait_for_capture_device(device, timeout_sec=0.8)
            _log(
                "step",
                "arecord_start",
                device=device,
                duration_sec=duration_sec,
                output=str(output_path),
                attempt=attempt,
                important=True,
            )
            try:
                result = _sp.run(
                    [
                        "arecord",
                        "-D",
                        device,
                        "-f",
                        "S16_LE",
                        "-r",
                        str(sample_rate),
                        "-c",
                        "1",
                        "-d",
                        str(duration_sec),
                        str(output_path),
                    ],
                    capture_output=True,
                    timeout=duration_sec + 5,
                )
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                last_stderr = stderr
                if result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 1000:
                    _log(
                        "info",
                        "arecord_ok",
                        device=device,
                        attempt=attempt,
                        bytes=output_path.stat().st_size,
                        stderr=stderr or "(none)",
                    )
                    return output_path
                _log(
                    "error",
                    "arecord_failed",
                    device=device,
                    attempt=attempt,
                    returncode=result.returncode,
                    stderr=stderr,
                )
                print(
                    f"[record_audio] arecord 失败 (device={device}, attempt={attempt}, "
                    f"code={result.returncode}): {stderr}",
                    flush=True,
                )
            except _sp.TimeoutExpired:
                _log("error", "arecord_timeout", device=device, attempt=attempt)
                print("[record_audio] arecord 超时", flush=True)
            except FileNotFoundError:
                _log("error", "arecord_not_found")
                print("[record_audio] arecord 命令未找到，请安装 alsa-utils", flush=True)
                return None
            except Exception as exc:
                _log("error", "arecord_exception", device=device, message=str(exc))
                print(f"[record_audio] arecord 异常: {exc}", flush=True)

            if attempt < 3:
                time.sleep(0.15 * attempt)
                wait_for_capture_device(device, timeout_sec=0.5 * attempt)

    if last_stderr:
        _log("error", "arecord_exhausted", stderr=last_stderr, devices=devices_to_try)
    return None


def _release_alsa_device(alsa_device: str | None = None, *, settle_ms: int = 150) -> None:
    """释放目标麦克风（kill 占用该设备的 arecord/parecord），避免与唤醒检测抢麦。"""
    release_capture_device(alsa_device, settle_ms=settle_ms)


def release_capture_device(
    alsa_device: str | None = None,
    *,
    settle_ms: int = 150,
    use_fuser: bool = False,
    allow_global_pkill: bool = True,
) -> None:
    """释放指定 ALSA 录音设备，并短暂等待驱动就绪。

    默认不用 fuser -k（易把 USB 麦节点打没）。
    allow_global_pkill=False 时跳过 pkill（保护常驻 arecord）。
    """
    import subprocess as _sp
    import time

    if allow_global_pkill:
        for cmd in ("arecord", "parecord"):
            try:
                _sp.run(["pkill", "-15", cmd], capture_output=True, timeout=2)
            except Exception:
                pass
        time.sleep(0.08)
        for cmd in ("arecord", "parecord"):
            try:
                _sp.run(["pkill", "-9", cmd], capture_output=True, timeout=2)
            except Exception:
                pass

    if use_fuser:
        from src.adapters.voice.alsa_audio_devices import parse_alsa_device

        parsed = parse_alsa_device(alsa_device or "")
        if parsed is not None:
            card, dev = parsed
            pcm_capture = f"/dev/snd/pcmC{card}D{dev}c"
            try:
                _sp.run(["fuser", "-k", pcm_capture], capture_output=True, timeout=2)
            except Exception:
                pass

    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)

    if alsa_device:
        from src.adapters.voice.alsa_audio_devices import wait_for_capture_device

        wait_for_capture_device(alsa_device, timeout_sec=min(2.0, settle_ms / 500 + 0.5))


def _base64_encode(data: bytes) -> str:
    """对 bytes 进行 base64 编码（标准库实现）。"""
    import base64
    return base64.b64encode(data).decode("ascii")
