"""Sherpa-ONNX 离线 TTS（板端合成，无需百度云端）。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from src.adapters.voice.audio_playback import play_wav_file, write_generated_audio_wav

DEFAULT_SHERPA_TTS_DIR = Path("models/vits-icefall-zh-aishell3")
DEFAULT_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "vits-icefall-zh-aishell3.tar.bz2"
)


def resolve_sherpa_tts_dir(model_dir: str | Path | None = None) -> Path:
    raw = model_dir or os.environ.get("EMBED_SHERPA_TTS_DIR") or DEFAULT_SHERPA_TTS_DIR
    path = Path(raw).expanduser()
    model_onnx = path / "model.onnx"
    if model_onnx.is_file():
        return path.resolve()
    raise FileNotFoundError(
        f"Sherpa TTS 模型目录无效：{path}\n"
        f"请先运行：python scripts/setup_sherpa_tts.py"
    )


def _rule_fsts(model_dir: Path) -> str:
    names = ("phone.fst", "date.fst", "number.fst")
    found = [str(model_dir / name) for name in names if (model_dir / name).is_file()]
    return ",".join(found)


def create_offline_tts(
    model_dir: Path,
    *,
    num_threads: int = 2,
    max_num_sentences: int = 2,
) -> Any:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise ImportError("请先安装 sherpa-onnx：pip install sherpa-onnx") from exc

    rule_fsts = _rule_fsts(model_dir)
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / "model.onnx"),
                lexicon=str(model_dir / "lexicon.txt"),
                tokens=str(model_dir / "tokens.txt"),
            ),
            num_threads=int(num_threads),
            provider="cpu",
        ),
        rule_fsts=rule_fsts,
        max_num_sentences=int(max_num_sentences),
    )
    if not tts_config.validate():
        raise RuntimeError(f"Sherpa TTS 配置无效：{model_dir}")
    return sherpa_onnx.OfflineTts(tts_config)


class SherpaOnnxTTSBackend:
    """Sherpa-ONNX 离线中文 TTS，接口与 BaiduTTSBackend.speak 对齐。"""

    def __init__(
        self,
        *,
        model_dir: str | Path = DEFAULT_SHERPA_TTS_DIR,
        output_path: str | Path = "data/tts_output.wav",
        speaker_id: int = 0,
        default_speed: float = 1.0,
        num_threads: int = 2,
        alsa_playback_device: str | None = None,
        prefer_capture_device: str | None = None,
        player_command: str = "aplay",
    ) -> None:
        self._model_dir = resolve_sherpa_tts_dir(model_dir)
        self.output_path = Path(output_path)
        self.speaker_id = int(speaker_id)
        self.default_speed = float(default_speed)
        self.num_threads = int(num_threads)
        self.alsa_playback_device = (alsa_playback_device or "").strip() or None
        self.prefer_capture_device = (prefer_capture_device or "").strip() or None
        self.player_command = player_command
        self.voice_id = str(speaker_id)
        self.volume = 10
        self.speed = 5
        self._tts: Any | None = None

    def preload(self) -> None:
        if self._tts is None:
            started = time.time()
            self._tts = create_offline_tts(self._model_dir, num_threads=self.num_threads)
            print(
                f"[SherpaTTS] 离线模型已加载：{self._model_dir.name} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )

    def speak(self, text: str, *, voice: str | None, volume: int | None, speed: float | None) -> None:
        del volume  # 离线 VITS 暂不支持音量参数
        if not text.strip():
            return
        self.preload()
        assert self._tts is not None

        sid = int(voice) if voice is not None and str(voice).isdigit() else self.speaker_id
        speed_value = _map_speed(speed if speed is not None else self.speed)

        started = time.time()
        audio = self._tts.generate(text.strip(), sid=sid, speed=speed_value)
        if len(audio.samples) == 0:
            raise RuntimeError("Sherpa TTS 未生成有效音频")

        write_generated_audio_wav(
            output_path=self.output_path,
            samples=audio.samples,
            sample_rate=int(audio.sample_rate),
        )
        play_wav_file(
            self.output_path,
            alsa_playback_device=self.alsa_playback_device,
            prefer_capture_device=self.prefer_capture_device,
            player_command=self.player_command,
        )
        duration = len(audio.samples) / float(audio.sample_rate)
        print(
            f"[SherpaTTS] 合成+播放 {duration:.1f}s 音频，耗时 {time.time() - started:.2f}s",
            flush=True,
        )

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = str(voice_id)
        if voice_id.isdigit():
            self.speaker_id = int(voice_id)

    def set_volume(self, volume: int) -> None:
        self.volume = max(0, min(15, int(volume)))

    def set_speed(self, speed: float) -> None:
        self.speed = max(0, min(15, int(round(float(speed)))))

    def is_configured(self) -> bool:
        try:
            resolve_sherpa_tts_dir(self._model_dir)
            return True
        except FileNotFoundError:
            return False


def _map_speed(raw: float | int) -> float:
    """把百度 0~15 语速映射到 Sherpa 0.5~2.0。"""
    numeric = float(raw)
    if numeric <= 2.0:
        return max(0.5, min(2.0, numeric))
    return max(0.5, min(2.0, 0.5 + (numeric / 15.0) * 1.5))
