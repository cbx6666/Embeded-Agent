"""TTS 后端工厂。"""

from __future__ import annotations

from typing import Any


def build_tts_backend(
    *,
    backend: str,
    output_path: str,
    alsa_playback_device: str | None,
    prefer_capture_device: str | None,
    sherpa_tts_dir: str,
    speaker_id: int = 0,
) -> Any:
    backend = backend.strip().lower()
    if backend in {"sherpa-onnx", "sherpa", "local", "offline"}:
        from src.adapters.voice.sherpa_tts_backend import SherpaOnnxTTSBackend

        return SherpaOnnxTTSBackend(
            model_dir=sherpa_tts_dir,
            output_path=output_path,
            speaker_id=speaker_id,
            alsa_playback_device=alsa_playback_device,
            prefer_capture_device=prefer_capture_device,
        )

    if backend in {"baidu", "cloud"}:
        from src.adapters.voice.baidu_tts_backend import BaiduTTSBackend

        return BaiduTTSBackend(
            output_path=output_path,
            alsa_playback_device=alsa_playback_device,
            prefer_capture_device=prefer_capture_device,
        )

    raise ValueError(f"未知 TTS backend：{backend}，可选：sherpa-onnx / baidu")
