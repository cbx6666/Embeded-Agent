"""根据 `VisionAffectConfig` 选择情绪推理后端。"""

from __future__ import annotations

from src.adapters.vision_affect.backends.deepface_emotion import DeepFaceEmotionBackend
from src.adapters.vision_affect.backends.noop_emotion import NoEmotionBackend
from src.adapters.vision_affect.backends.protocols import EmotionInferenceBackend
from src.adapters.vision_affect.backends.wujie_om import WuJieOmBackend
from src.adapters.vision_affect.backends.raf_resnet import RafEmotionBackend
from src.adapters.vision_affect.backends.wujie_vgg19 import WuJieVGG19Backend
from src.adapters.vision_affect.config import VisionAffectConfig


def build_emotion_backend(cfg: VisionAffectConfig) -> EmotionInferenceBackend:
    be = (cfg.emotion_backend or "wujie-om").strip().lower()
    if be in {"none", "off", "disabled"}:
        return NoEmotionBackend()
    if be == "raf" or be == "raf-db":
        return RafEmotionBackend(cfg.raf_checkpoint)
    if be in {"wujie-om", "om", "wujie_om"}:
        return WuJieOmBackend(cfg.wujie_om_model, device_id=cfg.wujie_om_device_id)
    if be in {"wujie-vgg19", "wujie", "fer-vgg19"}:
        return WuJieVGG19Backend(cfg.wujie_checkpoint)
    if be == "deepface":
        return DeepFaceEmotionBackend(deepface_model=cfg.deepface_model)
    raise ValueError(
        f"未知 emotion_backend={cfg.emotion_backend!r}，请使用: raf, wujie-om, wujie-vgg19, none, deepface"
    )
