"""视觉情绪/疲劳适配器的可调参数（全部留在适配器层，内核不可见）。

内核只消费最终发出的 `Event`；改阈值、帧率、窗口在此文件或构造 `VisionAffectConfig` 时完成。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionAffectConfig:
    """检测与上报节奏；与 `user_fatigue_updated` / `user_emotion_updated` 的 payload 约定见领域文档。"""

    camera_index: int = 0
    raf_checkpoint: str | None = None
    # WuJie1010 VGG19 checkpoint path (PrivateTest_model.t7)
    wujie_checkpoint: str | None = None
    # WuJie1010 VGG19 OM path (wujie_vgg19_static.om)
    wujie_om_model: str | None = None
    wujie_om_device_id: int = 0
    # 情绪推理：默认 `wujie-om`；`raf` 需 RAF-ResNet 权重；
    # `wujie-vgg19` 需 `PrivateTest_model.t7`；
    # `wujie-om` 需 `.om` 模型；`none` 仅疲劳
    emotion_backend: str = "wujie-om"
    deepface_model: str = "VGG-Face"

    # --- 疲劳：EAR 闭眼 + MAR 打哈欠/张口 + 两路滑动窗 + 融合滞回 ---
    ear_threshold: float = 0.21
    mar_yawn_threshold: float = 0.5
    fatigue_eye_weight: float = 0.75
    fatigue_mouth_weight: float = 0.25
    yawn_flag_min_ratio: float = 0.1
    perclos_window_sec: float = 10.0
    fatigue_periodic_emit_sec: float = 3.0
    # 连续闭眼超过该秒数，直接强制置为 high（减少“闭眼很久却不上高疲劳”）
    force_high_eye_closed_sec: float = 0.8
    # 眼部 PERCLOS 直接触发档位（忽略口部，提升“长闭眼”敏感度）
    force_high_eye_perclos: float = 0.55
    force_moderate_eye_perclos: float = 0.35

    # --- 采集节奏 ---
    target_fps: float = 8.0

    # --- 情绪：每 N 帧推理一次（降载）---
    emotion_every_n_frames: int = 4
    emotion_min_emit_interval_sec: float = 1.5

    # --- MediaPipe Face Mesh ---
    face_mesh_max_faces: int = 1
    face_mesh_refine_landmarks: bool = True
    face_mesh_min_detection_confidence: float = 0.5
    face_mesh_min_tracking_confidence: float = 0.5

    # --- 写入 Event.payload["source"]，便于日志区分 ---
    fatigue_event_source: str = "mediapipe_pipeline"
    emotion_event_source: str = "camera"

    # --- 状态持久化：每秒明细 + 日/周汇总（SQLite） ---
    enable_state_storage: bool = True
    state_stats_db_path: str = "data/state_stats.db"
    second_state_retention_days: int = 3
    state_cleanup_interval_sec: int = 300
