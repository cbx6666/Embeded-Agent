"""视觉管线共用：ACL OM 运行时、YOLO letterbox、Ultralytics 后处理。"""

from src.adapters.vision_common.acl_runtime import AscendOmSession, import_acl
from src.adapters.vision_common.preprocess import letterbox_bgr_for_yolo, resize_gray_face_patch
from src.adapters.vision_common.yolo_ultralytics_ops import (
    decode_yolo_detect_output,
    decode_yolo_pose_output,
)

__all__ = [
    "AscendOmSession",
    "import_acl",
    "letterbox_bgr_for_yolo",
    "resize_gray_face_patch",
    "decode_yolo_detect_output",
    "decode_yolo_pose_output",
]
