"""姿势 Event 已并入 behavior 模块（yolo26n-pose.om 同帧推断），勿再单独启 pose 线程。"""

from src.adapters.behavior.pose_inference import infer_posture_and_activity

__all__ = ["infer_posture_and_activity"]
