from __future__ import annotations

"""基于 YOLO 的姿势和行为检测适配器模块。

该模块负责从摄像头读取视频流，使用 YOLO 模型检测用户姿势，
并将检测结果转换为标准事件发送给 Agent 核心。
"""

import time
from dataclasses import dataclass
from threading import Thread
from typing import Callable, Optional

from src.agent.event import Event


@dataclass
class DetectionResult:
    """姿势检测结果。"""
    posture: str  # sitting, standing, leaning, lying
    activity: str  # studying, working, resting
    confidence: float
    timestamp: int


class YOLOPoseDetector:
    """基于 YOLO 的姿势和行为检测器。
    
    该类封装了 YOLO 模型的加载和推理过程，
    并提供简单的接口来获取用户姿势和行为状态。
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> None:
        """初始化检测器。
        
        Args:
            model_path: YOLO 模型文件路径
            device: 推理设备 (cpu/cuda/npu)
            confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model_loaded = False
        self._model = None

    def load_model(self) -> bool:
        """加载 YOLO 模型。
        
        Returns:
            加载是否成功
        """
        try:
            # 这里是模型加载的占位符
            # 在实际部署时会使用 ultralytics YOLO 库
            print(f"[YOLOPoseDetector] 正在加载模型: {self.model_path}")
            print(f"[YOLOPoseDetector] 使用设备: {self.device}")
            
            # 模拟模型加载
            self._model_loaded = True
            return True
        except Exception as e:
            print(f"[YOLOPoseDetector] 模型加载失败: {e}")
            return False

    def detect(self) -> Optional[DetectionResult]:
        """进行一次姿势检测。
        
        Returns:
            检测结果，如果未检测到有效结果则返回 None
        """
        if not self._model_loaded:
            print("[YOLOPoseDetector] 模型未加载，无法进行检测")
            return None

        # 这里是实际推理的占位符
        # 在实际部署时会调用 YOLO 模型进行推理
        # 现在返回模拟的检测结果
        return DetectionResult(
            posture="sitting",
            activity="studying",
            confidence=0.85,
            timestamp=int(time.time()),
        )


class PoseDetectionAdapter:
    """姿势检测适配器，负责将检测结果转换为标准事件。"""

    def __init__(
        self,
        detector: YOLOPoseDetector,
        event_callback: Callable[[Event], None],
        detection_interval: float = 5.0,  # 检测间隔（秒）
    ) -> None:
        """初始化适配器。
        
        Args:
            detector: YOLO 姿势检测器
            event_callback: 事件回调函数，用于发送检测结果
            detection_interval: 检测间隔（秒）
        """
        self.detector = detector
        self.event_callback = event_callback
        self.detection_interval = detection_interval
        self._running = False
        self._thread: Optional[Thread] = None
        self._last_posture: Optional[str] = None
        self._last_activity: Optional[str] = None

    def start(self) -> None:
        """启动检测线程。"""
        if self._running:
            return

        if not self.detector.load_model():
            print("[PoseDetectionAdapter] 启动失败：模型加载失败")
            return

        self._running = True
        self._thread = Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        print("[PoseDetectionAdapter] 检测已启动")

    def stop(self) -> None:
        """停止检测线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        print("[PoseDetectionAdapter] 检测已停止")

    def _detection_loop(self) -> None:
        """检测循环，定期执行检测并发送事件。"""
        while self._running:
            try:
                result = self.detector.detect()
                if result is not None:
                    self._process_result(result)
            except Exception as e:
                print(f"[PoseDetectionAdapter] 检测出错: {e}")

            time.sleep(self.detection_interval)

    def _process_result(self, result: DetectionResult) -> None:
        """处理检测结果并发送相应事件。
        
        Args:
            result: 检测结果
        """
        ts = result.timestamp

        # 姿势变化事件
        if result.posture != self._last_posture:
            self._last_posture = result.posture
            event = Event(
                type="user_posture_updated",
                timestamp=ts,
                payload={
                    "posture": result.posture,
                    "confidence": result.confidence,
                    "source": "yolo",
                },
            )
            self.event_callback(event)

        # 活动变化事件
        if result.activity != self._last_activity:
            self._last_activity = result.activity
            event = Event(
                type="user_activity_updated",
                timestamp=ts,
                payload={
                    "activity": result.activity,
                    "confidence": result.confidence,
                    "source": "yolo",
                },
            )
            self.event_callback(event)
