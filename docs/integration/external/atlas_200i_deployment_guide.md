# Atlas 200I DK A2 部署指南

## 概述

本文档介绍如何在华为 Atlas 200I DK A2 开发板上部署并运行带有 YOLO 姿势检测功能的桌宠 Agent。

## 硬件环境要求

- 华为 Atlas 200I DK A2 开发板
- USB 摄像头
- 网络连接（用于下载模型文件）

## 软件环境要求

- 操作系统：Ubuntu 20.04 或更高版本
- Python 3.8 或更高版本
- CANN（Compute Architecture for Neural Networks）

## 部署步骤

### 1. 准备环境

#### 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
```

#### 创建 Python 虚拟环境

```bash
cd /path/to/Embeded-Agent
python3 -m venv venv
source venv/bin/activate
```

#### 安装基本依赖

```bash
pip install -e .  # 安装项目本身（如果有 setup.py）
```

### 2. 安装 CANN 软件包

按照华为官方文档安装 CANN 软件包以支持 NPU 加速。

### 3. 安装 YOLO 相关依赖

```bash
pip install -r requirements_yolo.txt
```

### 4. 下载 YOLO 模型

```bash
cd /path/to/Embeded-Agent

# 方法 1：使用 ultralytics 自动下载
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

# 方法 2：手动下载
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
```

### 5. 配置摄像头

确保摄像头被正确识别：

```bash
ls /dev/video*
```

### 6. 配置检测器

编辑 `src/adapters/yolo_pose_detection.py`，根据需要调整参数：

- `model_path`: YOLO 模型文件路径
- `device`: 推理设备，在 Atlas 上设置为 'npu'
- `confidence_threshold`: 检测置信度阈值
- `detection_interval`: 检测间隔（秒）

### 7. 运行程序

```bash
cd /path/to/Embeded-Agent
python3 -m src.main
```

### 8. 测试功能

在程序运行后，你可以：

1. 输入 `/mock posture leaning` 测试姿势提醒功能
2. 输入 `/pose start` 启动真实的 YOLO 姿势检测（需要摄像头）
3. 输入 `/state` 查看当前状态
4. 输入 `/help` 查看所有可用命令

## 从占位符到真实实现

当前 `YOLOPoseDetector` 类中的检测逻辑是占位符。以下是完整的实现示例：

```python
# 在 src/adapters/yolo_pose_detection.py 中替换 load_model 和 detect 方法

from ultralytics import YOLO
import cv2
import numpy as np

def load_model(self) -> bool:
    try:
        print(f"[YOLOPoseDetector] 正在加载模型: {self.model_path}")
        self._model = YOLO(self.model_path)
        if self.device == 'npu':
            self._model.to('npu')
        else:
            self._model.to(self.device)
        self._model_loaded = True
        print(f"[YOLOPoseDetector] 模型加载成功，设备: {self.device}")
        return True
    except Exception as e:
        print(f"[YOLOPoseDetector] 模型加载失败: {e}")
        return False

def detect(self) -> Optional[DetectionResult]:
    if not self._model_loaded or self._model is None:
        return None
    
    try:
        # 从摄像头获取图像
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[YOLOPoseDetector] 无法打开摄像头")
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # 运行 YOLO 姿势检测
        results = self._model(frame, verbose=False)
        
        for result in results:
            if result.keypoints is not None:
                kpts = result.keypoints.data
                if len(kpts) > 0:
                    posture = self._classify_posture(kpts[0])
                    activity = self._classify_activity(posture, kpts[0])
                    return DetectionResult(
                        posture=posture,
                        activity=activity,
                        confidence=float(result.boxes.conf[0]) if result.boxes else 0.8,
                        timestamp=int(time.time()),
                    )
        
        return None
    except Exception as e:
        print(f"[YOLOPoseDetector] 检测失败: {e}")
        return None

def _classify_posture(self, keypoints) -> str:
    # 基于关键点判断姿势
    # nose, left_eye, right_eye, left_ear, right_ear, 
    # left_shoulder, right_shoulder, left_elbow, right_elbow, 
    # left_wrist, right_wrist, left_hip, right_hip, 
    # left_knee, right_knee, left_ankle, right_ankle
    
    # 简单示例实现
    nose_y = float(keypoints[0][1])
    shoulder_y = float((keypoints[5][1] + keypoints[6][1]) / 2)
    
    if nose_y > shoulder_y + 50:
        return "leaning"
    # 添加更多姿势判断逻辑
    
    return "sitting"

def _classify_activity(self, posture, keypoints) -> str:
    # 基于姿势和关键点判断活动
    if posture == "sitting":
        return "studying"
    return "resting"
```

## 故障排查

### 摄像头无法打开

- 检查摄像头连接
- 检查权限：`sudo chmod 666 /dev/video0`
- 尝试其他设备：`/dev/video1`

### NPU 不可用

- 确认 CANN 正确安装
- 检查环境变量设置
- 回退到 CPU 模式：修改 device='cpu'

### 模型加载失败

- 检查模型文件路径是否正确
- 确认模型文件完整
- 尝试重新下载模型

## 性能优化建议

1. 使用更小的模型（如 yolov8n-pose）提升推理速度
2. 调整检测间隔（detection_interval），平衡实时性和资源消耗
3. 降低输入图像分辨率
4. 使用多线程或异步处理

## 安全注意事项

1. 摄像头视频流仅在本地处理，不发送到网络
2. 定期清理保存的图像和日志文件
3. 遵循隐私保护相关法规

## 参考资料

- [华为 CANN 文档](https://www.hiascend.com/document)
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [Atlas 200I DK A2 产品页](https://www.hiascend.com/hardware/developer-kits)
