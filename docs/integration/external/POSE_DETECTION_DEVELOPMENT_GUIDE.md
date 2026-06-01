# 行为检测功能完整开发指南

## 目录

1. [项目概述](#项目概述)
2. [当前状态](#当前状态)
3. [硬件准备](#硬件准备)
4. [网络配置](#网络配置)
5. [环境搭建](#环境搭建)
6. [YOLO 模型集成](#yolo-模型集成)
7. [姿势分类算法](#姿势分类算法)
8. [测试与验证](#测试与验证)
9. [优化与调优](#优化与调优)
10. [常见问题](#常见问题)

---

## 项目概述

本项目旨在为桌宠 Agent 添加基于 YOLO 的实时姿势和行为检测功能，实现：

- **姿势检测**：识别用户是端坐、站立、趴着还是躺着
- **行为识别**：判断用户是否在学习、工作或休息
- **智能提醒**：在专注模式下提醒用户保持正确姿势

**目标平台**：华为 Atlas 200I DK A2 开发板（NPU 加速推理）

---

## 当前状态

### 已完成的工作

✅ **核心架构集成**：
- 扩展了状态模型，支持 `posture` 和 `activity` 字段
- 添加了相应的事件类型：`user_posture_updated`、`user_activity_updated`
- 实现了 reducer 和 policy 逻辑
- 集成到主程序，支持 `/pose start/stop` 命令

✅ **适配器框架**：
- `YOLOPoseDetector`：模型加载和推理封装（当前为占位符）
- `PoseDetectionAdapter`：检测循环和事件转换

✅ **测试覆盖**：
- 完整的单元测试

✅ **文档**：
- 部署指南
- 功能总结

### 待完成的核心任务

⚠️ **YOLO 真实推理实现**：
- 替换 `yolo_pose_detection.py` 中的占位符代码
- 在 Atlas 200I DK A2 上完成 NPU 推理

⚠️ **姿势分类算法**：
- 基于 YOLO 关键点实现准确的姿势判断
- 学习/工作/休息状态识别

⚠️ **网络配置**：
- 解决开发板联网问题

---

## 硬件准备

### 必需硬件

1. **Atlas 200I DK A2 开发板**
2. **USB 摄像头**（UVC 兼容，推荐 1080p/720p）
3. **USB 数据线**（Type-A to Type-C，支持数据传输）
4. **电源适配器**（单独供电）

### 可选硬件

1. **显示器**（HDMI 连接开发板，方便调试）
2. **键盘鼠标**（通过 USB 连接）
3. **网线**（如果使用路由器上网）

### 硬件连接

```
┌─────────────────┐
│  USB Camera     │ ◄─── USB 2.0/3.0
└─────────────────┘
         │
         ▼
┌───────────────────────────────┐
│  Atlas 200I DK A2             │
│  ├─ USB (摄像头)              │
│  ├─ USB (电脑/调试)           │
│  ├─ HDMI (显示器)             │
│  └─ Power (电源)              │
└───────────────────────────────┘
```

---

## 网络配置

### 方案一：通过 USB 共享电脑网络（推荐调试用）

#### Windows 端配置

1. **连接开发板**：用 USB 线连接电脑和开发板
2. **配置 USB 适配器 IP**：
   - 打开「网络和共享中心」→「更改适配器设置」
   - 找到「本地连接」或「USB Ethernet」适配器
   - 右键 → 属性 → IPv4
   - 设置 IP: `192.168.137.1`，子网掩码: `255.255.255.0`
3. **启用网络共享**：
   - 右键点击你的 WLAN/以太网连接
   - 属性 → 共享
   - 勾选「允许其他网络用户通过此计算机的 Internet 连接来连接」
   - 在下拉菜单中选择 USB 适配器

#### 开发板端配置

```bash
# 查看网络接口
ip addr show

# 配置 USB 网络（usb0 或 eth0）
sudo dhclient usb0

# 手动设置（如果 dhclient 失败）
sudo ip addr add 192.168.137.2/24 dev usb0
sudo ip route add default via 192.168.137.1

# 配置 DNS
sudo rm /etc/resolv.conf
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 114.114.114.114" | sudo tee -a /etc/resolv.conf

# 测试网络
ping -c 3 www.baidu.com
```

### 方案二：直接插网线（推荐部署用）

1. 用网线连接开发板的网口和路由器
2. 在开发板上执行：

```bash
# 自动获取 IP
sudo dhclient eth0

# 查看 IP
ip addr show eth0

# 测试网络
ping -c 3 www.baidu.com
```

### 方案三：使用 WiFi（如果开发板支持）

```bash
# 查看可用 WiFi
iwlist wlan0 scan | grep ESSID

# 连接 WiFi
sudo nmcli dev wifi connect "你的WiFi名" password "你的密码"

# 或使用 wpa_supplicant
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf
sudo dhclient wlan0
```

---

## 环境搭建

### 基础环境

**开发板系统**：Ubuntu 22.04 LTS（预装）

**Python 版本**：3.10+

### 步骤 1：系统依赖

```bash
# 更新系统
sudo apt update
sudo apt upgrade -y

# 安装必要的系统库
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    zlib1g-dev \
    libusb-1.0-0-dev
```

### 步骤 2：安装 CANN（华为 NPU 驱动）

**重要**：这是在 Atlas 200I DK A2 上运行 YOLO 的关键！

1. **下载 CANN 软件包**：
   - 访问 [华为昇腾社区](https://www.hiascend.com/software/cann)
   - 下载与你的开发板匹配的 CANN 版本（推荐 6.0.0.alpha005 或更高）
   - 需要注册账号并申请权限

2. **安装 CANN**：

```bash
# 上传软件包到开发板
# 可以通过 scp：
# scp cann_software.run HwHiAiUser@192.168.0.2:~

# 切换到 root 用户
sudo -i

# 给文件添加执行权限
chmod +x cann_software.run

# 安装
./cann_software.run --install

# 配置环境变量
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc

# 验证安装
npu-smi info
```

3. **如果遇到问题**：
   - 参考 [华为官方文档](https://www.hiascend.com/document)
   - 检查开发板是否有 NPU 硬件

### 步骤 3：Python 依赖

```bash
# 进入项目目录
cd ~/Embeded-Agent

# 创建并激活虚拟环境（可选）
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install -r requirements.txt

# 安装 YOLO 相关依赖
pip install -r requirements_yolo.txt
```

**requirements_yolo.txt 内容**：
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
PyYAML>=6.0
```

### 步骤 4：下载 YOLO 模型

```bash
# 方法一：通过 ultralytics 自动下载
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

# 方法二：手动下载
# 访问 https://github.com/ultralytics/assets/releases
# 下载 yolov8n-pose.pt（nano 版本，适合嵌入式设备）
# 或 yolov8s-pose.pt（small 版本，准确率更高）

# 放在项目根目录或指定路径
```

---

## YOLO 模型集成

### 当前占位符代码分析

打开 `src/adapters/yolo_pose_detection.py`，找到以下需要替换的部分：

1. **模型加载**（`load_model` 方法）
2. **推理实现**（`detect` 方法）

### 完整实现方案

#### 方案 A：使用 ultralytics + 任意设备（CPU/GPU/NPU）

```python
from ultralytics import YOLO
import numpy as np

class YOLOPoseDetector:
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",  # 可选: "cpu", "cuda", "npu"
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model_loaded = False
        self._model = None

    def load_model(self) -> bool:
        try:
            print(f"[YOLOPoseDetector] 正在加载模型: {self.model_path}")
            print(f"[YOLOPoseDetector] 使用设备: {self.device}")
            
            # 加载 YOLO 模型
            self._model = YOLO(self.model_path)
            
            # 如果是 NPU，需要额外配置（ultralytics 8.0.200+ 支持）
            if self.device == "npu":
                # 注意：需要 ultralytics 有华为 NPU 后端支持
                # 或使用华为自己的推理框架
                pass
            
            self._model_loaded = True
            return True
        except Exception as e:
            print(f"[YOLOPoseDetector] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect(self) -> Optional[DetectionResult]:
        if not self._model_loaded:
            print("[YOLOPoseDetector] 模型未加载，无法进行检测")
            return None

        try:
            import cv2
            
            # 打开摄像头（0 是默认摄像头）
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[YOLOPoseDetector] 无法打开摄像头")
                return None
            
            # 读取一帧
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("[YOLOPoseDetector] 无法读取摄像头画面")
                return None
            
            # 运行 YOLO 推理
            results = self._model(frame, device=self.device, verbose=False)
            
            # 处理结果
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.data.cpu().numpy()
                
                # 如果检测到多人，取第一个
                if len(keypoints) > 0:
                    person_kps = keypoints[0]
                    
                    # 基于关键点判断姿势
                    posture = self._classify_posture(person_kps)
                    activity = self._classify_activity(person_kps, frame)
                    confidence = float(results[0].boxes.conf[0]) if results[0].boxes is not None else 0.5
                    
                    return DetectionResult(
                        posture=posture,
                        activity=activity,
                        confidence=confidence,
                        timestamp=int(time.time()),
                    )
            
            return None
            
        except Exception as e:
            print(f"[YOLOPoseDetector] 检测出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _classify_posture(self, keypoints: np.ndarray) -> str:
        """基于关键点判断姿势。
        
        YOLOv8 关键点顺序：
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        
        Args:
            keypoints: (17, 3) 数组 (x, y, confidence)
            
        Returns:
            posture: "sitting", "standing", "leaning", "lying", "unknown"
        """
        # 关键点置信度检查
        valid_kps = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kps) < 10:
            return "unknown"
        
        # 提取关键关节
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        
        # 计算肩膀、髋关节、膝盖的平均 Y 坐标
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        
        # 计算肩膀和鼻子的高度差
        nose_shoulder_diff = nose[1] - shoulder_y
        
        # 判断姿势（需要根据摄像头角度调整阈值）
        if nose_shoulder_diff > 50:  # 鼻子比肩膀低很多 → 趴着
            return "leaning"
        
        # 判断是坐着还是站着（基于髋关节和膝盖的相对位置）
        hip_knee_diff = abs(hip_y - knee_y)
        shoulder_hip_diff = abs(shoulder_y - hip_y)
        
        if hip_knee_diff < 30 and shoulder_hip_diff > 50:
            # 膝盖和髋关节几乎在同一高度 → 坐着
            return "sitting"
        elif shoulder_hip_diff > 80 and hip_knee_diff > 50:
            # 肩膀、髋关节、膝盖都有明显距离 → 站着
            return "standing"
        else:
            # 其他情况需要更多判断
            return "sitting"  # 默认坐着
    
    def _classify_activity(self, keypoints: np.ndarray, frame) -> str:
        """基于关键点和画面判断活动状态。
        
        Args:
            keypoints: (17, 3) 数组
            frame: 图像帧（用于场景分析）
            
        Returns:
            activity: "studying", "working", "resting"
        """
        # 简单实现：基于手腕位置判断
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # 如果手腕在肩膀高度附近，可能在学习/工作
        wrist_shoulder_level = (
            abs(left_wrist[1] - left_shoulder[1]) < 80 or
            abs(right_wrist[1] - right_shoulder[1]) < 80
        )
        
        if wrist_shoulder_level:
            # 可以进一步通过图像分析是否有书本、电脑等（可选）
            return "studying"
        else:
            return "resting"
```

#### 方案 B：使用华为 ATC 转换模型（NPU 优化）

如果需要更好的 NPU 性能，可以使用华为 ATC 工具转换模型：

```bash
# 1. 导出 ONNX 模型
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
model.export(format='onnx', opset=12)
"

# 2. 使用 ATC 转换为 OM 模型（华为 NPU 专用格式）
atc --model=yolov8n-pose.onnx --framework=5 --output=yolov8n-pose --input_format=NCHW --input_shape="images:1,3,640,640" --out_nodes="output0:0;output1:0" --soc_version=Ascend310B1

# 3. 使用华为推理 API（ACL）加载和运行 OM 模型
# 这部分需要使用华为的 CANN Python API
```

### 关键文件修改总结

你需要修改的文件：
```
src/adapters/yolo_pose_detection.py
├── load_model()  # 替换为真实的 YOLO 加载
└── detect()      # 替换为真实的推理 + 姿势分类
    ├── _classify_posture()   # 姿势判断
    └── _classify_activity()  # 活动识别
```

---

## 姿势分类算法

### YOLOv8 Pose 关键点说明

YOLOv8 检测到的 17 个人体关键点：

| ID | 部位 | 说明 |
|----|------|------|
| 0 | nose | 鼻子 |
| 1 | left_eye | 左眼 |
| 2 | right_eye | 右眼 |
| 3 | left_ear | 左耳 |
| 4 | right_ear | 右耳 |
| 5 | left_shoulder | 左肩 |
| 6 | right_shoulder | 右肩 |
| 7 | left_elbow | 左肘 |
| 8 | right_elbow | 右肘 |
| 9 | left_wrist | 左手腕 |
| 10 | right_wrist | 右手腕 |
| 11 | left_hip | 左髋 |
| 12 | right_hip | 右髋 |
| 13 | left_knee | 左膝 |
| 14 | right_knee | 右膝 |
| 15 | left_ankle | 左踝 |
| 16 | right_ankle | 右踝 |

每个关键点格式：`[x, y, confidence]`

### 姿势判断策略（进阶版）

#### 1. 计算关键角度

```python
def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """计算三点形成的角度（p2 是顶点）。"""
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.arccos(np.clip(cos_angle, -1, 1)) * (180 / np.pi)
    return angle
```

#### 2. 坐姿判断

```python
def _is_sitting(self, keypoints: np.ndarray) -> bool:
    # 判断髋关节和膝盖的角度
    left_knee_angle = self._calculate_angle(
        keypoints[11],  # hip
        keypoints[13],  # knee
        keypoints[15],  # ankle
    )
    right_knee_angle = self._calculate_angle(
        keypoints[12],
        keypoints[14],
        keypoints[16],
    )
    
    # 如果膝盖角度小于 120 度，很可能是坐着
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    return avg_knee_angle < 120
```

#### 3. 站姿判断

```python
def _is_standing(self, keypoints: np.ndarray) -> bool:
    # 肩膀、髋关节、膝盖应该近似在一条直线上
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
    
    # 判断比例
    shoulder_hip_ratio = abs(shoulder_y - hip_y) / (hip_y - knee_y + 1e-8)
    return 0.8 < shoulder_hip_ratio < 1.2  # 近似 1:1
```

#### 4. 趴姿判断

```python
def _is_leaning(self, keypoints: np.ndarray) -> bool:
    # 鼻子和肩膀的 Y 坐标差
    nose_y = keypoints[0][1]
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    
    # 如果鼻子比肩膀低很多，很可能是趴着
    return (nose_y - shoulder_y) > 50
```

### 学习/工作/休息状态识别

可以基于以下特征：

1. **手腕位置**：在学习时，手腕通常在桌面高度
2. **头部姿态**：低头时更可能在学习
3. **场景分析**（可选）：检测画面中是否有书本、电脑

```python
def _classify_activity(self, keypoints: np.ndarray, frame) -> str:
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    # 判断手腕是否在桌面高度（相对于肩膀）
    wrist_height_near_shoulder = (
        abs(left_wrist[1] - left_shoulder[1]) < 100 or
        abs(right_wrist[1] - right_shoulder[1]) < 100
    )
    
    # 判断头部是否在低头状态
    nose_y = keypoints[0][1]
    eye_avg_y = (keypoints[1][1] + keypoints[2][1]) / 2
    head_down = nose_y > eye_avg_y + 10
    
    if wrist_height_near_shoulder or head_down:
        return "studying"
    else:
        return "resting"
```

### 算法优化建议

1. **使用时间平滑**：不要只看单帧，而是综合最近几帧的结果
2. **个性化阈值**：让用户可以校准自己的姿势基准
3. **多人处理**：如果有多个人，选择最大或最近的人
4. **背景过滤**：只关注感兴趣区域（ROI）

---

## 测试与验证

### 步骤 1：单元测试

```bash
# 运行测试
cd ~/Embeded-Agent
python -m pytest tests/test_pose_detection.py -v

# 或运行所有测试
python -m unittest discover -s tests -v
```

### 步骤 2：Mock 测试（不依赖硬件）

```bash
# 启动程序
python -m src.main

# 测试命令
/mock posture sitting
/mock posture leaning
/mock posture standing
/mock activity studying

# 查看状态
/state

# 启动专注模式
开始专注 25 分钟

# 模拟趴着（会触发提醒）
/mock posture leaning
```

### 步骤 3：真实摄像头测试

```bash
# 启动程序
python -m src.main

# 启动姿势检测
/pose start

# 观察输出
# 应该会看到检测到的姿势和活动

# 测试提醒
开始专注 25 分钟
# 然后故意趴着，等待提醒

# 停止检测
/pose stop
```

### 步骤 4：性能测试

```bash
# 测试检测 FPS
python3 -c "
from src.adapters.yolo_pose_detection import YOLOPoseDetector
import time

detector = YOLOPoseDetector()
detector.load_model()

start = time.time()
count = 0
for i in range(100):
    result = detector.detect()
    if result:
        count += 1
end = time.time()

print(f'检测速度: {count / (end - start):.2f} FPS')
"
```

---

## 优化与调优

### 1. 性能优化

#### 降低分辨率

```python
# 在 detect() 中调整输入尺寸
results = self._model(frame, imgsz=320, device=self.device)  # 320 而不是 640
```

#### 使用更小的模型

- `yolov8n-pose.pt` (nano) - 最快，推荐
- `yolov8s-pose.pt` (small) - 平衡
- `yolov8m-pose.pt` (medium) - 更准确但更慢

#### 降低检测频率

```python
# 在 main.py 或配置中调整
PoseDetectionAdapter(
    detector=detector,
    event_callback=agent_core.handle_event,
    detection_interval=5.0,  # 改为 5 秒一次（原来是 1-2 秒）
)
```

### 2. 准确率优化

#### 调整置信度阈值

```python
YOLOPoseDetector(
    confidence_threshold=0.3,  # 降低阈值以捕获更多结果
)
```

#### 基于场景的校准

让用户可以校准自己的姿势：

```bash
# 新增命令：校准坐姿
/calibrate sitting

# 新增命令：校准站姿
/calibrate standing
```

### 3. 用户体验优化

#### 调整冷却时间

在 `src/agent/policy.py` 中调整：

```python
COOLDOWN_DURATION = 10 * 60  # 10 分钟，可调整
```

#### 自定义提醒文本

修改 `policy.py` 中的提醒文案：

```python
speak("你趴在桌上啦，坐端正一点对颈椎好哦~")
```

---

## 常见问题

### Q1: 无法打开摄像头

**问题**：`cv2.VideoCapture(0)` 失败

**解决方案**：
```bash
# 检查摄像头设备
ls -l /dev/video*

# 检查权限
sudo chmod 666 /dev/video0

# 确认摄像头被识别
v4l2-ctl --list-devices
```

### Q2: YOLO 推理太慢

**问题**：FPS 太低

**解决方案**：
- 使用更小的模型（yolov8n-pose）
- 降低输入尺寸（imgsz=320）
- 使用 NPU 加速（CANN）
- 降低检测频率

### Q3: 姿势判断不准确

**问题**：经常误判

**解决方案**：
- 调整 `_classify_posture` 中的阈值
- 使用时间平滑（连续几帧都一致才判断）
- 进行个性化校准
- 改进算法（使用机器学习模型而不是规则）

### Q4: 开发板无法联网

**问题**：之前遇到的问题

**解决方案**：
1. 优先考虑用网线连接路由器
2. 如果一定要用 USB 共享：
   - 确认 Windows 网络共享已启用
   - 检查防火墙设置
   - 手动配置 IP 和 DNS
3. 用手机热点（如果有 USB 网卡）

### Q5: CANN/NPU 驱动安装失败

**问题**：无法安装或使用 NPU

**解决方案**：
- 确认开发板型号匹配的 CANN 版本
- 参考 [华为官方文档](https://www.hiascend.com/document)
- 可以先用 CPU 模式测试，再优化到 NPU
- 如果 NPU 实在有问题，可以用 CPU 模式运行（性能会差一些）

---

## 开发路线图

### Phase 1: 基础功能（当前）
- ✅ 集成 YOLO 占位符
- ✅ 事件和状态扩展
- ⏳ **替换为真实 YOLO 推理**（下一步）
- ⏳ 实现基础姿势分类

### Phase 2: 算法优化
- ⏳ 基于关键点的准确姿势判断
- ⏳ 学习/工作/休息状态识别
- ⏳ 时间平滑和去抖动

### Phase 3: 性能优化
- ⏳ NPU 推理优化
- ⏳ 降低资源占用
- ⏳ 热启动和快速恢复

### Phase 4: 功能增强
- ⏳ 更多姿势类型
- ⏳ 疲劳检测（基于眨眼、姿态变化）
- ⏳ 个性化配置
- ⏳ 可视化展示

---

## 参考资源

### 文档
- [项目 README](../README.md)
- [功能总结](./FEATURE_SUMMARY_POSE_DETECTION.md)
- [部署指南](./atlas_200i_deployment_guide.md)
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [华为 CANN 文档](https://www.hiascend.com/document)

### 代码
- `src/adapters/yolo_pose_detection.py` - 主要修改文件
- `src/agent/state/types.py` - 状态类型
- `src/agent/event/types.py` - 事件类型
- `src/agent/policy.py` - 决策逻辑
- `tests/test_pose_detection.py` - 测试文件

### 工具
- `config_pose_detection.yaml` - 配置文件示例
- `requirements_yolo.txt` - 依赖清单

---

## 快速检查清单

在开始之前，请确认：

- [ ] 硬件连接正确（摄像头、USB、电源）
- [ ] 开发板能正常启动
- [ ] 网络已配置好（能访问外网安装依赖）
- [ ] Python 环境已搭建（Python 3.10+）
- [ ] 项目代码已传输到开发板
- [ ] YOLO 模型已下载
- [ ] （可选）CANN 已安装用于 NPU 加速

完成上述步骤后，开始替换 `yolo_pose_detection.py` 中的占位符代码！

---

**祝你开发顺利！** 🚀

如有问题，可以查看已有文档或继续讨论。
