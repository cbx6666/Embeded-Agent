# YOLO26 手机检测 — OM 部署与共用预处理

## 架构

```text
摄像头 BGR
    │
    ├─ 表情 WuJie OM：人脸 crop → resize_gray_face_patch (vision_common)
    │
    └─ 手机+手腕：
          letterbox_bgr_for_yolo (vision_common，与 Ultralytics 一致)
                │
                ├─ yolo26n.om      → decode_yolo_detect_output (Ultralytics NMS)
                └─ yolo26n-pose.om → decode_yolo_pose_output
                          │
                          └─ 手腕邻近 / 上半区手机 → BehaviorAdapter
```

**人在画内**：仅 `yolo26n-pose`（`person_count_pose`）。`yolo26n` detect **只用于手机** (cls67)。

**在场状态机**（`PersonPresenceTracker`，默认宽限 10 帧）：
- 有人 → `absent_frames=0`，`phase=present`
- 连续无人 → `absent_frames` 每帧 +1
- `absent_frames≤10`（`absent_grace`）且仅有手机 → 仍算分心
- `absent_frames>10` → `phase=left`，不再因手机判分心，直到再次有人

**分心（present）**：检出手机 且（手腕邻近 / 放宽邻近 / 上半区高置信手机）。

- **共用**：`src/adapters/vision_common/acl_runtime.py`（`AscendOmSession`，WuJie 与 YOLO 共用）
- **前处理**：`letterbox_bgr_for_yolo` / `resize_gray_face_patch`
- **后处理**：`yolo_ultralytics_ops.py` 调用 `ultralytics.utils.nms.non_max_suppression` 与 `scale_boxes` / `scale_coords`

## 生成 .om

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/export_yolo26_to_om.sh
# 默认 IMGSZ=320, SOC_VERSION=Ascend310B1（Atlas 200I DK A2）
```

产物：`models/yolo26/yolo26n.om`、`models/yolo26/yolo26n-pose.om`

## 运行

```bash
# 自动：有 .om 则用 NPU，否则 .pt + CPU
python scripts/test_phone_hand_detection.py --camera 0 --backend auto

# 强制 OM
python scripts/test_phone_hand_detection.py --camera 0 --backend om --imgsz 320

# 强制 PyTorch
python scripts/test_phone_hand_detection.py --camera 0 --backend pt --device cpu
```

## 说明

- OM **不是**零代码：前处理、ACL 推理、后处理已封装，但 **ATC 转换需在板子上做一次**。
- 后处理依赖已安装的 `ultralytics`（只用其 NMS/坐标变换，推理在 NPU）。
- 当前 ONNX/OM 为 YOLO26 **end2end** 输出 `(1,300,6)` / `(1,300,57)`，后处理已自动识别；若重导 `nms=False` 则走原始头 + Ultralytics NMS。
- 同一 NPU 上加载多个 `.om` 须共用 ACL 上下文（见 `acl_runtime._AclDeviceRuntime`），否则第二个模型会导致第一个 `execute` 失败。
