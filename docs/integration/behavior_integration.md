# 行为识别模块

## Event

| Event.type | 语义 | 边界 |
|------------|------|------|
| `user_presence_updated` | 用户是否在场、离席或未知。 | 只上报在场事实。 |
| `user_attention_updated` | 用户当前是否专注、分心或空闲。 | 只上报注意力事实；`behavior` 是正式字段，不是附属字段。 |

建议的闭集：

- `presence`: `present` / `away` / `unknown`
- `attention`: `focused` / `distracted` / `idle`
- `behavior`: `working` / `phone_use` / `staring` / `desk_rest` / `away`

## Action

本模块当前无专用 Action。

说明：

- 行为模块只报告“看到了什么”。
- 如果用户分心、离席、长时间异常，仍然只发 Event。
- 是否提醒、如何提醒，由 LLM-centered `DecisionPipeline` 决策，再由 `ActionRealizer` 转成显示、语音、灯光模块的能力 Action。

## Adapter

| Adapter | 责任 |
|---------|------|
| `BehaviorAdapter` | 将摄像头、关键点、行为分类模型或规则结果转换成 `user_presence_updated` 与 `user_attention_updated`。 |
| `PhoneHandProximityDetector` | YOLO26 detect（COCO `cell phone`）+ YOLO26-pose（手腕）邻近判定。 |
| `PhoneHandCameraAdapter` | 摄像头线程 + 上述检测，上报 `phone_use` / `working`。 |

### YOLO26 手机 + 手腕邻近（已实现）

- 权重：Ultralytics 官方 [assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) — `yolo26n.pt`、`yolo26n-pose.pt`
- 下载：`python scripts/download_yolo26_models.py` → `models/yolo26/`
- 依赖：`pip install -r requirements-behavior.txt`
- 联调：`python scripts/test_phone_hand_detection.py`
- **手机漏检时**：默认开启 **Face Mesh 低头融合**（与 `vision_affect` 同源 MediaPipe，不共用 WuJie 情绪 OM）：`低头 + pose 有人在 + 手腕在画面上半区` 可辅助判 `phone_use`；日志字段 `head_assist=1`。关闭：`--no-head-down-fusion`。
- 调参：`--phone-conf 0.15`、`--hold-seconds 0.8`、`--distance-ratio 1.2`；对比 OM/PT：`--backend pt`。
