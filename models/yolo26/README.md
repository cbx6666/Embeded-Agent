# YOLO26 权重（官方）

从 [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) 下载：

| 文件 | 用途 |
|------|------|
| `yolo26n.pt` | 目标检测（COCO class 67 = cell phone） |
| `yolo26n-pose.pt` | 人体姿态（手腕关键点 9/10） |
| `yolo26n.om` / `yolo26n-pose.om` | Ascend 部署用（已提交） |

```bash
python scripts/download_yolo26_models.py
# 需要重新导出 OM 时：
bash scripts/export_yolo26_to_om.sh
```

`.pt` 与 `.onnx` 已加入 `.gitignore`；`.pt` 需在每台机器上下载，`.onnx` 为 ATC 中间产物。
