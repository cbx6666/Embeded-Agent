# YOLO26 权重（官方）

从 [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) 下载：

| 文件 | 用途 |
|------|------|
| `yolo26n.pt` | 目标检测（COCO class 67 = cell phone） |
| `yolo26n-pose.pt` | 人体姿态（手腕关键点 9/10） |

```bash
python scripts/download_yolo26_models.py
```

`.pt` 文件已加入 `.gitignore`，需在每台机器上执行上述命令。
