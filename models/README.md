# 模型资产

推理权重与 Ascend OM 均放在此目录，按子目录区分用途：

| 目录 | 说明 |
|------|------|
| `wujie/` | WuJie FER2013 情绪 OM（全栈默认） |
| `yolo26/` | 行为检测 YOLO26 OM（手机/姿态） |
| `sherpa-onnx-kws-*` | Sherpa 唤醒词（按需下载） |
| `vits-*` | Sherpa TTS（按需下载） |

运行时产物（SQLite、录音等）在 `data/`，不在此目录。
