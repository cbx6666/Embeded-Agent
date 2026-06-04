# 共享 Python 环境使用说明（视觉测试）

本文档用于确保板子上的所有开发者和测试人员复用同一套 Python 环境，避免重复下载依赖，并快速完成摄像头与视觉链路测试。

## 目标

- 所有人使用统一环境：`/opt/ai-envs/shared`
- 测试和开发前先激活共享环境
- 新人按文档加入组后即可使用，无需重复安装
- 视觉测试前完成摄像头权限与 Ascend 运行时检查

> 本文档约定共享环境目录名为 `shared`，避免使用业务含义过强的目录名。

## 依赖清单（与仓库同步）

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 项目声明的 Python 依赖（安装源） |
| `requirements.lock.txt` | 共享环境当前已安装版本快照（排障/回溯） |
| `requirements-pose.txt` | 可选：YOLO 姿势检测（`--pose`，未默认装入 shared） |

**共享环境已安装的核心包（2026-06-01 更新）**：`opencv-python-headless`、`mediapipe`、`numpy`、`pygame`、`sounddevice`、`deepface`、`tf-keras`、`tensorflow`（随 DeepFace），以及板载 Ascend 的 `acl`（系统路径）。

## 管理员：安装 / 更新共享环境依赖

在能访问 PyPI 的机器上，由管理员执行（勿在个人 venv 重复装一套）：

```bash
cd /path/to/Embeded-Agent
source /opt/ai-envs/shared/bin/activate

# 安装或升级项目依赖
python -m pip install -r requirements.txt

# 生成锁文件供团队对照（提交到仓库）
python -m pip freeze > requirements.lock.txt
```

安装完成后做一次导入自检：

```bash
python -c "
import cv2, mediapipe, numpy, pygame, sounddevice, deepface, tf_keras
print('cv2', cv2.__version__)
print('mediapipe', mediapipe.__version__)
print('numpy', numpy.__version__)
print('pygame', pygame.version.ver)
print('deepface / tf_keras ok')
"
python -c "import acl; print('acl ok')"   # NPU 情绪需要
```

> 若 `pip` 报 DNS/网络错误，先确认外网或镜像可用后再重试。安装过程中若提示 Ascend 相关包缺 `decorator`/`psutil`，一般不影响视觉/桌宠测试，需要 NPU 全链路时再单独补装。

## 日常使用（所有人）

### 1) 开发/测试前先激活共享环境

若命令行前仍显示 **`(.venv)`**，说明项目虚拟环境仍在生效，会先盖住 shared。请先退出再激活：

```bash
cd ~/Embeded-Agent
deactivate          # 重复执行直到提示符里不再有 (.venv)
source /opt/ai-envs/shared/bin/activate
which python        # 必须是 /opt/ai-envs/shared/bin/python
echo $VIRTUAL_ENV   # 必须是 /opt/ai-envs/shared
```

> **历史问题（已修复）**：若 `source /opt/ai-envs/shared/bin/activate` 后 `which python` 仍指向 `/home/drbin/Embeded-Agent/.venv`，说明 shared 的 `activate` 被错误生成。管理员应把 `VIRTUAL_ENV` 指回 `/opt/ai-envs/shared`（见 `bin/activate` 内基于脚本路径的 `realpath` 写法），并检查 `pyvenv.cfg` 与 `bin/*` 脚本的 shebang。

**更省事（推荐）**：不依赖 `activate`，直接用共享 Python：

```bash
cd ~/Embeded-Agent
./scripts/run_with_shared_env.sh scripts/test_camera_modules.py --vision-only --emotion-backend deepface --camera 0
```

或：

```bash
/opt/ai-envs/shared/bin/python scripts/test_camera_modules.py --vision-only --emotion-backend deepface --camera 0
```

> `scripts/test_camera_modules.py` 在检测到当前 Python 缺 `cv2/mediapipe` 时，会自动改用共享 Python 重新运行（可用 `EMBED_NO_SHARED_REEXEC=1` 关闭）。

### 2) 运行项目测试

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

### 3) 快速确认版本

```bash
which python   # 应为 /opt/ai-envs/shared/bin/python

python -c "
import cv2, mediapipe, pygame, deepface, tf_keras
print('cv2', cv2.__version__, 'mediapipe', mediapipe.__version__, 'pygame', pygame.version.ver)
"
python -c "import acl; print('acl ok')" 2>/dev/null || echo "acl 不可用（仅影响 wujie-om NPU 情绪）"
ls -lh models/wujie/wujie_vgg19_static.om 2>/dev/null || true
```

## 测试前一次性准备（新同学必做）

### 1) 加入共享环境用户组

需要管理员执行（示例用户名 `alice`）：

```bash
sudo usermod -aG aiusers alice
```

### 2) 加入摄像头设备组

视觉测试需要访问 `/dev/video*`，需要在 `video` 组内。  
需要管理员执行（示例用户名 `alice`）：

```bash
sudo usermod -aG video alice
```

### 3) 如需 NPU 推理，加入 Ascend 设备组

需要管理员执行（示例用户名 `alice`）：

```bash
sudo usermod -aG HwBaseUser,HwHiAiUser alice
```

> 执行 `usermod` 后必须重新登录（或重启会话）才会生效。

### 4) 登录后自检组权限

```bash
groups
```

预期至少包含：`aiusers`、`video`。  
如果要跑 NPU，还应包含：`HwBaseUser`、`HwHiAiUser`。

## 视觉测试推荐流程

### 1) 激活共享环境

```bash
source /opt/ai-envs/shared/bin/activate
```

### 2) （仅 NPU）加载 Ascend 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 3) 检查摄像头可读

```bash
python - <<'PY'
import cv2
for idx in [0, 1]:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    ok, frame = cap.read()
    print("idx", idx, "open", cap.isOpened(), "ok", ok, "shape", None if frame is None else frame.shape)
    cap.release()
PY
```

### 4) 启动视觉 + 桌宠（推荐联调，仅需摄像头）

在项目根目录、已激活 shared 环境：

```bash
cd /path/to/Embeded-Agent

# 不调用 LLM，先确认摄像头疲劳/表情（DeepFace，无需 NPU）
python scripts/test_camera_modules.py --vision-only --emotion-backend deepface --camera 0

# 视觉 + pygame 桌宠窗口
python scripts/test_camera_modules.py --vision --screen --emotion-backend deepface --camera 0

# 桌宠全屏（VNC/HDMI，需 export DISPLAY=:1）
python scripts/test_camera_modules.py --screen-only --screen-fullscreen
python -m src.main
# 默认全栈（桌宠全屏+视觉+语音）；仅 CLI：python -m src.main --llm
```

完整 Agent（需配置 `.env` 中 `DEEPSEEK_API_KEY`）：

```bash
export DISPLAY=:1
python -m src.main
# 或指定摄像头：python -m src.main --camera 0
```

### 5) 仅启动视觉事件（main 入口）

- 只测疲劳（不测情绪）：

```bash
python -m src.main --vision --camera 0 --emotion-backend none
```

- 测 NPU OM 情绪（需先 `source` Ascend 环境变量）：

```bash
python -m src.main --vision --camera 0 --emotion-backend wujie-om
```

- 行为识别暂无摄像头自动管线，可在 CLI 用 `/mock behavior working` 等模拟。

## 团队约定（请遵守）

- 在本仓库做测试、调试、改代码时，统一使用 `/opt/ai-envs/shared`
- 不要在个人目录重复安装同一套视觉依赖（`mediapipe/opencv/acl` 等）
- 需要新增依赖时，先在群里/评审里同步，再由管理员统一更新共享环境
- 更新后建议同步 `requirements.lock.txt`，方便回溯和排障

## 常见问题

### Q1: 激活时报权限不足

- 先确认自己在 `aiusers` 组内：`groups`
- 若刚被加组，请重新登录后再试

### Q2: 导入失败或版本异常

- 确认激活的是共享环境：`which python`
- 预期应指向 `/opt/ai-envs/shared/bin/python`

### Q2b: deepface 报 `No module named 'tf_keras'`

多见于命令行前带有 `(.venv)`，实际用的是项目 `.venv` 而非 shared。任选其一：

```bash
deactivate
source /opt/ai-envs/shared/bin/activate
cd ~/Embeded-Agent
python scripts/test_camera_modules.py --vision-only --emotion-backend deepface --camera 0
```

或在当前 `.venv` 内补装：

```bash
pip install tf-keras
# 若 mediapipe 与 protobuf 冲突，再执行：
pip install "protobuf>=4.25.3,<5"
```

### Q3: 摄像头打不开（`can't open camera by index`）

- 确认自己在 `video` 组：`groups`
- 确认设备存在：`ls -l /dev/video*`
- 用上面的 OpenCV 小脚本测试 `camera index`（常见是 `0`）

### Q4: NPU 情绪不出结果（提示 ACL/OM 不可用）

- 先执行 Ascend 环境脚本（见上文步骤 2）
- 检查 ACL 是否可导入：

```bash
python -c "import acl; print('acl ok')"
```

- 检查 OM 文件是否存在（示例路径）：

```bash
ls -lh models/wujie/wujie_vgg19_static.om
```

### Q5: `python: command not found`

- 未激活环境时可能出现，先执行：

```bash
source /opt/ai-envs/shared/bin/activate
```

- 再确认：`which python`

### Q6: 组已添加但仍无权限

- `usermod` 后如果不重新登录，组信息不会刷新
- 退出当前终端会话并重新登录后再测

---

## YOLO26 手机 + 手腕邻近（行为分心）

依赖与权重（在项目根目录执行）：

```bash
source /opt/ai-envs/shared/bin/activate
pip install -r requirements-behavior.txt
python scripts/download_yolo26_models.py

# 导出 OM（必须用共享 Python，不要用 base conda 的 python3）
deactivate   # 若提示符有 (base)，先退出 conda
bash scripts/export_yolo26_to_om.sh

# ONNX 已生成、仅重跑 ATC 时：
# deactivate && SKIP_ONNX=1 bash scripts/export_yolo26_to_om.sh

# ATC 报 np.float_ / NumPy 2.0：说明 atc 调到了 conda 的 python3，务必 deactivate 后再跑
```

权重保存到 `models/yolo26/`（官方 release [v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0)）。

摄像头联调：

```bash
python scripts/test_phone_hand_detection.py --camera 0
# 无桌面时自动 --no-gui；约每 3s 打印心跳，避免误以为卡死
python scripts/test_phone_hand_detection.py --camera 0 --imgsz 320 --heartbeat 3
python scripts/test_phone_hand_detection.py --publish-events
```

若 `import ultralytics` 报 NumPy/matplotlib 冲突，请使用共享环境并保证 `numpy<2`（见 `requirements-behavior.txt`）。


