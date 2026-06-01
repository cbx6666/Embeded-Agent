# 共享 Python 环境使用说明（视觉测试）

本文档用于确保板子上的所有开发者和测试人员复用同一套 Python 环境，避免重复下载依赖，并快速完成摄像头与视觉链路测试。

## 目标

- 所有人使用统一环境：`/opt/ai-envs/shared`
- 测试和开发前先激活共享环境
- 新人按文档加入组后即可使用，无需重复安装
- 视觉测试前完成摄像头权限与 Ascend 运行时检查

> 本文档约定共享环境目录名为 `shared`，避免使用业务含义过强的目录名。

## 日常使用（所有人）

### 1) 开发/测试前先激活共享环境

```bash
source /opt/ai-envs/shared/bin/activate
```

### 2) 运行项目测试

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

### 3) 快速确认版本

```bash
python -c "import cv2, mediapipe, acl; print('cv2', cv2.__version__, 'mediapipe', mediapipe.__version__, 'acl ok')"
ls -lh external/fer_wujie1010/FER2013_VGG19/wujie_vgg19_static.om
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

### 4) 启动视觉链路

- 只测疲劳（不测情绪）：

```bash
python -m src.main --vision --camera 0 --emotion-backend none
```

- 测 NPU OM 情绪：

```bash
python -m src.main --vision --camera 0 --emotion-backend wujie-om
```

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
ls -lh external/fer_wujie1010/FER2013_VGG19/wujie_vgg19_static.om
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


