# adapters

`adapters/` 是输入输出适配层，负责把外部世界转换成系统可处理的事件，或把系统动作转换成具体输出。

## 职责

- 接收用户输入（CLI、mock、摄像头、麦克风、显示侧传感器等）
- 解析 mock 命令
- 执行控制台输出、显示输出、语音输出
- 为后续硬件接入提供边界

## 文件说明

- `cli_input.py`：命令行输入适配器
- `mock_input.py`：将 `/mock ...` 命令转换为状态更新事件（含 `emotion`、`fatigue`）
- `console_output.py`：将 `speak`、`display` 动作映射为控制台输出
- `pet_display.py`：桌宠显示适配器；消费 `display` / `render_pet_expression`，并可上报 `display_sensor_updated`
- `voice_adapter.py`：语音输入输出适配器；上报 `voice_input_captured`，消费 `speak` / `play_voice`
- `vision_affect/`：**唯一**承载摄像头检测逻辑（MediaPipe、EAR/PERCLOS、可选 ResNet18）；向上只发 **`user_fatigue_updated` / `user_emotion_updated`**。可调参数见包内 **`VisionAffectConfig`**。**不使用 YOLO**；内核不实现、也不应依赖本包内部算法。

## 视觉可选依赖

见仓库根目录 **`requirements-vision.txt`**。安装后可用：

```bash
python -m src.main --vision
python -m src.main --vision --raf-ckpt path/to/raf_resnet18.pth
```

环境变量 **`RAF_RESNET18_CKPT`** 可代替 `--raf-ckpt`。

## 后续扩展方向

- `mic_input.py` 接入更完整的实时语音输入
- `screen_output.py` / `tts_output.py` / `led_output.py` 等更细粒度的硬件适配器

适配层的原则是：外部接口可以变化，但内部事件和动作模型尽量稳定。
