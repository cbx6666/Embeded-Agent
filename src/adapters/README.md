# adapters

`adapters/` 是输入输出适配层，负责把外部世界转换成系统可处理的事件，或把系统动作转换成具体输出。

## 职责

- 接收用户输入（CLI、mock、摄像头、麦克风、显示侧传感器等）
- 解析 mock 命令
- 执行控制台输出、显示输出、语音输出
- 为后续硬件接入提供边界

## 文件说明

- `cli_input.py`：命令行输入适配器
- `mock_input.py`：将 `/mock ...` 命令转换为状态更新事件（含 `presence`、`attention`、`behavior`、`emotion`、`fatigue`）
- `console_output.py`：将 `speak`、`display` 动作映射为控制台输出
- `pet_display.py`：桌宠显示适配器；消费 `display` / `render_pet_expression` / `set_light_state`，并上报 `display_sensor_updated`
- `voice_adapter.py`：语音输入输出适配器；上报 `speech_recognized`，消费 `speak`
- `behavior_adapter.py`：行为识别适配器；发出行为线索与行为汇总事件
- `vision_affect/`：摄像头 + **MediaPipe Face Mesh**；**疲劳** 由 **EAR**（眼部 PERCLOS）与 **MAR**（打哈欠/张口，滑动窗内占比）加权融合后滞回分档；**情绪** 默认 **WuJie-OM**，也支持 `wujie-vgg19`、`raf`、`deepface`、`none`。模型和后端放在 **`backends/`**，**向上只发** `user_fatigue_updated` / `user_emotion_updated`。**不使用 YOLO**；内核不 import 本包内部实现。

## 视觉依赖

见根目录 **`requirements.txt`**（含 `mediapipe`、`deepface` 等）。启动示例：

```bash
python -m src.main --vision
python -m src.main --vision --emotion-backend raf --raf-ckpt path/to/raf_resnet18.pth
python -m src.main --vision --emotion-backend wujie-om --wujie-om path/to/model.om
```

- 环境变量 **`EMBED_EMOTION_BACKEND`** 可设 `wujie-om` / `wujie-vgg19` / `raf` / `deepface` / `none`（覆盖 `--emotion-backend`）。
- 环境变量 **`WUJIE_OM_MODEL`** 和 **`WUJIE_OM_DEVICE_ID`** 可配置 WuJie-OM 模型路径和 Ascend 设备 ID。
- 环境变量 **`WUJIE_VGG19_CKPT`** 可配置 WuJie VGG19 checkpoint。
- 环境变量 **`RAF_RESNET18_CKPT`** 在 RAF 模式下可代替 `--raf-ckpt`。

## 后续扩展方向

- `mic_input.py` 接入更完整的实时语音输入
- `screen_output.py` / `tts_output.py` / `led_output.py` 等更细粒度的硬件适配器

适配层的原则是：外部接口可以变化，但内部事件和动作模型尽量稳定。
