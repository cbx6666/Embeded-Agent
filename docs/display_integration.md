# 显示屏与桌宠表情：我们要负责的 Event、Action 和 Adapter

本文只讲**和「显示屏负责桌宠表情输出，并给传感器预留接入口」**相关的内容。  
更细的屏幕驱动协议、帧动画资源、硬件引脚说明可继续补在本文后续小节或 PR 说明中。

---

## 一、我们这条线要负责什么

| 类别 | 我们关心的部分 |
|------|----------------|
| **Event（事件）** | `display_sensor_updated` |
| **Action（动作）** | `display`、`render_pet_expression` |
| **Adapter（适配器）** | `src/adapters/pet_display.py` |

**边界**：显示驱动、屏幕刷新率、亮度读取、挂接在显示板上的传感器采样，都放在 `src/adapters/`；内核只看到标准 `Event` / `Action`，不关心 SPI/I2C/UART、具体屏幕型号、动画资源格式。

---

## 二、白话：我们在做什么、逻辑是什么

1. 内核后续如果要让桌宠露出某种表情，会发标准 `Action`，而不是直接调用硬件驱动。
2. 显示适配器收到 `render_pet_expression` 或通用 `display` 后，负责把它翻译成具体显示屏上的图像、文本或动画。
3. 如果显示板上顺手接了亮度、帧率、触摸、姿态之类的小传感器，也由显示适配器采集后转成标准事件上报。
4. 适配器只调用 `handle_event` 往上送 `display_sensor_updated`，不直接改状态。
5. 内核同学后续决定这些事件是否写入状态、是否反过来触发表情切换或别的动作。

整体逻辑一句话：**Adapter → Event → 内核 → Action → Adapter**

---

## 三、Event 有哪些（与本业务线直接相关）

| Event（`type`） | 作用（白话） | 典型谁产生 |
|-----------------|-------------|------------|
| `display_sensor_updated` | 上报显示侧当前表情，以及屏幕侧可选传感数据（如亮度、FPS、扩展传感器值） | `PetDisplayAdapter` |

（仓库里已有类型以 `src/agent/event/types.py` 为准。）

---

## 四、Action 有哪些（本业务线场景下会怎么用）

| Action（`type`） | 作用（白话） | 在本业务线中 |
|------------------|-------------|--------------|
| `display` | 通用显示动作，可显示文本或状态 | 可继续兼容简单屏幕提示、调试文案 |
| `render_pet_expression` | 指定桌宠应呈现的表情/动画 | 作为显示屏主动作，供内核后续发出“happy / idle / sleepy / angry”等表情指令 |

（仓库里已有类型以 `src/agent/action/types.py` 为准。）

---

## 五、Adapter 有哪些（作用是什么）

| Adapter | 作用（白话） | 与本业务线的关系 |
|---------|-------------|------------------|
| `PetDisplayAdapter` | 消费 `display` / `render_pet_expression`，驱动具体显示硬件；同时可把显示侧采样转成 `display_sensor_updated` | 本业务线主适配器 |

---

## 六、与项目组 / 内核的衔接

- 本 PR 是否**新增或修改** `Event` / `Action` 类型：**是**；新增 `display_sensor_updated`、`render_pet_expression`。  
  - `display_sensor_updated.payload` 要点：`expression` 必填；`source` 必填；`brightness` / `fps` / `sensor_values` / `screen_id` 选填。  
  - `render_pet_expression.payload` 要点：`expression` 必填；`style` / `intensity` / `duration_ms` / `sensor_hint` 选填。
- **交付给内核/联调方**：关键 Event 的 payload 示例（建议由适配器按需上报，可 1Hz～2Hz 周期上报，或在显示状态变更时上报一次）。

```json
{
  "type": "display_sensor_updated",
  "timestamp": 1713772800,
  "payload": {
    "expression": "happy",
    "source": "pet_display",
    "brightness": 68,
    "fps": 24,
    "sensor_values": {
      "touch": false,
      "ambient_light": 312
    },
    "screen_id": "main_oled"
  }
}
```
