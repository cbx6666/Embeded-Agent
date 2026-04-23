# 显示与灯光输出模块

## Event

| Event.type | 语义 | 边界 |
|------------|------|------|
| `display_sensor_updated` | 显示侧设备状态变化，如亮度、FPS、触摸状态。 | 只上报设备事实，不承载用户状态和业务结果。 |

## Action

| Action.type | 语义 | 边界 |
|-------------|------|------|
| `display` | 显示文本、状态、结果或界面。 | 显示模块只执行渲染，不理解用户命令。提醒语义放在 payload 的 `kind` / `level` / `reason` 中。 |
| `render_pet_expression` | 渲染桌宠表情或视觉动画。 | 只负责视觉表现，不负责提醒策略。 |
| `set_light_state` | 设置灯光状态、颜色、模式、亮度。 | 只负责灯光执行，不负责决定何时提醒。 |

建议的 `display.payload` 字段：

- `text`
- `title`
- `status`
- `kind`
- `level`
- `reason`

建议的 `set_light_state.payload` 字段：

- `state`
- `color`
- `pattern`
- `brightness`
- `duration_ms`
- `kind`
- `level`
- `reason`

## Adapter

| Adapter | 责任 |
|---------|------|
| `PetDisplayAdapter` | 消费 `display`、`render_pet_expression`，驱动屏幕和 UI。 |
| `LightOutputAdapter` | 消费 `set_light_state`，驱动灯光设备。 |
