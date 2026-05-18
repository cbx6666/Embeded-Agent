# 环境感知模块

## Event

| Event.type | 语义 | 边界 |
|------------|------|------|
| `light_level_updated` | 光照强度变化。 | 只上报环境事实，不表达“该不该提醒”。 |
| `temperature_humidity_updated` | 温湿度变化。 | 只上报环境事实。 |
| `noise_level_updated` | 噪声水平变化。 | 只上报环境事实。 |

建议的 payload 字段：

- `source`
- `level`
- `is_low_light`
- `temperature_c`
- `humidity_pct`
- `temperature_level`
- `humidity_level`
- `noise_db`
- `is_noisy`

## Action

本模块当前无专用 Action。

说明：

- 环境模块是感知模块，只负责发 Event。
- 如果 `DecisionPipeline` 决定做环境提醒，应由 `ActionRealizer` 转成显示、语音、灯光模块各自的能力 Action。

## Adapter

| Adapter | 责任 |
|---------|------|
| `environment_sensor_input.py` | 读取光照、温湿度、噪声传感器并发出标准环境事件。 |
