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
- 三类 environment Event 在 `DecisionPipeline` 中**跳过 LLM**（只更新 state，不触发对话决策）。
- 若后续需要环境提醒，应由内核规则或 `ActionRealizer` 转成显示、语音、灯光 Action。

## Adapter（ESP32 / STM32 USB）

外设由 **ESP32**（或 STM32 中转）采集，经 **USB 串口（CH340，`/dev/ttyUSB0`）** 以文本行发给 Atlas。

| 组件 | 路径 |
|------|------|
| `Esp32EnvironmentAdapter` | `src/adapters/environment/esp32_sensor_adapter.py` |
| 行解析 | `src/adapters/environment/parser.py` |
| 分级阈值 | `src/adapters/environment/levels.py` |
| 串口读取 | `src/adapters/environment/serial_reader.py`（优先 pyserial，回退 termios） |

### 支持的串口格式

**1. ESP32 JSON（当前固件，每行一条，115200 8N1）**

| ESP32 字段 | Agent Event / 字段 |
|------------|-------------------|
| `temperature` | `temperature_humidity_updated` → `temperature_c` |
| `humidity` | `temperature_humidity_updated` → `humidity_pct` |
| `lux` | `light_level_updated` → `light_lux` |
| `noise_db` | `noise_level_updated` → `noise_db` |

示例：

```json
{"temperature":25.90,"humidity":61.50,"lux":0.00,"noise_db":70.33}
```

**2. STM32 JSON（字段别名）**

```json
{"temperature_c":24.5,"humidity_pct":42.0,"light_lux":88.0,"noise_db":55.0}
```

**3. Legacy 光照文本**

```
Light: 150.5 lx
```

单行可只含部分字段；适配器会与上次读数合并后再发 Event。

### level 枚举（默认阈值与 asr-test 对齐）

| 维度 | level 值 |
|------|----------|
| 光照 | `dark` / `low` / `normal` / `bright` |
| 温度 | `low` / `normal` / `high` |
| 湿度 | `dry` / `normal` / `humid` |
| 噪声 | `low` / `normal` / `high` |

### 启动

全栈 `main` 默认开启（`--no-environment` 可关）：

```bash
python -m src.main --no-screen --no-voice
# 或单独测传感器：
python scripts/test_esp32_environment.py --duration 15
```

环境变量：

- `EMBED_ESP32_SENSOR_PORT`（默认 `/dev/ttyUSB0`）
- `EMBED_ESP32_SENSOR_BAUD`（默认 `115200`）
- `EMBED_ENV_LOW_LIGHT_LUX`（默认 `120`）
- `EMBED_ENV_LOW_TEMPERATURE_C`（默认 `18`）
- `EMBED_ENV_DRY_HUMIDITY_PCT`（默认 `30`）
- `EMBED_ENV_NOISY_DB`（默认 `65`）

CLI：

- `--esp32-sensor-port` / `--esp32-sensor-baud`
- `--env-low-light-lux` / `--env-low-temperature-c` / `--env-dry-humidity-pct` / `--env-noisy-db`
