# Action

## 职责

`action/` 定义 Agent 对外执行的标准动作词表和构造函数。它是 `ActionRealizer` 和 `DeviceAdapter` 之间的稳定协议。

当前只保留 **5 种**真实可执行动作：

- `speak`
- `display`
- `start_timer`
- `stop_timer`
- `set_tts_volume`

已删除且不再生成/执行：`set_tts_voice`、`set_tts_speed`、`render_pet_expression`、`set_light_state`、`start_voice_capture`、`stop_voice_capture`。

## 不负责什么

本目录不理解用户语义，不调用 LLM，不做安全策略，不直接访问硬件。

## 核心文件

- `action_model.py`：统一 `Action` dataclass
- `action_builders.py`：五种动作的受控构造函数
- `types.py`：`ActionType` 闭集与 `ACTION_TYPE_SET`
- `realizer.py`：`ActionRealizer`，把 `Intent` 落地为 `Action[]`

## 上游和下游

- 上游：`decision/*_handler.py` 产出的 `Intent`
- 下游：`device/adapter.py` 及 `adapters/console_output.py`、板级语音/显示适配器

## 扩展方式

新增 action：先加入 `types.py` → `action_builders.py` → `realizer.py` → `device/adapter.py` 及硬件适配器 → 补测试。

## 示例

```python
from src.agent.action.action_builders import speak, display, start_timer

actions = [start_timer(1500), speak("开始专注 25 分钟。"), display("开始专注 25 分钟。")]
```
