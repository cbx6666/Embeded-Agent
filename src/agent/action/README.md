# Action

## 职责

`action/` 定义 Agent 对外执行的标准动作词表和构造函数。它是 `ActionRealizer` 和 `DeviceAdapter` 之间的稳定协议。

## 不负责什么

本目录不理解用户语义，不调用 LLM，不做安全策略，不直接访问硬件。

## 核心文件

- `action_model.py`：统一 Action dataclass。
- `action_builders.py`：受控动作构造函数和动作白名单校验。
- `types.py`：注册 ActionType。

## 上游和下游

上游是 `decision/action_realizer.py`。下游是 `execution/device_adapter.py` 和具体 adapters。

## 扩展方式

新增 action：先加入 `types.py`，再添加 builder，并在 `DeviceAdapter` 或设备适配器中实现执行路径。

## 示例

```python
action = display("Focus timer started.", reason="focus_start")
```
