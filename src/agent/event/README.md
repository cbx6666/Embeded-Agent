# Event

## 职责

`event/` 定义进入 AgentCore 的标准事件模型。所有外部输入都必须先被 adapters 转成 Event。

## 不负责什么

本目录不读设备、不做语义理解、不生成 Intent 或 Action。

## 核心文件

- `event_model.py`：统一 Event dataclass。
- `types.py`：注册 EventType。
- `event_builders.py`：传感器/语音/行为输入到标准 Event 的薄构造函数。

## 上游和下游

上游是 CLI、摄像头、麦克风、传感器适配器。下游是 `AgentCore.handle_event` 和 `reduce_state`。

## 扩展方式

新增事件：加入 `types.py`，如需便捷构造则补充 `event_builders.py`，再在 reducer 中决定是否需要状态归约。

## 示例

```python
event = Event(type="user_text_input", timestamp=now, payload={"text": "hello"})
```
