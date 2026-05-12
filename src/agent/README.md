# Agent

## 职责

`agent/` 是嵌入式 Agent 的主领域包，负责把标准事件送入 LLM-centered 主链路：

`Event -> Reducer -> ProfileSnapshot -> LLMAgentOrchestrator -> IntentPlanValidator -> DeterministicGuard -> ActionRealizer -> DeviceAdapter`

## 不负责什么

本目录不直接绑定摄像头、麦克风、TTS 或灯光硬件；真实设备接入在 `adapters/`。本目录也不保留规则中心规划器、候选生成器或策略包装。

## 核心文件

- `core.py`：单事件主调度器，串联状态归约、记忆、决策和设备边界。
- `reducer.py`：只做 `Event -> AgentState` 的确定性状态归约。
- `llm_agent/`：四角色 LLM 认知链路。
- `decision/`：schema 校验、Guard 和 Action 落地。
- `memory/`：LLM-managed Memory 和 ProfileSnapshot。
- `runtime/`：运行时循环、trace 和动作结果。

## 上游和下游

上游是 CLI、传感器适配器和内部 runtime event。下游是标准 Action、ActionResult、trace 和 MemoryStore。

## 扩展方式

新增能力时优先判断它属于认知、边界、动作还是设备：认知放在 LLM 角色 prompt/schema，边界放在 validator/guard，动作放在 `action/` 和 `ActionRealizer`，设备实现放在 adapters。

## 示例

```python
actions, results = core.handle_event(event)
```
