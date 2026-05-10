# Agent

`src/agent/` 负责系统的事件驱动决策闭环。

## 职责

- 定义标准 `Event`、`State`、`Intent`、`Action`
- 根据事件归约状态
- 基于规则和受约束的 LLM 辅助生成意图
- 将意图稳定转换为动作
- 执行动作并将结果回流为内部事件

## 工作流

当前处理链路如下：

```text
Event
-> reducer 更新 State
-> planner 生成候选 Intent
-> （可选）LLM 辅助选择 Intent
-> IntentGuard 校验 Intent
-> realizer 生成 Action
-> AgentCore 执行 Action
-> ActionResult 回流为 system_triggered Event
-> AgentLoop 判断是否继续
```

其中：

- `AgentCore` 负责单事件处理
- `AgentLoop` 负责一轮闭环调度
- LLM 仅参与意图判断和文本回复，不直接修改状态，也不直接生成动作

## 目录

- `action/`：动作模型
- `event/`：事件模型
- `state/`：状态模型
- `decision/`：决策层，包含 `intent / planner / llm_intent_planner / intent_guard / policy / realizer`
- `runtime/`：运行层，包含 `action_result / internal_events / autonomy / loop / trace`
- `core.py`：核心调度器
- `reducer.py`：状态归约

## 验证终端

项目提供独立的自主化验证终端，可用于手动注入事件、触发自主检查、运行内置场景并查看 trace：

```bash
python -m src.agent_lab
```

## 设计约束

- 规则层负责边界与候选意图
- LLM 仅在允许范围内辅助选择意图
- Intent 必须经过统一校验
- 动作执行结果必须可回流、可追踪
- 自主行为默认低打扰，并受 cooldown、presence、mode 等条件约束
