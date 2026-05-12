# Decision

## 职责

`decision/` 是 LLM 输出到设备动作之间的确定性边界。

`AgentRun -> IntentPlanValidator -> DeterministicGuard -> ActionRealizer -> Action`

## 不负责什么

本目录不做关键词语义理解，不读取策略 YAML，不维护旧式规划/候选/仲裁/处理器链路，不直接执行硬件。

## 核心文件

- `intent_model.py`：注册 intent 类型、AgentIntent 和 IntentPlan。
- `validator.py`：拒绝未注册 intent、action 字段和 state_patch。
- `guard.py`：执行 presence safety、cooldown、高风险阻断等硬边界。
- `action_realizer.py`：把批准的 IntentPlan 转成注册 Action。
- `decision_pipeline.py`：串联 LLM Agent 与确定性边界。

## 上游和下游

上游是 `llm_agent/` 输出的 AgentRun。下游是 `runtime/device_adapter.py` 执行的 Action。

## 扩展方式

新增 intent：先加入 `REGISTERED_INTENT_TYPES`，再在 `ActionRealizer` 中实现动作落地，并补 validator/guard 测试。

## 示例

```python
result = pipeline.decide(
    previous_state=previous,
    current_state=state,
    event=event,
    llm_service=llm,
)
```
