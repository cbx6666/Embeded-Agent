# Decision

## 职责

`decision/` 是 Rule/LLM 输出到设备动作之间的确定性边界。

`IntentPlan -> DecisionPostProcessor -> IntentPlanValidator -> DeterministicGuard -> ActionRealizer -> Action`

## 不负责什么

本目录不做关键词语义理解，不直接读取 store，不维护旧式规划/候选/仲裁/处理器链路，不直接执行硬件。

## 核心文件

- `intent_model.py`：注册 intent 类型、AgentIntent 和 IntentPlan。
- `rule_intent_builder.py`：P0B 结构化事件的 0 LLM 计划构造。
- `autonomous_check_policy.py`：P1 的状态、趋势、source 和 cooldown 前置门控。
- `decision_post_processor.py`：Rule/LLM 共用的后处理链。
- `validator.py`：拒绝未注册 intent、action 字段和 state_patch。
- `guard.py`：执行 presence safety、cooldown、高风险阻断等硬边界。
- `action_realizer.py`：把批准的 IntentPlan 转成注册 Action。
- `decision_pipeline.py`：串联 Rule/LLM 计划来源与确定性边界。

## 上游和下游

上游是 `AgentCore` 传入的 `Event`、`AgentState`、`PersonalContext` 和 `LLMService`。内部由 `llm_agent/` 输出 `AgentRun`。下游是 `execution/device_adapter.py` 执行的 Action。

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
