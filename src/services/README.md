# services

`services/` 放运行时公共服务，这些模块不直接承担核心业务决策，但会被核心调度器调用。

## 职责

- 提供专注倒计时能力
- 维护最近事件、最近消息和专注记录
- 提供一个可替换的大模型服务接口

## 文件说明

- `timer_service.py`：管理专注模式倒计时和 tick 回调
- `memory_service.py`：维护 recent events、recent messages、focus sessions
- `llm_service.py`：当前为模板回复实现，后续可替换为真实模型服务

## 设计原则

- 服务层只提供能力，不直接决定业务状态流转
- 尽量保持无状态或低耦合
- 保持可替换性，例如将模板回复替换成真实 LLM 时，不影响上层接口

## 后续扩展方向

- 将 `timer_service` 扩展为更稳定的调度器
- 将 `memory_service` 扩展为统计与摘要服务
- 将 `llm_service` 接入端侧或云端模型
