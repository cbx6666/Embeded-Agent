# agent

`agent/` 是系统的软件核心，负责维护统一状态、处理标准事件、生成标准动作，并把整条运行链路串起来。

## 职责

- 定义 `AgentState`
- 定义标准 `Event`
- 定义标准 `Action`
- 根据事件更新状态
- 根据当前状态和规则生成动作
- 调度输出、定时器、记忆和存储服务

## 文件说明

- `state/`：状态模型包，按领域拆成多个子模块
- `event/`：标准事件模型包
- `action/`：标准动作模型包
- `reducer.py`：只负责“事件如何更新状态”
- `policy.py`：只负责“当前状态下应该触发什么动作”
- `core.py`：核心调度器，串起 reducer、policy、output、memory、storage

## 设计原则

- 状态、事件、动作分离
- adapters 只负责适配，不负责定义领域模型
- reducer 与 policy 分离
- 核心逻辑不直接依赖真实硬件
- 先保证 MVP 可运行，再逐步扩展规则复杂度

## 具体例子

- 状态例子：`state.focus.active = True`
- 事件例子：`Event(type="user_presence_updated", payload={"presence": "away"})`
- 动作例子：`Action(type="display", payload={"text": "用户已离席"})`

## 多人协作：内核与适配器

- **不写内核、只接硬件/算法的一端**：只维护 **`event`/`action` 的 types 与 factories**、**`adapters/`**、**`docs/<主题>_integration.md`**；**现阶段不要改** `reducer` / `policy` / `core` / `state` / `memory_service`，详见 **`docs/team_integration_guide.md`**。  
- **写内核**：在协作者契约稳定后，维护 `reducer` / `policy` / `core`（先 `reduce_state` 再 `decide_actions` 再 `_execute_actions`；`policy` 不直接改状态），按需扩展 `state` 与记忆，并补充测试。
