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
- `Event` 表示模块上报的事实或执行结果
- `Action` 表示内核下发的能力命令
- adapters 只负责适配，不负责定义领域模型
- reducer 与 policy 分离
- 核心逻辑不直接依赖真实硬件
- 先保证 MVP 可运行，再逐步扩展规则复杂度

## 具体例子

- 状态例子：`state.focus.active = True`
- 事件例子：`Event(type="speech_recognized", payload={"text": "开始专注 25 分钟"})`
- 动作例子：`Action(type="display", payload={"text": "专注倒计时已启动", "kind": "notification"})`

## 多人协作：内核与适配器

- 输入 / 输出侧协作者：优先维护 `event` / `action` 的 types、factories 和 `adapters/`，不要把业务策略写进适配层。
- 内核协作者：维护 `state` / `reducer` / `policy` / `core`，保证“先归约状态，再决策动作，再执行动作”的流程稳定。
