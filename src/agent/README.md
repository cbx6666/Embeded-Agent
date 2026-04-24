# Agent Layer

这份文档不再只讲“有哪些模块”，而是直接说明：

1. 当前 agent 到底是怎么工作的
2. 一个事件进来之后会经过哪些步骤
3. 为什么现在它已经形成了一个保守的闭环

## 一句话理解

当前 agent 层可以理解成一个两层结构：

- `AgentCore`：处理单个事件
- `AgentLoop`：把单个事件处理结果继续回流，形成一轮闭环

也就是说，现在系统不是“来一个事件，吐一串动作，就结束”。

现在是：

1. 先处理这个事件
2. 执行动作
3. 看动作执行结果
4. 必要时生成内部事件
5. 再根据内部事件判断要不要继续
6. 直到本轮结束或达到 `max_steps`

## 先看最重要的工作流

当前一轮完整工作流是：

```text
外部 Event
  -> AgentLoop.run_once(event)
    -> AgentCore.handle_event_with_results(event)
      -> reducer.reduce_state(...)
      -> memory_service.record_event(...)
      -> planner.plan_intents(...)
      -> realizer.realize_actions(...)
      -> AgentCore._execute_actions(...)
      -> 产出 ActionResult[]
    -> internal_events.build_internal_events_from_results(...)
    -> 如果有内部 system_triggered Event，继续送回 AgentCore
    -> 最多循环 max_steps 次
  -> 返回本轮所有 Action
```

如果只看单次处理链路，也就是 `AgentCore` 内部，是：

```text
Event
  -> 更新 State
  -> 记录 Memory
  -> 规划 Intent
  -> 生成 Action
  -> 执行 Action
  -> 得到 ActionResult
```

## 各模块在工作流里的位置

### `reducer.py`

只负责一件事：

“这个事件发生后，状态应该怎么变。”

它不决定是否说话，不决定是否提醒，也不决定是否调用 LLM。

例如：

- `focus_start_requested`：把 `focus.active` 设为 `True`
- `focus_stop_requested`：结束 focus，会写入一条 focus session
- `user_attention_updated`：更新 `state.user.attention`
- `tts_started`：把 `state.interaction.dialogue_state` 设为 `speaking`

所以 reducer 是纯状态层。

### `planner.py`

只负责一件事：

“看到当前事件和当前状态后，系统打算做什么。”

planner 的输出不是 `Action`，而是 `Intent`。

这是故意的，因为这里要先解决“决策”，而不是马上去控制设备。

planner 现在会判断这些问题：

- 这个事件要不要响应
- 是用户主动事件，还是系统内部事件
- 应该回复用户、开始专注、结束专注，还是什么都不做
- 这次响应是不是需要 LLM
- 这次是不是主动提醒
- 提醒是不是在 cooldown 里

### `realizer.py`

只负责一件事：

“把 `Intent` 变成真正可执行的 `Action`。”

例如：

- `answer_user` -> `speak` + `display`
- `start_focus` -> `start_timer` + `speak/display` + `set_light_state`
- `suggest_rest` -> 提醒动作
- `adjust_environment_feedback` -> 低打扰环境反馈

realizer 这一层才真正关心：

- 要不要 `speak`
- 要不要 `display`
- 要不要 `set_light_state`
- 当前是不是 `silent`
- 当前是不是已经在 `speaking/listening`

### `core.py`

`AgentCore` 把上面几层串起来。

它不负责多轮闭环，只负责“处理一个事件”。

它现在的两个主要接口是：

- `handle_event(event) -> list[Action]`
- `handle_event_with_results(event) -> tuple[list[Action], list[ActionResult]]`

第二个接口是这次闭环能力的关键，因为闭环要看动作执行结果。

### `internal_events.py`

这层负责把动作执行结果转成内部事件。

也就是说，系统现在会“观察自己刚刚做了什么，以及做得成不成功”。

当前支持的回流规则很保守：

- `speak` / `display` 成功 -> `agent_response_completed`
- `start_timer` 成功 -> `focus_timer_started`
- `stop_timer` 成功 -> `focus_timer_stopped`
- 任意动作失败 -> `action_failed`
- 没有动作，或只有 `none` -> 不生成内部事件

### `loop.py`

`AgentLoop` 是闭环管理器。

它做的事情不是“重新决策”，而是“让多个事件串成一轮 decision cycle”。

它的逻辑可以直接理解成：

```python
queue = [外部事件]
while queue 非空 and step < max_steps:
    current_event = queue.pop(0)
    actions, results = core.handle_event_with_results(current_event)
    internal_events = build_internal_events_from_results(...)
    queue.extend(internal_events)
```

所以 `AgentLoop` 不是替代 `AgentCore`，而是在它外面加了一个事件回流层。

## 系统到底怎么“闭环”

闭环不是指 agent 会无目的地一直行动。

闭环的意思是：

“动作执行完以后，系统会根据执行结果再判断一次，而不是直接结束。”

举个实际例子。

### 例子 1：用户说“开始专注 25 分钟”

输入事件：

```python
Event(
    type="focus_start_requested",
    timestamp=1000,
    payload={"duration_sec": 1500, "source": "user"}
)
```

实际经过的步骤：

1. `AgentLoop.run_once(...)` 收到这个事件
2. `AgentCore.handle_event_with_results(...)` 开始处理
3. `reducer` 更新状态：
   - `focus.active = True`
   - `focus.target_duration_sec = 1500`
   - `interaction.mode = "focus"`
4. `memory_service` 记录这次事件
5. `planner` 看到这是 `focus_start_requested`
   - 生成 `Intent(type="start_focus")`
6. `realizer` 把它转成动作：
   - `start_timer`
   - `speak("已开始 25 分钟专注。")`
   - `display("已开始 25 分钟专注。")`
   - `set_light_state("thinking")`
7. `AgentCore._execute_actions(...)` 执行动作
8. 每个动作生成 `ActionResult`
9. `internal_events.py` 根据结果生成内部事件：
   - `focus_timer_started`
   - `agent_response_completed`
10. `AgentLoop` 把这两个内部事件继续送回 `AgentCore`
11. `planner` 对这些内部事件通常返回 `no_op`
12. 队列空了，这一轮结束

这里的闭环体现在：

- 系统不是只“发出动作”
- 而是会“看到动作结果，再判断一次”

### 例子 2：系统主动做 focus 健康检查

输入事件不是用户输入，而是系统自己构造的：

```python
build_autonomous_check_event(state, now_ts=6000, reason="focus_health_check")
```

这个事件进入系统后：

1. `reducer` 基本不改状态，因为它只是一个内部触发
2. `planner` 看到 `system_triggered + trigger=focus_health_check`
3. planner 读取当前状态：
   - 用户是否在场
   - 当前是不是 focus 模式
   - 疲劳是不是 `moderate/high`
   - 注意力是不是 `distracted`
   - 当前是不是在 cooldown 里
4. 如果条件满足，planner 会输出：
   - `suggest_rest`
   - 或 `remind_distraction`
5. `realizer` 决定动作形式
   - away 时不 `speak`
   - silent 时不 `speak`
   - 正在 speaking/listening 时不重复 `speak`
   - 优先用规则模板，不调 LLM

也就是说，自主检查并不是“让 LLM 随便想一句话”。

它现在更像一个规则驱动的守门员：

- 只有条件满足才提醒
- 冷却中不重复提醒
- 用户不在场时不主动打扰

### 例子 3：动作执行失败

假设某个动作失败了，比如：

```python
ActionResult(
    action_type="start_timer",
    success=False,
    timestamp=4000,
    reason="mock failure"
)
```

系统会做什么：

1. `internal_events.py` 把它转成：

```python
Event(
    type="system_triggered",
    payload={
        "trigger": "action_failed",
        "action_type": "start_timer",
        "reason": "mock failure",
    }
)
```

2. `AgentLoop` 把这个内部事件送回 `AgentCore`
3. `planner` 看到 `trigger == "action_failed"`
4. planner 生成一个规则型 `answer_user`
5. `realizer` 输出错误反馈动作

所以现在 agent 有最基本的“失败后回流处理”能力，而不是失败了就默默结束。

## LLM 在整个流程里的真实位置

这个点非常重要。

当前 LLM 不在流程前面，也不在流程最核心的位置。

LLM 的位置是：

- planner 已经决定“这是一个需要文本回复的场景”
- realizer 在生成 `answer_user` 的具体文本时
- 如果 `Intent.requires_llm=True`，才调用 LLM

所以真实流程不是：

```text
事件 -> LLM -> 动作
```

而是：

```text
事件 -> 状态更新 -> 规则决策 -> 需要时调用 LLM 生成回复文本 -> 动作
```

这意味着：

- LLM 不直接改状态
- LLM 不直接决定硬件动作
- LLM 不参与 reducer
- LLM 不参与 cooldown 判定
- 自主检查默认不调用 LLM

当前只有“普通用户输入的对话回复”这类场景，LLM 才更可能被调用。

## 为什么要拆成 Event / Intent / Action 三层

因为三者解决的是不同问题。

### Event

表示“发生了什么事实”。

例如：

- 用户说了一句话
- timer 到点了
- 检测到疲劳
- 动作执行失败了

### Intent

表示“系统想做什么”。

例如：

- 回复用户
- 开始专注
- 提醒休息
- 提醒回到任务

### Action

表示“系统具体怎么做”。

例如：

- `speak`
- `display`
- `start_timer`
- `stop_timer`
- `set_light_state`

这样拆开的好处是：

- planner 只做决策，不碰设备
- realizer 只做动作生成，不改状态
- LLM 只生成文本，不直接控制硬件

## 当前有哪些明确的边界规则

这些规则决定了 agent 虽然能自主，但不会太吵、太乱。

### 1. 用户 `away`

- 系统主动提醒不 `speak`
- 可以 `display`
- 不做高打扰休息提醒

### 2. `silent` 模式

- 不 `speak`
- 可以 `display`
- 可以 `set_light_state`

### 3. 正在 `speaking` 或 `listening`

- 不重复 `speak`
- 有提示也优先降级成 `display`

### 4. `focus` 模式

- 用户主动输入仍然会回复
- 系统主动提醒必须经过 cooldown
- 自主提醒更克制

### 5. 自主事件默认不用 LLM

例如：

- `periodic_check`
- `focus_health_check`
- `environment_check`
- `action_failed`

这些默认都走规则模板。

## 为什么现在说它“已经闭环”

因为它已经满足这几个条件：

### 1. 能处理外部事件

比如用户输入、timer 事件、状态更新事件。

### 2. 能根据状态和规则做决策

不是简单写死一个动作，而是会结合：

- focus 状态
- fatigue
- attention
- presence
- cooldown
- dialogue_state

### 3. 能执行动作并得到结果

动作执行后会得到 `ActionResult`。

### 4. 能把结果回流成内部事件

成功和失败都可以变成内部 `system_triggered` 事件。

### 5. 能根据内部事件再判断一次

这一步由 `AgentLoop` 完成。

### 6. 有边界防止无限循环

`max_steps` 会强制截断。

所以它不是“无限自主 agent”，而是“有边界、有回流、有停止条件的保守闭环 agent”。

## 现在最应该从哪里读代码

如果你想顺着一次流程看代码，推荐按这个顺序读：

1. [loop.py](/d:/Homework/embed/project/src/agent/loop.py:1)
   看一轮闭环怎么调度
2. [core.py](/d:/Homework/embed/project/src/agent/core.py:1)
   看单个事件怎么被处理
3. [reducer.py](/d:/Homework/embed/project/src/agent/reducer.py:1)
   看事件怎么改状态
4. [planner.py](/d:/Homework/embed/project/src/agent/planner.py:1)
   看决策规则
5. [realizer.py](/d:/Homework/embed/project/src/agent/realizer.py:1)
   看动作怎么生成
6. [internal_events.py](/d:/Homework/embed/project/src/agent/internal_events.py:1)
   看结果怎么回流
7. [autonomy.py](/d:/Homework/embed/project/src/agent/autonomy.py:1)
   看自主检查事件入口
8. [trace.py](/d:/Homework/embed/project/src/agent/trace.py:1)
   看调试信息记录

## 测试对应了哪些工作流

当前关键测试在：

- [test_agent_core.py](/d:/Homework/embed/project/tests/test_agent_core.py:1)
- [test_agent_loop.py](/d:/Homework/embed/project/tests/test_agent_loop.py:1)

它们覆盖的不是抽象概念，而是具体工作流：

- 用户文本输入 -> speak/display
- focus_start -> start_timer -> 内部事件回流 -> 闭环结束
- focus_health_check -> suggest_rest
- cooldown 内不重复提醒
- silent 模式不 speak
- speaking 状态不重复 speak
- action_failed -> 错误反馈
- max_steps 阻止无限回流

## 总结

现在的 agent 工作方式可以概括成一句更具体的话：

它先用 `reducer` 把“发生了什么”写进状态，再由 `planner` 判断“系统现在想做什么”，再由 `realizer` 把这个意图落成动作；动作执行后会产出 `ActionResult`，`AgentLoop` 再根据结果决定要不要生成内部事件继续处理，因此它已经从“单向响应器”变成了一个“有状态、有反馈回流、有停止条件的闭环决策器”。
