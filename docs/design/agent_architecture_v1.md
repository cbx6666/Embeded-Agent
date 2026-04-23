# Agent 软件架构设计 V1

## 1. 概述

本文说明智能陪伴与情境交互终端 Agent 的软件架构。当前版本目标是在未接入真实硬件前，先完成一个可运行、可测试、可扩展的软件核心 MVP，用于验证 Agent Runtime 的基础闭环。

## 2. 架构目标

- 先跑通 CLI / mock 输入、状态更新、策略决策、动作执行和持久化。
- 保持核心领域模型稳定，后续硬件只通过 Adapter 接入。
- 让团队可以并行开发视觉、语音、显示、环境传感器等模块。
- 避免在内核中耦合摄像头、麦克风、屏幕、TTS、传感器等具体实现。

## 3. 分层设计

| 层级 | 职责 | 当前实现 |
|------|------|----------|
| Input Layer | 接收外部输入并翻译为标准 Event。 | CLI 文本输入、mock 状态输入。 |
| State Layer | 维护统一状态模型。 | `src/agent/state/` 下的用户、交互、专注、环境、冷却和记忆状态。 |
| Decision Layer | 根据 Event 和 AgentState 生成 Action。 | `reducer.py` 与 `policy.py`。 |
| Action Layer | 执行标准 Action。 | 控制台 `speak` / `display`、专注计时器启动/停止。 |
| Storage Layer | 持久化状态快照与最近记忆。 | JSON 文件存储。 |

## 4. 核心模型

| 模型 | 路径 | 职责 |
|------|------|------|
| State | `src/agent/state/` | 定义 Agent 内部状态。 |
| Event | `src/agent/event/` | 定义外部输入或系统内部触发的标准事件。 |
| Action | `src/agent/action/` | 定义内核输出给执行层的标准动作。 |

当前主要事件包括：

- `user_text_input`
- `speech_recognized`
- `focus_start_requested`
- `focus_stop_requested`
- `user_presence_updated`
- `user_attention_updated`
- `user_emotion_updated`
- `light_level_updated`
- `temperature_humidity_updated`
- `noise_level_updated`
- `timer_ticked`
- `timer_finished`

当前主要动作包括：

- `speak`
- `display`
- `set_light_state`
- `start_timer`
- `stop_timer`
- `none`

## 5. Adapter 职责

`adapters` 层不定义核心领域模型，只负责两类转换：

- 把真实世界输入转换为标准 Event。
- 把标准 Action 映射为具体输出。

示例：

- CLI 适配器将“开始专注 25 分钟”翻译为 `focus_start_requested`。
- mock 适配器将 `/mock emotion tired` 翻译为 `user_emotion_updated`。
- 摄像头适配器可以将“用户离席”翻译为 `user_presence_updated`。
- 麦克风适配器可以将语音识别结果翻译为 `speech_recognized`。

## 6. 主运行流程

```text
接收输入
-> Adapter 翻译为标准 Event
-> reducer 更新 AgentState
-> policy 生成 Action
-> output / timer 执行动作
-> memory 维护最近记录
-> storage 持久化
```

## 7. 当前 MVP 范围

- 用户文本输入后，Agent 必须返回一条响应。
- 用户可通过“开始专注 X 分钟”进入专注模式。
- 用户可通过“结束专注”停止专注模式。
- 倒计时到期后自动结束专注并输出提醒。
- mock 命令可更新 presence、attention、emotion、fatigue。
- 当用户 focused 且 tired 且已经专注一段时间时，可触发休息提醒。
- 同类提醒具备基础冷却机制。
- 最近事件、最近消息和专注记录可保存在状态中并落盘。

## 8. 后续扩展

- 扩展事件模型，接入视觉、语音、按键和环境传感器。
- 扩展动作模型，接入屏幕渲染、TTS 播报、LED 控制等输出。
- 接入真实 LLM，替换当前轻量回复服务。
- 将 JSON 存储升级为 SQLite 或其他嵌入式数据库。
