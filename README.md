# 智能陪伴与情境交互终端 Agent MVP

本仓库当前实现的是一个可运行的 Agent 软件核心 MVP，重点是先把与硬件无关的运行时跑通，再为后续接入摄像头、麦克风、屏幕、灯光和环境传感器预留清晰边界。

## 当前 MVP 已实现什么

- CLI 文本交互
- mock 状态更新
- 专注模式启动与结束
- 倒计时结束提醒
- 简单主动提醒与冷却控制
- 最近事件、最近消息、专注记录保存
- 连续情绪样本短期记录与窗口摘要
- JSON 文件持久化与启动恢复

## 当前还没有实现什么

- 真实硬件接入
- 语音 ASR / TTS
- 摄像头视觉识别
- 传感器采集
- 真正的大模型推理
- 多 Agent 协作和复杂插件系统

## 当前架构

当前软件采用简化版分层设计 V1，只保留 5 层：

1. Input Layer
2. State Layer
3. Decision Layer
4. Action Layer
5. Storage Layer

主流程：

```text
接收输入事件
-> 更新状态
-> 判断是否触发动作
-> 生成动作
-> 执行动作
-> 记录日志 / 最近记忆
```

## 事件、状态、动作的职责分工

当前核心领域模型已经拆成三个包：

- `src/agent/state/`
  只定义 Agent 内部状态，并进一步拆成：
  - `types.py`
  - `user_state.py`
  - `interaction_state.py`
  - `focus_state.py`
  - `environment_state.py`
  - `cooldown_state.py`
  - `memory_state.py`
  - `agent_state.py`

- `src/agent/event/`
  只定义标准事件，例如：
  - `user_text_input`
  - `focus_start_requested`
  - `focus_stop_requested`
  - `user_presence_updated`
  - `user_attention_updated`
  - `user_emotion_updated`
  - `timer_ticked`
  - `timer_finished`

- `src/agent/action/`
  只定义标准动作，例如：
  - `speak`
  - `display`
  - `start_timer`
  - `stop_timer`

## adapters 的职责

`adapters/` 专门负责把真实世界输入翻译成标准事件，或者把标准动作映射到具体输出。

例如：
- CLI 输入适配器把“开始专注 25 分钟”翻译为 `focus_start_requested`
- mock 输入适配器把 `/mock emotion tired` 翻译为 `user_emotion_updated`
- 后续摄像头适配器可以把“用户离席”翻译为 `user_presence_updated`
- 后续麦克风适配器可以把语音识别结果翻译为 `user_text_input`

## 目录结构

```text
src/
  main.py
  agent/
    action/
      __init__.py
      action_model.py
      factories.py
      types.py
    event/
      __init__.py
      event_model.py
      types.py
    state/
      __init__.py
      agent_state.py
      cooldown_state.py
      environment_state.py
      focus_state.py
      interaction_state.py
      memory_state.py
      types.py
      user_state.py
    core.py
    policy.py
    reducer.py
  adapters/
    cli_input.py
    console_output.py
    mock_input.py
  services/
    llm_service.py
    memory_service.py
    timer_service.py
  storage/
    json_store.py
tests/
  test_agent_core.py
docs/
  agent_architecture_v1.txt
  commands.txt
  emotion_stream_memory_v1.md
```

## 如何运行

环境要求：
- Python 3.10+

启动：

```bash
python -m src.main
```

启动后可输入：
- `你好`
- `开始专注 25 分钟`
- `结束专注`
- `现在状态如何`
- `/mock presence present`
- `/mock attention focused`
- `/mock emotion tired`
- `/state`
- `/history`
- `/help`
- `/exit`

## 支持的命令

普通文本：
- `你好`
- `开始专注 25 分钟`
- `结束专注`
- `现在状态如何`

mock 命令：
- `/mock presence present|away|unknown`
- `/mock attention focused|distracted|idle`
- `/mock emotion neutral|tired|stressed|happy`

系统命令：
- `/state`
- `/history`
- `/help`
- `/exit`

## 测试

```bash
python -m unittest discover -s tests -v
```

## 后续如何扩展到硬件接入

- 保持 `state/`、`event/`、`action/` 作为核心领域模型稳定层
- 在 `adapters/` 中增加摄像头、麦克风、按键、传感器、屏幕、TTS、LED 等适配器
- 让新的硬件适配器继续输出标准事件、消费标准动作
- 将 `llm_service.py` 替换为真实本地或云端模型服务
- 将 `json_store.py` 替换为 SQLite 或嵌入式数据库实现
