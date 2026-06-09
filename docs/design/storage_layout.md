# Storage Layout

## 1. Data Domain

当前 `data/` 按数据域划分本地文件布局：

```text
data/
  runtime/
    runtime_store.json
    state_stats.db

  memory/
    preferences.json

  user/
    user_profiles.json

  experiments/
    lab/
    runtime/
    retrieval/
    replay/
    traces/
```

| Domain | 内容 | 生命周期 | 是否属于 production runtime |
| --- | --- | --- | --- |
| `runtime/` | 当前 Agent 运行状态快照 | 可覆盖、随当前运行演进 | 是 |
| `memory/` | 用户偏好记忆（`MemoryService`） | 长期保留、按用户分桶 | 是 |
| `user/` | 显式用户资料和偏好（`UserProfileStore`） | 长期保留、权威可编辑 | 是 |
| `experiments/` | 实验和调试输出 | 实验性、可删除再生成 | 否 |

## 2. `runtime/runtime_store.json`

`data/runtime/runtime_store.json` 是 `JsonStore` 保存的 `AgentState` runtime snapshot。

它是：

- 当前 Agent runtime 状态快照
- `AgentState.to_dict()` 的持久化结果
- 包含 `user`、`interaction`、`focus`、`environment`、`cooldown`、`runtime_history` 和 `current_user_id`

它不是：

- 完整历史数据库
- 偏好记忆文件
- 用户显式 profile
- replay event log

`RuntimeHistory` 是 `AgentState` 的子结构，随 `runtime_store.json` 一起保存。

## 3. `memory/preferences.json`

`data/memory/preferences.json` 是 `MemoryService` 的持久化文件。

它保存从 `speech_recognized` 中确定性提取的用户偏好文本，典型结构：

```json
{
  "default": [
    {"content": "以后请不要频繁提醒我", "timestamp": 1710000000}
  ]
}
```

特点：

- 由 `MemoryService` 异步写入，不阻塞 `AgentCore.handle_event`
- 命中偏好标记词且非琐碎文本时才记录
- 供两个 LLM 入口在 prompt 前检索（`retrieve_preferences`）

它不是：

- LLM 记忆抽取/评审/巩固管线的输出（已删除）
- 带 evidence/confidence/decay 的长期记忆图谱

## 4. `user/user_profiles.json`

`data/user/user_profiles.json` 是 `UserProfileStore` 的持久化文件。

它保存 authoritative explicit user profile，包括：

- 用户 ID、display name
- 年龄、性别、身份、爱好等显式资料
- reminder style、TTS 设置等显式偏好

显式 profile 的业务入口是 `UserProfileService`。决策时 `AgentCore._preferences()` 会把 profile 偏好与 `MemoryService` 检索结果合并进 LLM prompt。

## 5. `experiments/`

`data/experiments/` 保存实验和调试输出，不属于 production runtime state。

| 目录 | 内容 |
| --- | --- |
| `experiments/runtime/` | 长期运行实验输出（部分脚本仍引用旧 `long_term_memory.json` 路径，属历史实验产物） |
| `experiments/retrieval/` | retrieval quality experiment 输出 |
| `experiments/replay/` | debug replay 输出 |
| `experiments/traces/` | 手动导出的 trace dump |

这些文件用于评估和调试，不应被当作当前生产 runtime truth。

## 6. 当前不拆 `runtime_store` 的原因

- `RuntimeHistory` 仍是 reducer、Guard、summary_builder 和 LLM prompt 的短期上下文来源
- 拆成多个 state 文件会引入同步、恢复顺序和迁移问题
- 单文件 `AgentState` snapshot 降低风险，继续支持现有 `JsonStore` 协议
