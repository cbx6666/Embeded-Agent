# Storage Layout

## 1. Data Domain

当前 `data/` 只做本地文件布局整理，不改变任何 store schema、runtime pipeline 或 replay 协议。

目标目录按数据域划分：

```text
data/
  runtime/
    runtime_store.json
    state_stats.db

  memory/
    long_term_memory.json

  user/
    user_profiles.json

  experiments/
    lab/
      runtime_store.json
      long_term_memory.json
      user_profiles.json

    runtime/
      <experiment_name>/

    retrieval/
      retrieval_quality/

    replay/
      <replay_run_name>/

    traces/
      <optional trace dumps>
```

| Domain | 内容 | 生命周期 | 是否属于 production runtime |
| --- | --- | --- | --- |
| `runtime/` | 当前 Agent 运行状态快照和运行期统计文件 | 可覆盖、随当前运行演进 | 是 |
| `memory/` | 长期记忆持久化文件 | 长期保留、可衰减、可冲突处理 | 是 |
| `user/` | 显式用户资料和偏好 | 长期保留、权威可编辑 | 是 |
| `experiments/` | lab、runtime experiment、retrieval、replay、trace、metrics、report 输出 | 实验性、可删除再生成 | 否 |

## 2. `runtime/runtime_store.json`

`data/runtime/runtime_store.json` 是 `JsonStore` 保存的 `AgentState` runtime snapshot。

它是：

- 当前 Agent runtime 状态快照。
- `AgentState.to_dict()` 的持久化结果。
- 包含 `user`、`interaction`、`focus`、`environment`、`cooldown`、`runtime_history` 和 `current_user_id`。

它不是：

- 完整历史数据库。
- 长期记忆。
- 用户显式 profile。
- replay event log。

`RuntimeHistory` 目前仍是 `AgentState` 的子结构，因此会随 `runtime_store.json` 一起保存。

## 3. `memory/long_term_memory.json`

`data/memory/long_term_memory.json` 是 `LongTermMemoryStore` 的持久化文件。

它保存已经通过 `MemoryValidator` 的 `LongTermMemory`，典型字段包括：

- `memory_type`
- `content`
- `confidence`
- `evidence`
- `metadata`
- `decay`
- `status`
- `contradiction_of`

特点：

- 长期保存。
- 带证据链。
- 带 confidence。
- 支持 contradiction 标记。
- 支持 decay。
- 由 `LongTermMemoryPipeline` 写入。
- 由 `PersonalContextBuilder` 读取。

它不是 LLM 直接生成的自由文本仓库。LLM 只能提出 `MemoryCandidate`，不能直接写入该文件。

## 4. `user/user_profiles.json`

`data/user/user_profiles.json` 是 `UserProfileStore` 的持久化文件。

它保存 authoritative explicit user profile，包括：

- 用户 ID。
- display name。
- 年龄、性别、身份、爱好等显式资料。
- reminder style、speech style、TTS 设置、disliked topics 等显式偏好。

它不是：

- LLM-generated memory。
- 行为推断结果。
- runtime history。

显式 profile 的业务入口是 `UserProfileService`。如果 LLM 从对话中观察到偏好，只能进入长期记忆候选链路，不能直接写 `UserProfileStore`。

## 5. `experiments/`

`data/experiments/` 保存实验和调试输出，不属于 production runtime state。

当前约定：

| 目录 | 内容 |
| --- | --- |
| `experiments/lab/` | `src/agent_lab.py` 开发验证终端使用的 runtime/memory/profile 文件 |
| `experiments/runtime/` | `study_session`、`long_term_memory`、`multi_user_isolation`、`hallucination_resistance` 等长期运行实验输出 |
| `experiments/retrieval/` | retrieval quality experiment 输出 |
| `experiments/replay/` | debug replay 输出 |
| `experiments/traces/` | 手动导出的 trace dump 或独立 trace 分析文件 |

实验输出可能包含：

- `events.json`
- `action_timeline.json`
- `trace_logs.json`
- `memory_snapshots.json`
- `personalization_snapshots.json`
- `metrics.json`
- `report.md`
- retrieval cases/results/score breakdown
- replay summary

这些文件用于评估和调试，不应被当作当前生产 runtime truth。

## 6. 为什么现在不拆 `runtime_state` / `runtime_history`

当前逻辑分层已经存在：

- `AgentState` 表示当前运行状态。
- `RuntimeHistory` 是 `AgentState` 的短期运行窗口子结构。
- `LongTermMemory` 独立存放在 `data/memory/long_term_memory.json`。
- `UserProfile` 独立存放在 `data/user/user_profiles.json`。

因此，当前阶段只整理物理目录，不拆 `runtime_store.json` 内部结构。

不拆的原因：

- `RuntimeHistory` 仍然是 reducer、guard、PersonalContextBuilder 和 runtime trace 的短期上下文来源。
- 拆成多个 state 文件会引入同步、恢复顺序和迁移问题。
- 当前目标是 storage layout cleanup，不是 storage system redesign。
- 保持单文件 `AgentState` snapshot 可以降低风险，并继续支持现有 `JsonStore` 协议。

后续如果 runtime history 的体积、查询需求或保留周期发生明显变化，再考虑独立物理文件或更专门的历史存储。

