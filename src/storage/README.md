# Storage

`storage/` 是持久化层，只负责读写本地数据，不做 LLM 推理、当前决策或设备执行。

当前文件：

- `json_store.py`：运行期 `AgentState` 快照，默认路径为 `data/runtime/runtime_store.json`。
- `long_term_memory_store.py`：已验证的 `LongTermMemory` 持久化，默认路径为 `data/memory/long_term_memory.json`。
- `user_profile_store.py`：显式 `UserProfile` 持久化，默认路径为 `data/user/user_profiles.json`。

存储层只负责持久化，不做用户认知融合；融合逻辑在 `agent/user/personal_context_builder.py`。
