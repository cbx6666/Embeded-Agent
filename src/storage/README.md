# Storage

`storage/` 是持久化层，只负责读写本地数据，不做 LLM 推理和业务决策。

当前文件：

- `json_store.py`：运行期 `AgentState` 快照。
- `long_term_memory_store.py`：已验证的 `LongTermMemory` 持久化。
- `user_profile_store.py`：显式 `UserProfile` 持久化。

上层逻辑只依赖读写接口，不关心底层介质；后续可以替换为 SQLite 或其它嵌入式存储。
