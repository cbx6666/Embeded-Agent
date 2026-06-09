# Storage

`storage/` 是持久化层，只负责读写本地数据，不做 LLM 推理、当前决策或设备执行。

当前文件：

- `json_store.py`：运行期 `AgentState` 快照，默认路径为 `data/runtime/runtime_store.json`。
- `user_profile_store.py`：显式 `UserProfile` 持久化，默认路径为 `data/user/user_profiles.json`。

偏好记忆不在本层：异步偏好记忆由 `agent/memory/memory_service.py` 自带 JSON 落地，
默认路径为 `data/memory/preferences.json`。
