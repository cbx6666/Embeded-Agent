# Services

`services/` 放应用服务和外部能力适配，不放领域模型。

当前职责：

- `runtime_history_service.py`：维护 `RuntimeHistory` 的短期窗口和滚动统计。
- `user_profile_service.py`：管理显式 `UserProfile` 的业务入口。
- `llm_service.py`：DeepSeek Chat Completions 调用适配；生产链路不内置本地 mock。
- `timer_service.py`：计时器能力适配。

领域语义仍在 `src/agent/`：`RuntimeHistory`、`LongTermMemory`、`UserProfile` 和
`PersonalContext` 的模型与管线不在 services 中定义。

当前没有独立的 `MemoryService`：短期历史由 `RuntimeHistoryService` 维护，长期记忆由
`agent/memory/LongTermMemoryPipeline` 与 `storage/LongTermMemoryStore` 负责。
