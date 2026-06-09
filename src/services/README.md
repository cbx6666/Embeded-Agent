# Services

`services/` 放应用服务和外部能力适配，不放领域模型。

当前职责：

- `user_profile_service.py`：管理显式 `UserProfile` 的业务入口。
- `user_profile_model.py`：`UserInfo` / `UserPreference` / `UserProfile` 显式画像数据模型。
- `llm_service.py`：DeepSeek Chat Completions 调用适配；生产链路不内置本地 mock。
- `timer_service.py`：计时器能力适配。

记忆与短期历史已收敛到 `src/agent/`：

- 短期窗口由 `agent/state/runtime_history.py` 的 `RuntimeHistoryService` 维护。
- 异步偏好记忆由 `agent/memory/memory_service.py` 的 `MemoryService` 负责
  （JSON 落地，服务两个 LLM 入口）。
