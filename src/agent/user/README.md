# User

`agent/user/` 是用户认知层。

它包含两类边界不同、但都围绕用户建模的对象：

- `UserProfile`：用户明确声明或系统明确配置的静态权威资料，例如姓名、年龄、显式偏好和展示设置。
- `PersonalContext`：面向决策层的动态用户上下文快照，由 `UserProfile`、`LongTermMemory` 和 `RuntimeHistory` 组合生成。

`UserProfileService` 位于 `src/services/`，`UserProfileStore` 位于 `src/storage/`。
`agent/user/` 只保留用户认知层的模型和上下文构建器。

`PersonalContextBuilder` 只读各来源并做冲突处理、优先级融合和 prompt 压缩，不直接写入 store。
