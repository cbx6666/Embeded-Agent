# 显示输出模块

## Action（当前闭集）

| Action.type | 语义 | 边界 |
|-------------|------|------|
| `display` | 显示文本、状态、结果或界面。 | 显示模块只执行渲染，不理解用户命令。提醒语义放在 payload 的 `kind` / `reason` 中。 |

已删除且不再由 Agent 生成：`render_pet_expression`、`set_light_state`。

建议的 `display.payload` 字段：

- `text`
- `title`
- `status`
- `kind`
- `reason`

## Adapter（当前实现）

| 模块 | 责任 |
|------|------|
| `ConsoleOutput` | 默认开发输出：将 `display` / `speak` 打印到终端或日志 |
| `ScreenDisplayAdapter` + `screen_window.py` | 有 DISPLAY 时驱动窗口桌宠 |
| `HeadlessPetDisplay` | 无 DISPLAY 时渲染 PNG 并推送到 `PetPreviewServer` |
| `src/pc_display/` | 独立 PC USB 显示 demo，不在主 Agent 执行链路 |

主链路：`ActionRealizer` 生成 `display` → `DeviceAdapter.output.execute()` → 上述适配器之一。

桌宠表情（idle / listening / thinking / speaking / focus_mode）由 `ScreenDisplayAdapter.sync_visual_state()` 驱动：
每个事件处理完成后，`main.py` 将 `AgentState.interaction.dialogue_state` 与专注计时同步到 pygame 帧。
`display` 的 `kind=notification` 只更新顶部状态文案，不改变脸型；播报中脸型由 `tts_started` → `dialogue_state=speaking` 切换。

## 决策链路

是否显示、显示什么，由以下入口决定（经对应 Handler + `ActionRealizer`）：

- 用户语音：`speech_recognized` → `SpeechLlmHandler`
- 自主关怀：`behavior_distraction_check` / `wellness_care_check` / `environment_care_check`（各独立周期与 prompt）
- 传感器播报：`sensor_status_report`（确定性规则，不调 LLM）

`display_sensor_updated` 事件已从主链删除；显示侧传感器若需接入，应评估是否有 reducer 闭环后再单独设计。
