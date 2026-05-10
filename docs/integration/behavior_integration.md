# 行为识别模块

## Event

| Event.type | 语义 | 边界 |
|------------|------|------|
| `user_presence_updated` | 用户是否在场、离席或未知。 | 只上报在场事实。 |
| `user_attention_updated` | 用户当前是否专注、分心或空闲。 | 只上报注意力事实；`behavior` 是正式字段，不是附属字段。 |

建议的闭集：

- `presence`: `present` / `away` / `unknown`
- `attention`: `focused` / `distracted` / `idle`
- `behavior`: `working` / `phone_use` / `staring` / `desk_rest` / `away`

## Action

本模块当前无专用 Action。

说明：

- 行为模块只报告“看到了什么”。
- 如果用户分心、离席、长时间异常，仍然只发 Event。
- 是否提醒、如何提醒，由内核再转成显示、语音、灯光模块的能力 Action。

## Adapter

| Adapter | 责任 |
|---------|------|
| `BehaviorAdapter` | 将摄像头、关键点、行为分类模型或规则结果转换成 `user_presence_updated` 与 `user_attention_updated`。 |
