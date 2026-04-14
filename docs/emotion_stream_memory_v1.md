# 情绪流记录与摘要机制（V1）

本文档说明情绪识别模块在 Agent 中的接入方式，目标是在不间断检测场景下避免“逐帧长期存储”带来的压力，同时给策略决策提供可用上下文。

## 1. 设计目标

- 支持连续情绪检测输入（如摄像头 + RAF-DB 分类模型）。
- 只保留短期原始样本，避免内存和持久化快速膨胀。
- 自动生成固定时间窗摘要，供状态查询和策略决策参考。
- 保持 `event -> state/memory -> policy -> action` 的职责边界清晰。

## 2. 职责边界

- `Event`：描述观察到的事实（例如 `user_emotion_updated`）。
- `State/Memory`：记录近期样本与窗口摘要。
- `Policy`：读取摘要后决定是否生成 `Action`。
- `Action`：系统执行行为（播报、显示、计时器等），不承载模型推理细节。

## 3. 当前实现

### 3.1 输入事件

情绪识别模块持续上报：

- 事件类型：`user_emotion_updated`
- 关键 payload 字段：
  - `emotion`
  - `confidence`
  - `source`
  - `person_id`（可选）
  - `model`（可选）
  - `raf_emotion` / `raf_label_id`（可选）

### 3.2 记忆结构

`MemoryState` 新增两个字段：

- `emotion_samples`：短期原始样本队列。
- `emotion_summaries`：窗口摘要队列。

### 3.3 记录与聚合逻辑

`MemoryService.record_event` 在接收到 `user_emotion_updated` 时会：

1. 把事件转为一条 `emotion_samples` 记录。
2. 按固定窗口（默认 `60s`）尝试生成摘要：
   - `dominant_emotion`
   - `distribution`
   - `avg_confidence`
   - `sample_count`
   - `start_ts` / `end_ts` / `window_sec`
3. 通过 `trim` 控制容量：
   - `emotion_samples` 默认最多 `120` 条
   - `emotion_summaries` 默认最多 `60` 条

## 4. 对外可见变化

- `/history` 输出新增：
  - `emotion_samples`
  - `emotion_summaries`
- “现在状态如何”会在回复中附带最近窗口主导情绪信息。

## 5. 推荐接入方式（情绪模块）

1. 模型侧持续推理，但不要逐帧都进 Agent 主流程。
2. 适配器层做采样/防抖后，再发 `user_emotion_updated`。
3. 由 Agent 的 `MemoryService` 完成窗口摘要与容量治理。

## 6. 后续可扩展点

- 增加“变化触发”策略（如压力连续上升才提醒）。
- 对不同 `person_id` 分人维护摘要。
- 引入日级/周级离线归档（CSV 或数据库）。
- 加入摘要质量指标（最小样本数、置信度阈值）。
