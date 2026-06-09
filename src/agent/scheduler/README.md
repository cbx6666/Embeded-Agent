# Scheduler

## 职责

`scheduler/autonomous_scheduler.py` 是多任务周期调度器，按各任务独立周期与优先级产生
`system_triggered` 事件：

```text
behavior_distraction_check : interval=20s,  priority=1
wellness_care_check        : interval=30s,  priority=2
environment_care_check     : interval=60s,  priority=3
sensor_status_report       : interval=300s, priority=4
（speech_recognized        : priority=0，外部事件，不由调度器产生）
```

每个任务持有 `remaining_sec` / `due` / `priority`，调度器按真实流逝时间递减倒计时。

## 防打断与剩余时间保留

- 每个 tick 最多发出一个事件：在可运行的 due 任务中选优先级最高（priority 数字最小）。
- 当更高优先级任务正在运行（`busy_priority_provider` 返回更小优先级数）时，低优先级任务
  倒计时**冻结**（不递减、不发出、不重置），高优先级结束后从原处继续。
- 已到点（`due`）但被抢占的任务保持 `due`，等高优先级结束后立即执行（pending_due）。
- 任务真正发出后，`remaining_sec` 从完整周期重新开始。
- 高频 state_only 事件不经过调度器，不会占用或阻塞它。

## 不负责

- 调用 LLM、生成 Intent / Action
- 中断正在进行的 LLM（只保证后续 speak 不打断语音，交给 Guard）
- 旧 trigger（`periodic_state_check`、`wellness_check`、`environment_check` 等，已删除）

`run_due()` 可供测试同步触发，无需后台线程；`task_status()` 输出各任务状态便于调试。
