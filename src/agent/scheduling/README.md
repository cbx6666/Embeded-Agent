# Scheduling

`scheduling/` 只负责按系统时间产生低频自主检查事件：

`AutonomousScheduler -> system_triggered -> EventPriorityRouter -> AutonomousCheckPolicy`

调度间隔、启用项和轮询频率位于
`src/agent/config/policy_config.py::AutonomousScheduleConfig`。Scheduler 不直接调用
LLM，也不判断是否提醒用户。
