"""Runtime 包公开入口。

本包保存 Agent 主循环、动作执行结果、内部事件回流和 trace 模型。上游是
AgentCore 和 DeviceAdapter，下游是 CLI、测试和调试界面。

Runtime 不做 LLM 规划、不生成长期记忆，也不改变 IntentPlan；它只负责让事件
和动作结果在嵌入式运行时中安全流转。
"""

__all__ = [
    "action_result",
    "autonomy",
    "device_adapter",
    "internal_events",
    "loop",
    "trace",
]
