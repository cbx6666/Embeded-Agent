"""RuntimeHistory 子系统公开入口。

为避免 AgentState 初始化时形成循环导入，本包入口只导出纯数据模型；
运行期历史服务位于 `src.services.runtime_history_service`。
"""
from src.agent.history.runtime_history import RuntimeHistory

__all__ = ["RuntimeHistory"]
