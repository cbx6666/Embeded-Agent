from src.agent.action.action_model import Action


# 下面这些辅助函数用于构造常见动作，避免上层直接拼装字典。
def speak(text: str, kind: str | None = None) -> Action:
    """构造一条播报类动作。"""
    payload = {"text": text}
    if kind:
        payload["kind"] = kind
    return Action(type="speak", payload=payload)



def display(text: str, kind: str | None = None) -> Action:
    """构造一条显示类动作。"""
    payload = {"text": text}
    if kind:
        payload["kind"] = kind
    return Action(type="display", payload=payload)



def start_timer(duration_sec: int) -> Action:
    """构造一条启动定时器动作。"""
    return Action(type="start_timer", payload={"duration_sec": duration_sec})



def stop_timer() -> Action:
    """构造一条停止定时器动作。"""
    return Action(type="stop_timer", payload={})



def none_action() -> Action:
    """构造一条空动作。"""
    return Action(type="none", payload={})
