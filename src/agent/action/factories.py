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



def render_pet_expression(
    expression: str,
    *,
    style: str | None = None,
    intensity: float | None = None,
    duration_ms: int | None = None,
    sensor_hint: dict[str, object] | None = None,
) -> Action:
    """构造桌宠显示屏表情动作。"""
    payload: dict[str, object] = {"expression": expression}
    if style:
        payload["style"] = style
    if intensity is not None:
        payload["intensity"] = max(0.0, min(1.0, float(intensity)))
    if duration_ms is not None:
        payload["duration_ms"] = max(0, int(duration_ms))
    if sensor_hint:
        payload["sensor_hint"] = sensor_hint
    return Action(type="render_pet_expression", payload=payload)



def play_voice(
    *,
    text: str,
    voice: str | None = None,
    emotion: str | None = None,
    interrupt: bool = False,
    volume: int | None = None,
) -> Action:
    """构造语音播报动作。"""
    payload: dict[str, object] = {
        "text": text,
        "interrupt": bool(interrupt),
    }
    if voice:
        payload["voice"] = voice
    if emotion:
        payload["emotion"] = emotion
    if volume is not None:
        payload["volume"] = max(0, min(100, int(volume)))
    return Action(type="play_voice", payload=payload)



def start_timer(duration_sec: int) -> Action:
    """构造一条启动定时器动作。"""
    return Action(type="start_timer", payload={"duration_sec": duration_sec})



def stop_timer() -> Action:
    """构造一条停止定时器动作。"""
    return Action(type="stop_timer", payload={})



def none_action() -> Action:
    """构造一条空动作。"""
    return Action(type="none", payload={})
