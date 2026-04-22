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


def start_voice_capture(source: str, trigger: str | None = None) -> Action:
    """构造一条启动语音采集动作。"""
    payload = {"source": source}
    if trigger:
        payload["trigger"] = trigger
    return Action(type="start_voice_capture", payload=payload)


def stop_voice_capture(source: str, reason: str | None = None) -> Action:
    """构造一条停止语音采集动作。"""
    payload = {"source": source}
    if reason:
        payload["reason"] = reason
    return Action(type="stop_voice_capture", payload=payload)


def set_tts_voice(voice_id: str) -> Action:
    """构造一条设置音色动作。"""
    return Action(type="set_tts_voice", payload={"voice_id": voice_id})


def set_tts_volume(volume: int) -> Action:
    """构造一条设置音量动作。"""
    return Action(type="set_tts_volume", payload={"volume": volume})


def set_tts_speed(speed: float) -> Action:
    """构造一条设置语速动作。"""
    return Action(type="set_tts_speed", payload={"speed": speed})


def environment_alert(sensor: str, level: str, message: str) -> Action:
    """构造一条环境提醒动作。"""
    return Action(
        type="environment_alert",
        payload={"sensor": sensor, "level": level, "message": message},
    )


def none_action() -> Action:
    """构造一条空动作。"""
    return Action(type="none", payload={})
