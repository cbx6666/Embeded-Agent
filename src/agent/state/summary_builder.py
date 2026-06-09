from __future__ import annotations

"""各 LLM 决策入口进入 prompt 前的紧凑状态汇总构建器。

- ``build_wellness_care_summary``：疲劳 / 负面情绪 / 姿态的窗口化关怀汇总（每 30s）。
- ``build_behavior_distraction_summary``：玩手机分心的严格窗口判定（每 20s）。
- ``build_environment_care_summary``：仅环境（光照 / 温度 / 湿度 / 噪声）关怀（每 60s）。
- ``build_sensor_status_summary``：每 5 分钟一次的传感器数值快照（确定性播报用）。

各汇总只整理信号，不写死“只要焦虑就一定 speak”；是否提醒由 Python 阈值 + prompt +
memory + Guard 共同决定。
"""

from collections import Counter
from typing import Any

from src.agent.policy_config import (
    BehaviorDistractionCheckPolicy,
    EnvironmentCareCheckPolicy,
    WellnessCareCheckPolicy,
)
from src.agent.state.agent_state import AgentState

# ---- severity 映射 ----------------------------------------------------------
_FATIGUE_SEVERITY: dict[str, float] = {
    "none": 0.0,
    "neutral": 0.0,
    "normal": 0.0,
    "mild": 0.4,
    "tired": 0.4,
    "low_energy": 0.4,
    "moderate": 0.65,
    "fatigued": 0.65,
    "weary": 0.65,
    "sleepy": 0.65,
    "drowsy": 0.65,
    "high": 0.8,
    "severe": 0.95,
    "exhausted": 0.95,
}

_EMOTION_SEVERITY: dict[str, float] = {
    "neutral": 0.0,
    "calm": 0.0,
    "happy": 0.0,
    "tired": 0.35,
    "low": 0.35,
    "down": 0.35,
    "sad": 0.55,
    "upset": 0.55,
    "unhappy": 0.55,
    "depressed": 0.55,
    "anxious": 0.7,
    "stressed": 0.7,
    "frustrated": 0.7,
    "worried": 0.7,
    "nervous": 0.7,
    "irritated": 0.7,
    "angry": 0.85,
    "fearful": 0.85,
    "panic": 0.85,
}

_ABNORMAL_ENV_LEVELS = frozenset(
    {"high", "low", "dark", "dim", "hot", "cold", "noisy", "loud", "humid", "dry", "bright"}
)


def _severity_of(value: str, mapping: dict[str, float]) -> float:
    return mapping.get(str(value or "").strip().lower(), 0.0)


def _trend_average_confidence(state: AgentState, signal: str) -> float | None:
    trend = state.runtime_history.signal_trends.get(signal)
    if not isinstance(trend, dict):
        return None
    summary = trend.get("confidence_summary")
    if not isinstance(summary, dict):
        return None
    average = summary.get("average")
    if average is None:
        return None
    try:
        return float(average)
    except (TypeError, ValueError):
        return None


def _effective_confidence(explicit: float | None, trend_average: float | None) -> float:
    if explicit is not None:
        return max(0.0, min(1.0, float(explicit)))
    if trend_average is not None:
        return max(0.0, min(1.0, float(trend_average)))
    return 0.0


def _signal_trend(state: AgentState, signal: str, mapping: dict[str, float]) -> str:
    """根据信号 current/previous 的 severity 变化估计趋势。"""

    trend = state.runtime_history.signal_trends.get(signal)
    if not isinstance(trend, dict):
        return "unknown"
    current = trend.get("current")
    if current is None:
        return "unknown"
    previous = trend.get("previous")
    if previous is None:
        return "stable"
    cur_sev = _severity_of(str(current), mapping)
    prev_sev = _severity_of(str(previous), mapping)
    if cur_sev > prev_sev:
        return "rising"
    if cur_sev < prev_sev:
        return "falling"
    return "stable"


# ---- environment ------------------------------------------------------------
def _env_abnormal_items(state: AgentState) -> list[dict[str, Any]]:
    env = state.environment
    items: list[dict[str, Any]] = []

    temp = env.temperature_c
    if temp is not None:
        if temp >= 30:
            items.append(_env_item("temperature", temp, "high", _scale(temp, 30, 38), "通风或降低室温"))
        elif temp <= 15:
            items.append(_env_item("temperature", temp, "low", _scale_low(temp, 15, 5), "注意保暖或调高室温"))
    elif str(env.temperature_level or "").lower() in {"high", "hot"}:
        items.append(_env_item("temperature", None, "high", 0.6, "通风或降低室温"))
    elif str(env.temperature_level or "").lower() in {"low", "cold"}:
        items.append(_env_item("temperature", None, "low", 0.6, "注意保暖或调高室温"))

    humidity = env.humidity_pct
    if humidity is not None:
        if humidity >= 70:
            items.append(_env_item("humidity", humidity, "high", _scale(humidity, 70, 90), "适当除湿或通风"))
        elif humidity <= 30:
            items.append(_env_item("humidity", humidity, "low", _scale_low(humidity, 30, 10), "适当加湿"))
    elif str(env.humidity_level or "").lower() in {"high", "humid"}:
        items.append(_env_item("humidity", None, "high", 0.6, "适当除湿或通风"))
    elif str(env.humidity_level or "").lower() in {"low", "dry"}:
        items.append(_env_item("humidity", None, "low", 0.6, "适当加湿"))

    noise = env.noise_db
    if noise is not None:
        if noise >= 65:
            items.append(_env_item("noise", noise, "high", _scale(noise, 65, 90), "降低噪声或换安静位置"))
    elif str(env.noise_level or "").lower() in {"high", "noisy", "loud"}:
        items.append(_env_item("noise", None, "high", 0.6, "降低噪声或换安静位置"))

    light = env.light_lux
    if light is not None:
        if light <= 150:
            items.append(_env_item("light", light, "dim", _scale_low(light, 150, 20), "调亮灯光"))
        elif light >= 800:
            items.append(_env_item("light", light, "bright", _scale(light, 800, 1500), "适当调暗或减少屏幕亮度"))
    elif str(env.light_level or "").lower() in {"low", "dark", "dim"}:
        items.append(_env_item("light", None, "dim", 0.6, "调亮灯光"))
    elif str(env.light_level or "").lower() in {"high", "bright"}:
        items.append(_env_item("light", None, "bright", 0.6, "适当调暗或减少屏幕亮度"))

    return items


def _env_item(env_type: str, value: float | None, level: str, severity: float, hint: str) -> dict[str, Any]:
    return {
        "type": env_type,
        "value": value,
        "level": level,
        "severity": round(max(0.5, min(0.95, severity)), 3),
        "suggestion_hint": hint,
    }


def _scale(value: float, start: float, end: float) -> float:
    if end <= start:
        return 0.6
    ratio = (value - start) / (end - start)
    return 0.5 + 0.45 * max(0.0, min(1.0, ratio))


def _scale_low(value: float, start: float, end: float) -> float:
    if start <= end:
        return 0.6
    ratio = (start - value) / (start - end)
    return 0.5 + 0.45 * max(0.0, min(1.0, ratio))


def _environment_block(state: AgentState) -> dict[str, Any]:
    env = state.environment
    abnormal = _env_abnormal_items(state)
    dominant = "none"
    if abnormal:
        dominant = max(abnormal, key=lambda item: item["severity"])["type"]
    return {
        "temperature_c": env.temperature_c,
        "temperature_level": env.temperature_level,
        "humidity_pct": env.humidity_pct,
        "humidity_level": env.humidity_level,
        "noise_db": env.noise_db,
        "noise_level": env.noise_level,
        "light_lux": env.light_lux,
        "light_level": env.light_level,
        "abnormal_items": abnormal,
        "dominant_environment_signal": dominant,
    }


def _attention_records_in_window(
    state: AgentState,
    *,
    window_sec: int,
    check_time: int | None,
) -> list[dict[str, Any]]:
    if check_time is None:
        return list(state.runtime_history.attention_records)
    cutoff = int(check_time) - max(1, int(window_sec))
    return [
        item
        for item in state.runtime_history.attention_records
        if isinstance(item, dict) and int(item.get("timestamp", 0)) >= cutoff
    ]


def _focus_summary_block(state: AgentState) -> dict[str, Any]:
    if not state.focus.active:
        return {"active": False}
    remaining = state.focus.remaining_sec
    try:
        remaining_int = max(0, int(remaining)) if remaining is not None else None
    except (TypeError, ValueError):
        remaining_int = None
    remaining_minutes = (
        round(remaining_int / 60, 1) if remaining_int is not None else None
    )
    return {
        "active": state.focus.active,
        "elapsed_sec": state.focus.elapsed_sec,
        "remaining_sec": remaining_int,
        "remaining_minutes": remaining_minutes,
        "target_duration_sec": state.focus.target_duration_sec,
    }


def _max_phone_confidence(phone_records: list[dict[str, Any]], fallback: float) -> float:
    """取窗口内 phone_use 记录的最高置信度；没有记录时回退到当前 state 置信度。"""

    best = fallback
    for item in phone_records:
        try:
            best = max(best, float(item.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
    return best


def _is_distraction_trigger_candidate(
    state: AgentState,
    *,
    policy: BehaviorDistractionCheckPolicy,
    check_time: int | None,
) -> tuple[bool, dict[str, Any]]:
    """基于最近窗口的占比 + 最近是否仍在玩来判定，而非依赖检查瞬间的 current_behavior。

    瞬时判定（要求那一刻恰好 phone_use / distracted）会因 YOLO 个别帧漏检而落空，
    导致「明明一直在玩却迟迟不提醒」；这里改为窗口统计，更贴近真实持续行为。
    """

    records = _attention_records_in_window(state, window_sec=policy.window_sec, check_time=check_time)
    phone_records = [
        item
        for item in records
        if str(item.get("behavior", "")).strip().lower() == "phone_use"
    ]
    yolo_records = [
        item for item in phone_records if bool(item.get("yolo_phone_detected"))
    ]
    working_records = [
        item
        for item in records
        if str(item.get("behavior", "")).strip().lower() == "working"
    ]
    phone_use_events = len(phone_records)
    # 占比只在 phone_use vs working 之间算，避免其它行为记录稀释。
    competing = phone_use_events + len(working_records)
    ratio = (phone_use_events / competing) if competing else 0.0
    total = len(records)

    last_phone_ts = None
    for item in phone_records:
        try:
            last_phone_ts = max(last_phone_ts or 0, int(item.get("timestamp", 0)))
        except (TypeError, ValueError):
            continue
    recent_active = False
    if last_phone_ts is not None:
        if check_time is None:
            recent_active = True
        else:
            recent_active = int(check_time) - int(last_phone_ts) <= policy.recent_active_sec

    confidence = _max_phone_confidence(phone_records, float(state.user.behavior_confidence or 0.0))

    presence = str(state.user.presence or "").strip().lower()
    detail = {
        "phone_use_events": phone_use_events,
        "yolo_phone_events": len(yolo_records),
        "window_records": total,
        "phone_use_ratio": round(ratio, 3),
        "max_phone_confidence": round(confidence, 3),
        "user_presence": presence,
        "last_phone_age_sec": (
            None if last_phone_ts is None or check_time is None else int(check_time) - int(last_phone_ts)
        ),
    }

    # 不在场时默认不提醒；但若窗口内已有持续 phone_use（pose 漏检造成的 away 误判），仍允许触发。
    if presence != policy.require_presence:
        if phone_use_events < policy.min_phone_use_events or not recent_active:
            return False, {**detail, "reason": f"user not present (presence={presence})"}

    if phone_use_events < policy.min_phone_use_events:
        return False, {**detail, "reason": "insufficient phone_use events in window"}
    if len(yolo_records) < policy.min_yolo_phone_events:
        return False, {**detail, "reason": "insufficient yolo phone detections in window"}
    if ratio < policy.min_phone_use_ratio:
        return False, {**detail, "reason": "phone_use ratio below threshold"}
    if not recent_active:
        return False, {**detail, "reason": "no recent phone_use (likely already stopped)"}
    if confidence < policy.min_confidence:
        return False, {**detail, "reason": "behavior confidence below threshold"}
    if policy.require_yolo_phone_on_latest:
        latest_phone = phone_records[-1] if phone_records else None
        if latest_phone is None or not bool(latest_phone.get("yolo_phone_detected")):
            return False, {**detail, "reason": "latest phone_use record lacks yolo phone detection"}

    return True, {**detail, "reason": "sustained phone_use in window (ratio + recent + yolo)"}


def build_behavior_distraction_summary(
    state: AgentState,
    *,
    memories: dict[str, Any] | None = None,
    policy: BehaviorDistractionCheckPolicy | None = None,
    check_time: int | None = None,
) -> dict[str, Any]:
    """构造 behavior_distraction_check 进入 LLM 前的严格分心汇总。"""

    policy = policy or BehaviorDistractionCheckPolicy()
    records = _attention_records_in_window(state, window_sec=policy.window_sec, check_time=check_time)
    phone_records = [
        item
        for item in records
        if str(item.get("behavior", "")).strip().lower() == "phone_use"
    ]
    yolo_records = [
        item for item in phone_records if bool(item.get("yolo_phone_detected"))
    ]
    trigger_candidate, trigger_detail = _is_distraction_trigger_candidate(
        state, policy=policy, check_time=check_time
    )

    return {
        "check_time": check_time,
        "user_presence": state.user.presence,
        "current_attention": state.user.attention,
        "current_behavior": state.user.behavior,
        "behavior_confidence": state.user.behavior_confidence,
        "attention_confidence": state.user.attention_confidence,
        "current_activity": state.user.current_activity,
        "window_sec": policy.window_sec,
        "window_phone_use_events": len(phone_records),
        "window_yolo_phone_events": len(yolo_records),
        "distraction_event_count": state.runtime_history.distraction_event_count,
        "recent_attention_records": records[-6:],
        "trigger_candidate": trigger_candidate,
        "trigger_detail": trigger_detail,
        "focus_summary": _focus_summary_block(state),
        "recent_reminders": [
            {
                "reason": item.get("reason"),
                "timestamp": item.get("timestamp"),
                "text": item.get("text"),
            }
            for item in state.runtime_history.reminder_records[-5:]
            if str(item.get("reason", "")) == "distraction_reminder"
        ],
        "memories": dict(memories or {}),
    }


# ---- 窗口化统计（wellness_care_check 用）-----------------------------------
def _recent_signal_values(
    state: AgentState, signal: str, *, window_sec: int, check_time: int | None
) -> list[dict[str, Any]]:
    """取某信号最近窗口内的时间序列样本（来自 signal_trends.recent_values）。"""

    trend = state.runtime_history.signal_trends.get(signal)
    if not isinstance(trend, dict):
        return []
    values = [item for item in trend.get("recent_values", []) if isinstance(item, dict)]
    if check_time is None:
        return values
    cutoff = int(check_time) - max(1, int(window_sec))
    return [item for item in values if int(item.get("timestamp", 0) or 0) >= cutoff]


def _value_in(item: dict[str, Any], target_set: frozenset[str]) -> bool:
    return str(item.get("value", "")).strip().lower() in target_set


def _sustained_tail_sec(
    values: list[dict[str, Any]], target_set: frozenset[str], *, check_time: int | None
) -> int:
    """以最新样本为终点，连续命中 target_set 的时间长度（秒）。

    最新样本不在集合内时返回 0（避免把已恢复的状态算成持续）。
    """

    if not values:
        return 0
    run: list[dict[str, Any]] = []
    for item in reversed(values):
        if _value_in(item, target_set):
            run.append(item)
        else:
            break
    if not run:
        return 0
    latest_ts = int(run[0].get("timestamp", 0) or 0)
    earliest_ts = int(run[-1].get("timestamp", 0) or 0)
    end_ts = max(latest_ts, int(check_time)) if check_time is not None else latest_ts
    return max(0, end_ts - earliest_ts)


def _ratio_in(values: list[dict[str, Any]], target_set: frozenset[str]) -> float:
    if not values:
        return 0.0
    hits = sum(1 for item in values if _value_in(item, target_set))
    return round(hits / len(values), 3)


def _peak_confidence(values: list[dict[str, Any]]) -> float:
    confs = [
        float(item["confidence"]) for item in values if item.get("confidence") is not None
    ]
    return round(max(confs), 3) if confs else 0.0


def _dominant_value(values: list[dict[str, Any]], target_set: frozenset[str]) -> str | None:
    counts = Counter(
        str(item.get("value", "")).strip().lower()
        for item in values
        if _value_in(item, target_set)
    )
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _emotion_valence(label: str, negative_labels: frozenset[str]) -> str:
    norm = str(label or "").strip().lower()
    if norm in negative_labels:
        return "negative"
    if norm in {"happy", "calm", "content", "positive"}:
        return "positive"
    return "neutral"


_WELLNESS_REASON_TO_FOCUS = {
    "rest_reminder": "fatigue",
    "emotion_reminder": "emotion",
    "posture_reminder": "posture",
}


def _last_wellness_broadcast_focus(state: AgentState) -> str | None:
    """最近一次 wellness 关怀播报对应的 focus（fatigue/emotion/posture）。"""

    for item in reversed(state.runtime_history.reminder_records):
        reason = str(item.get("reason", ""))
        focus = _WELLNESS_REASON_TO_FOCUS.get(reason)
        if focus:
            return focus
    return None


def _pick_recommended_care_focus(
    *,
    fatigue_trigger: bool,
    emotion_trigger: bool,
    posture_trigger: bool,
    strong_negative: bool,
    very_long_fatigue: bool,
    last_focus: str | None,
) -> str:
    """在已触发的关怀维度里选 focus；若上次已播某一类且其他类仍触发，则优先轮换。"""

    active = [
        name
        for name, triggered in (
            ("fatigue", fatigue_trigger),
            ("emotion", emotion_trigger),
            ("posture", posture_trigger),
        )
        if triggered
    ]
    if not active:
        return "none"

    pool = list(active)
    deprioritized: str | None = None
    if last_focus in pool and len(pool) > 1:
        deprioritized = last_focus
        pool = [focus for focus in pool if focus != last_focus]

    if "emotion" in pool and strong_negative and not very_long_fatigue:
        return "emotion"
    if "fatigue" in pool and very_long_fatigue:
        return "fatigue"
    if "fatigue" in pool:
        return "fatigue"
    if "emotion" in pool:
        return "emotion"
    if "posture" in pool:
        return "posture"
    # 只剩被轮换掉的上次 focus 时仍允许播（例如只有姿态仍在触发）。
    if deprioritized:
        return deprioritized
    return active[0]


def build_wellness_care_summary(
    state: AgentState,
    *,
    memories: dict[str, Any] | None = None,
    policy: WellnessCareCheckPolicy | None = None,
    check_time: int | None = None,
) -> dict[str, Any]:
    """wellness_care_check：疲劳 / 负面情绪 / 姿态的窗口化关怀汇总。

    三者 OR 关系：任一成立即 ``should_care=True``。只含疲劳/情绪/姿态，
    不含环境。Python 在这里算出 ``should_care`` 与 ``selected_intent`` 候选，
    LLM 只负责文案，不能否决强触发，也不能改成环境提醒。
    """

    policy = policy or WellnessCareCheckPolicy()
    win = policy.recent_window_sec

    # ---- fatigue ----
    fatigue_vals = _recent_signal_values(state, "fatigue", window_sec=win, check_time=check_time)
    current_fatigue = str(state.user.fatigue_level or "none").strip().lower()
    fatigue_conf = _effective_confidence(
        state.user.fatigue_confidence, _trend_average_confidence(state, "fatigue")
    )
    sustained_high_sec = _sustained_tail_sec(fatigue_vals, policy.fatigue_high_levels, check_time=check_time)
    sustained_mod_sec = _sustained_tail_sec(
        fatigue_vals, policy.fatigue_moderate_levels, check_time=check_time
    )
    high_ratio = _ratio_in(fatigue_vals, policy.fatigue_high_levels)
    fatigue_peak = _peak_confidence(fatigue_vals)
    fatigue_trigger = (
        sustained_high_sec >= policy.fatigue_sustained_high_sec
        or sustained_mod_sec >= policy.fatigue_sustained_moderate_or_high_sec
        or high_ratio >= policy.fatigue_high_ratio_threshold
        or (
            fatigue_peak >= policy.fatigue_peak_confidence_threshold
            and current_fatigue in policy.fatigue_moderate_levels
        )
    )
    fatigue_block = {
        "current_level": current_fatigue,
        "current_confidence": round(fatigue_conf, 4),
        "sustained_high_sec": sustained_high_sec,
        "sustained_moderate_or_high_sec": sustained_mod_sec,
        "high_ratio_recent_window": high_ratio,
        "peak_confidence_recent_window": fatigue_peak,
        "trend": _signal_trend(state, "fatigue", _FATIGUE_SEVERITY),
        "trigger_candidate": bool(fatigue_trigger),
    }

    # ---- emotion ----
    emotion_vals = _recent_signal_values(state, "emotion", window_sec=win, check_time=check_time)
    current_emotion = str(state.user.emotion or "neutral").strip().lower()
    emotion_conf = _effective_confidence(
        state.user.emotion_confidence, _trend_average_confidence(state, "emotion")
    )
    negative_streak_sec = _sustained_tail_sec(
        emotion_vals, policy.emotion_negative_labels, check_time=check_time
    )
    negative_ratio = _ratio_in(emotion_vals, policy.emotion_negative_labels)
    emotion_peak = _peak_confidence(emotion_vals)
    dominant_negative = _dominant_value(emotion_vals, policy.emotion_negative_labels)
    if dominant_negative is None and current_emotion in policy.emotion_negative_labels:
        dominant_negative = current_emotion
    emotion_trigger = (
        negative_streak_sec >= policy.emotion_negative_streak_sec
        or negative_ratio >= policy.emotion_negative_ratio_threshold
        or current_emotion in policy.emotion_strong_labels
        or (dominant_negative is not None and dominant_negative in policy.emotion_strong_labels)
    )
    emotion_block = {
        "current_emotion": current_emotion,
        "current_valence": _emotion_valence(current_emotion, policy.emotion_negative_labels),
        "current_confidence": round(emotion_conf, 4),
        "negative_ratio_recent_window": negative_ratio,
        "dominant_negative_emotion": dominant_negative,
        "negative_streak_sec": negative_streak_sec,
        "peak_confidence_recent_window": emotion_peak,
        "trigger_candidate": bool(emotion_trigger),
    }

    # ---- posture ----
    posture_vals = _recent_signal_values(state, "posture", window_sec=win, check_time=check_time)
    current_posture = str(state.user.posture or "unknown").strip().lower()
    sustained_bad_sec = _sustained_tail_sec(posture_vals, policy.posture_bad_levels, check_time=check_time)
    bad_ratio = _ratio_in(posture_vals, policy.posture_bad_levels)
    current_is_bad = current_posture in policy.posture_bad_levels
    posture_signal = (
        sustained_bad_sec >= policy.posture_sustained_bad_sec
        or bad_ratio >= policy.posture_bad_ratio_threshold
    )
    posture_trigger = posture_signal and (
        current_is_bad if policy.posture_require_current_bad else True
    )
    posture_block = {
        "current_posture": current_posture,
        "current_is_bad": current_is_bad,
        "bad_posture_ratio_recent_window": bad_ratio,
        "sustained_bad_posture_sec": sustained_bad_sec,
        "trigger_candidate": bool(posture_trigger),
    }

    # ---- 汇总 care_triggers + recommended_care_focus ----
    care_triggers: list[dict[str, Any]] = []
    if fatigue_trigger:
        care_triggers.append({"type": "fatigue", "level": current_fatigue})
    if emotion_trigger:
        care_triggers.append({"type": "emotion", "label": dominant_negative or current_emotion})
    if posture_trigger:
        care_triggers.append({"type": "posture", "label": current_posture})

    strong_negative = (
        current_emotion in policy.emotion_strong_labels
        or (dominant_negative is not None and dominant_negative in policy.emotion_strong_labels)
    )
    very_long_fatigue = sustained_high_sec >= 2 * policy.fatigue_sustained_high_sec
    last_wellness_focus = _last_wellness_broadcast_focus(state)

    recommended = _pick_recommended_care_focus(
        fatigue_trigger=fatigue_trigger,
        emotion_trigger=emotion_trigger,
        posture_trigger=posture_trigger,
        strong_negative=strong_negative,
        very_long_fatigue=very_long_fatigue,
        last_focus=last_wellness_focus,
    )

    should_care = bool(care_triggers)
    if not should_care:
        care_reason = "no notable fatigue / negative emotion / bad posture in recent window"
    else:
        parts = []
        if fatigue_trigger:
            parts.append(
                f"fatigue(sustained_high={sustained_high_sec}s,high_ratio={high_ratio})"
            )
        if emotion_trigger:
            parts.append(
                f"emotion({dominant_negative or current_emotion},streak={negative_streak_sec}s,ratio={negative_ratio})"
            )
        if posture_trigger:
            parts.append(f"posture({current_posture},ratio={bad_ratio})")
        care_reason = "; ".join(parts)

    return {
        "check_time": check_time,
        "recent_window_sec": win,
        "user_presence": state.user.presence,
        "fatigue": fatigue_block,
        "emotion": emotion_block,
        "posture": posture_block,
        "care_triggers": care_triggers,
        "recommended_care_focus": recommended,
        "last_wellness_focus": last_wellness_focus,
        "should_care": should_care,
        "care_reason": care_reason,
        "focus_summary": _focus_summary_block(state),
        "recent_reminders": [
            {
                "reason": item.get("reason"),
                "timestamp": item.get("timestamp"),
                "text": item.get("text"),
            }
            for item in state.runtime_history.reminder_records[-5:]
            if str(item.get("reason", "")) in {"rest_reminder", "emotion_reminder", "posture_reminder"}
        ],
        "memories": dict(memories or {}),
    }


def build_environment_care_summary(
    state: AgentState,
    *,
    memories: dict[str, Any] | None = None,
    policy: EnvironmentCareCheckPolicy | None = None,
    check_time: int | None = None,
) -> dict[str, Any]:
    """environment_care_check：仅环境（光照/温度/湿度/噪声）关怀汇总。

    不含 fatigue/emotion/posture。是否播由 LLM 判断，可 no_op；只产 environment_warning。
    """

    policy = policy or EnvironmentCareCheckPolicy()
    env = state.environment
    abnormal = _env_abnormal_items(state)
    abnormal_types = {item["type"] for item in abnormal}

    def _block(kind: str, value: Any, level: Any) -> dict[str, Any]:
        return {
            "value": value,
            "level": str(level or "normal"),
            "abnormal": kind in abnormal_types,
        }

    triggers = [
        {
            "type": item["type"],
            "level": item["level"],
            "severity": item["severity"],
            "suggestion_hint": item.get("suggestion_hint"),
        }
        for item in abnormal
    ]
    should_consider = bool(triggers)
    care_reason = (
        "; ".join(f"{t['type']}={t['level']}" for t in triggers)
        if triggers
        else "no abnormal environment readings"
    )

    return {
        "check_time": check_time,
        "user_presence": state.user.presence,
        "light": _block("light", env.light_lux, env.light_level),
        "temperature": _block("temperature", env.temperature_c, env.temperature_level),
        "humidity": _block("humidity", env.humidity_pct, env.humidity_level),
        "noise": _block("noise", env.noise_db, env.noise_level),
        "environment_triggers": triggers,
        "should_consider_care": should_consider,
        "care_reason": care_reason,
        "recent_reminders": [
            {
                "reason": item.get("reason"),
                "timestamp": item.get("timestamp"),
                "text": item.get("text"),
            }
            for item in state.runtime_history.reminder_records[-5:]
            if str(item.get("reason", "")) == "environment_warning"
        ],
        "memories": dict(memories or {}),
    }


def build_sensor_status_summary(
    state: AgentState, *, check_time: int | None = None
) -> dict[str, Any]:
    """为 sensor_status_report 提供具体传感器数值与用户状态快照。"""

    environment = _environment_block(state)
    return {
        "check_time": check_time,
        "temperature_c": state.environment.temperature_c,
        "temperature_level": state.environment.temperature_level,
        "humidity_pct": state.environment.humidity_pct,
        "humidity_level": state.environment.humidity_level,
        "noise_db": state.environment.noise_db,
        "noise_level": state.environment.noise_level,
        "light_lux": state.environment.light_lux,
        "light_level": state.environment.light_level,
        "presence": state.user.presence,
        "attention": state.user.attention,
        "fatigue_level": state.user.fatigue_level,
        "emotion": state.user.emotion,
        "posture": state.user.posture,
        "abnormal_items": [
            {"type": item["type"], "value": item["value"], "level": item["level"]}
            for item in environment["abnormal_items"]
        ],
    }
