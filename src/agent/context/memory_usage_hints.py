from __future__ import annotations

"""临时「记忆使用策略」视图（memory_usage_hints）。

每次 LLM 调用前，规则层把已有的 **UserProfile + 多维 Memory + 运行记录 + 当前状态**
临时整理成一份「本轮如何使用记忆」的策略，供 prompt 使用。它：

- **不是新记忆**，不落盘，不写入 ``user_memory.json``；
- **不写死兴趣爱好枚举**（不依赖「篮球/音乐/笑话/相声」这类固定词）；
- **不生成最终文案**，只做整理 / 筛选 / 归类 / 限制，自然语言仍由 LLM 生成；
- ``wellness_care`` 返回 ``personalization_candidates``（5~8 条、多类去重），由 LLM 自选融入；
  其他场景仍可用 ``recommended_content`` 轮换点缀。

候选与分类的依据优先级：
``memory.tags`` / ``memory.type`` > profile 结构化字段 > 极少量通用语义关键词兜底。
关键词只用于把候选**归类**（如「音乐」→ relaxing_content），不用于决定某条偏好是否存在；
用户以后喜欢「京剧 / 播客 / 羽毛球 / 散步 / 历史故事」也能走同一套逻辑。

输入的 ``memories`` 是 ``MemoryService.retrieve_user_context`` 的 ``by_type`` 结构：
``{plural_group_key: [{content, confidence, tags, evidence}, ...]}``。
"""

from typing import Any

from src.agent.memory.memory_model import GROUP_KEY_BY_TYPE

# 主链路真实存在的 context_type。
_PERSONALIZED_CONTEXTS = frozenset(
    {"speech", "wellness_care", "behavior_distraction", "environment_care"}
)

# plural 分组键 -> memory.type，用于把 retrieve 结果还原成类型。
_TYPE_BY_GROUP_KEY = {group: mtype for mtype, group in GROUP_KEY_BY_TYPE.items()}

# wellness 场景传给 LLM 的个性化候选数量范围。
_PERSONALIZATION_MIN = 5
_PERSONALIZATION_MAX = 8

# 同类放松/娱乐候选去重：这些 category 合计最多保留 1 条，避免 5 条全是音乐/相声。
_RELAXING_CATEGORIES = frozenset({"relaxing_content", "unknown"})

# 候选 topic 标签（供 LLM 理解类别，非文案模板）。
_TOPIC_BY_CATEGORY: dict[str, str] = {
    "relaxing_content": "放松娱乐",
    "physical_activity": "运动兴趣",
    "focus_strategy": "专注习惯",
    "care_strategy": "关怀偏好",
    "companion_style": "陪伴风格",
    "dislike": "不喜欢的打扰",
    "habit": "生活习惯",
    "work_style": "学习工作状态",
    "emotion_pattern": "情绪习惯",
    "interest_topic": "兴趣爱好",
    "supervision": "监督偏好",
    "unknown": "用户偏好",
}

_WELLNESS_MEMORY_USAGE_INSTRUCTION = (
    "你不必每轮都显式说出用户记忆；让语气自然贴近这个人即可。"
    "可从 personalization_candidates 中选最自然的一条融入，也可以完全不点名。"
    "trigger_focus 只是背景，不要为了打卡而硬塞兴趣或记忆。"
)

# 通用语义分类关键词（仅作兜底归类，不作为主要逻辑，也不枚举具体爱好）。
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "relaxing_content": (
        "music", "song", "audio", "joke", "humor", "comedy", "podcast", "video", "story",
        "音乐", "歌", "笑话", "段子", "相声", "播客", "视频", "影", "剧", "放松", "听",
    ),
    "physical_activity": (
        "sport", "ball", "run", "walk", "exercise", "workout", "fitness", "stretch",
        "运动", "球", "跑", "散步", "走", "健身", "锻炼", "活动", "拉伸",
    ),
    "focus_strategy": (
        "focus", "study", "work", "pomodoro", "session", "plan", "task", "concentrate",
        "专注", "学习", "工作", "番茄", "计划", "任务", "目标", "沉浸",
    ),
    "interest_topic": (
        "history", "topic", "knowledge", "science", "news", "read",
        "历史", "话题", "知识", "科普", "故事", "阅读", "新闻",
    ),
}

# 表达「不喜欢被打断 / 说教 / 太正式 / 太长」的通用语义信号。
_AVOID_INTERRUPT_KEYWORDS = (
    "interrupt", "frequent", "remind", "nag", "lecture", "formal", "long",
    "打断", "频繁", "提醒", "唠叨", "说教", "命令", "正式", "啰嗦", "太长", "别管",
)
# 表达「沉浸 / 长时间专注」的通用语义信号（提醒需更短）。
_IMMERSION_KEYWORDS = ("沉浸", "专注", "immersive", "deep", "long", "连续")


def build_memory_usage_hints(
    *,
    profile: dict | None,
    preferences: dict | None,
    memories: dict | None,
    recent_interaction: dict | None = None,
    runtime: dict | None = None,
    context_type: str,
    current_state: dict | None = None,
    rotation_seed: int = 0,
    user_query: str | None = None,
) -> dict:
    """整理出本轮 LLM 如何使用记忆的临时策略（不落盘）。

    返回结构见模块文档；字段按 context_type 收敛。``rotation_seed`` 用于在多个兴趣候选间
    **轮换**本轮要点缀的兴趣（讲笑话 -> 打篮球 -> 听相声 …），由 ``AgentCore`` 维护并自增。
    """

    profile = profile or {}
    preferences = preferences or {}
    memories = memories or {}
    runtime = runtime or {}
    current_state = current_state or {}

    use_memory = context_type in _PERSONALIZED_CONTEXTS
    focus = _resolve_focus(context_type, current_state, user_query=user_query)

    raw_candidates = _collect_candidates(profile, preferences, memories)
    avoid_patterns = _collect_avoid_patterns(preferences, memories)
    style_hints = _collect_style_hints(preferences, memories)
    recently_used = _recent_angles(runtime)

    personalization_candidates: list[dict[str, Any]] = []
    memory_usage_instruction = ""
    if context_type == "wellness_care":
        pool = _collect_personalization_pool(profile, preferences, memories)
        personalization_candidates = _build_personalization_candidates(pool)
        candidates = personalization_candidates
        memory_usage_instruction = _WELLNESS_MEMORY_USAGE_INSTRUCTION
        recommended_content = None
    else:
        candidates = _filter_candidates_for_context(
            raw_candidates,
            context_type=context_type,
            focus=focus,
            current_state=current_state,
        )
        recommended_content = _pick_rotated_interest(candidates, rotation_seed=rotation_seed)

    recommended_angle, recommended_reason = _pick_recommended_angle(
        candidates,
        context_type=context_type,
        focus=focus,
        recently_used=recently_used,
        focus_active=bool(current_state.get("focus_active")),
    )

    personalization_level = _personalization_level(
        context_type=context_type,
        use_memory=use_memory,
        candidates=candidates,
        personalization_candidates=personalization_candidates,
    )
    preferred_tone = _preferred_tone(context_type, focus, avoid_patterns, style_hints)

    result: dict[str, Any] = {
        "context_type": context_type,
        "focus": focus,
        "use_memory": use_memory,
        "personalization_level": personalization_level,
        "preferred_tone": preferred_tone,
        "suggestion_candidates": candidates,
        "avoid_patterns": avoid_patterns,
        "style_hints": style_hints,
        "recently_used_angles": recently_used,
        "recommended_angle": recommended_angle,
        "recommended_content": recommended_content,
        "reason": recommended_reason,
    }
    if context_type == "wellness_care":
        result["personalization_candidates"] = personalization_candidates
        result["memory_usage_instruction"] = memory_usage_instruction
    return result


# 可作为「兴趣点缀」轮换的类别（真实兴趣/内容/活动，不含纯策略/习惯）。
_INTEREST_CATEGORIES = frozenset({"relaxing_content", "physical_activity", "interest_topic"})


def _pick_rotated_interest(
    candidates: list[dict[str, Any]], *, rotation_seed: int
) -> dict[str, Any] | None:
    """在兴趣候选中按 rotation_seed 轮换选一个，作为本轮要点缀的兴趣。

    候选按 label 稳定排序，确保跨调用顺序一致，``rotation_seed`` 自增即可轮换；
    没有任何兴趣候选时返回 None（例如姿态/环境场景，或用户没有兴趣记忆）。
    """

    interests = [
        c
        for c in candidates
        if c.get("category") in _INTEREST_CATEGORIES
        or c.get("memory_type") in {"hobby", "preference"}
    ]
    if not interests:
        return None
    interests = sorted(interests, key=lambda c: str(c.get("label") or ""))
    chosen = interests[int(rotation_seed) % len(interests)]
    return {
        "label": chosen.get("label"),
        "category": chosen.get("category"),
        "source": chosen.get("source"),
        "rotation_pool_size": len(interests),
        "note": "记忆原文是第三人称，请改写成对用户说的第二人称口语后自然点缀",
    }


# ---- 候选收集 ----------------------------------------------------------------
# 哪些 memory.type 可作为正向建议候选（dislike/interaction_style 进 avoid/style）。
_SUGGESTION_MEMORY_TYPES = frozenset(
    {"hobby", "preference", "care_strategy", "habit", "work_style"}
)


def _collect_candidates(
    profile: dict, preferences: dict, memories: dict
) -> list[dict[str, Any]]:
    """从 profile / preferences / memories 收集可用于个性化的候选，不做硬编码爱好枚举。"""

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(
        label: str,
        *,
        category: str,
        source: str,
        memory_type: str | None,
        confidence: float,
        reason: str,
    ) -> None:
        text = str(label or "").strip()
        if not text:
            return
        key = f"{source}:{text.lower()}"
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "label": text,
                "category": category,
                "source": source,
                "memory_type": memory_type,
                "confidence": round(float(confidence), 3),
                "reason": reason,
            }
        )

    # profile.hobbies / preference 内容类字段：显式资料，可信度最高。
    for hobby in _as_list(profile.get("hobbies")):
        _add(
            hobby,
            category=_categorize(hobby, []),
            source="profile",
            memory_type=None,
            confidence=0.95,
            reason="用户显式资料中的兴趣爱好",
        )
    for key, conf, why in (
        ("favorite_content_types", 0.9, "用户显式偏好的内容类型"),
        ("favorite_music_styles", 0.9, "用户显式偏好的音乐风格"),
    ):
        for value in _as_list(preferences.get(key)):
            _add(
                value,
                category=_categorize(value, []),
                source="preference",
                memory_type=None,
                confidence=conf,
                reason=why,
            )

    # memories：推断记忆，按 type 决定用途。
    for group_key, items in memories.items():
        memory_type = _TYPE_BY_GROUP_KEY.get(str(group_key), str(group_key))
        if memory_type not in _SUGGESTION_MEMORY_TYPES:
            continue
        for item in _as_list(items):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            tags = [str(t) for t in _as_list(item.get("tags"))]
            confidence = float(item.get("confidence") or 0.0)
            category = (
                "care_strategy" if memory_type == "care_strategy" else _categorize(content, tags)
            )
            _add(
                content,
                category=category,
                source="memory",
                memory_type=memory_type,
                confidence=confidence,
                reason=f"来自 {memory_type} 记忆",
            )
    return candidates


# 广义用户画像类型：wellness 个性化候选除兴趣外，也纳入陪伴风格 / 情绪习惯 / 不喜欢项。
_PERSONALIZATION_EXTRA_TYPES = frozenset({"interaction_style", "emotion_pattern", "dislike"})
_SUPERVISION_KEYWORDS = ("严格", "监督", "盯", "strict", "supervise")


def _collect_personalization_pool(
    profile: dict, preferences: dict, memories: dict
) -> list[dict[str, Any]]:
    """wellness 场景：收集更广的用户画像候选，不按 trigger 窄化。"""

    pool = list(_collect_candidates(profile, preferences, memories))
    seen = {f"{c.get('source')}:{str(c.get('label') or '').lower()}" for c in pool}

    def _append(
        label: str,
        *,
        category: str,
        source: str,
        memory_type: str,
        confidence: float,
        reason: str,
    ) -> None:
        text = str(label or "").strip()
        if not text:
            return
        key = f"{source}:{text.lower()}"
        if key in seen:
            return
        seen.add(key)
        pool.append(
            {
                "label": text,
                "category": category,
                "source": source,
                "memory_type": memory_type,
                "confidence": round(float(confidence), 3),
                "reason": reason,
            }
        )

    for group_key, items in memories.items():
        memory_type = _TYPE_BY_GROUP_KEY.get(str(group_key), str(group_key))
        if memory_type not in _PERSONALIZATION_EXTRA_TYPES:
            continue
        for item in _as_list(items):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            confidence = float(item.get("confidence") or 0.0)
            if memory_type == "interaction_style":
                category = "companion_style"
            elif memory_type == "emotion_pattern":
                category = "emotion_pattern"
            else:
                category = "dislike"
            _append(
                content,
                category=category,
                source="memory",
                memory_type=memory_type,
                confidence=confidence,
                reason=f"来自 {memory_type} 记忆",
            )

    for item in pool:
        label = str(item.get("label") or "")
        if item.get("category") in {"preference", "unknown"} and _contains_any(
            label.lower(), _SUPERVISION_KEYWORDS
        ):
            item["category"] = "supervision"
    return pool


def _build_personalization_candidates(
    pool: list[dict[str, Any]], *, limit: int = _PERSONALIZATION_MAX
) -> list[dict[str, Any]]:
    """多类去重后返回 5~8 条候选；不在 Python 层指定本轮必须用哪一条。"""

    if not pool:
        return []

    sorted_pool = sorted(
        pool,
        key=lambda c: (
            -float(c.get("confidence") or 0.0),
            str(c.get("label") or ""),
        ),
    )
    seen_labels: set[str] = set()
    category_counts: dict[str, int] = {}
    relaxing_used = False
    result: list[dict[str, Any]] = []

    for cand in sorted_pool:
        label = str(cand.get("label") or "").strip()
        if not label or label.lower() in seen_labels:
            continue
        category = str(cand.get("category") or "unknown")
        if category in _RELAXING_CATEGORIES and relaxing_used:
            continue
        if category_counts.get(category, 0) >= 2:
            continue
        seen_labels.add(label.lower())
        category_counts[category] = category_counts.get(category, 0) + 1
        if category in _RELAXING_CATEGORIES:
            relaxing_used = True
        entry = dict(cand)
        entry["topic"] = _TOPIC_BY_CATEGORY.get(category, "用户偏好")
        result.append(entry)
        if len(result) >= limit:
            break

    return result


def _collect_avoid_patterns(preferences: dict, memories: dict) -> list[dict[str, Any]]:
    """收敛「应避免的表达方式」：来自 dislike / 部分 work_style / interaction_style / 反感话题。"""

    patterns: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(label: str, *, source: str, reason: str) -> None:
        text = str(label or "").strip()
        if not text or text.lower() in seen:
            return
        seen.add(text.lower())
        patterns.append({"label": text, "source": source, "reason": reason})

    for topic in _as_list(preferences.get("disliked_topics")):
        _add(topic, source="profile", reason="用户显式不喜欢的话题/方式")

    for group_key, items in memories.items():
        memory_type = _TYPE_BY_GROUP_KEY.get(str(group_key), str(group_key))
        for item in _as_list(items):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            lower = content.lower()
            if memory_type == "dislike":
                _add(content, source="memory", reason="来自 dislike 记忆")
            elif memory_type in {"work_style", "interaction_style"} and _contains_any(
                lower, _AVOID_INTERRUPT_KEYWORDS + _IMMERSION_KEYWORDS
            ):
                _add(
                    content,
                    source="memory",
                    reason=f"来自 {memory_type}：暗示不喜欢被频繁打断/说教",
                )
    return patterns


def _collect_style_hints(preferences: dict, memories: dict) -> list[str]:
    """整理语气/风格提示：reminder_style / speech_style / interaction_style 记忆。"""

    hints: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        value = str(text or "").strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            hints.append(value)

    reminder_style = str(preferences.get("reminder_style") or "").strip().lower()
    if reminder_style == "gentle":
        _add("语气温和、像朋友")
    elif reminder_style:
        _add(f"提醒风格：{reminder_style}")
    speech_style = str(preferences.get("speech_style") or "").strip()
    if speech_style:
        _add(f"表达风格：{speech_style}")

    for group_key, items in memories.items():
        memory_type = _TYPE_BY_GROUP_KEY.get(str(group_key), str(group_key))
        if memory_type != "interaction_style":
            continue
        for item in _as_list(items):
            if isinstance(item, dict):
                _add(str(item.get("content") or "").strip())
    return hints


# ---- focus / 过滤 / 推荐方向 -------------------------------------------------
_NEGATIVE_EMOTIONS = frozenset(
    {
        "sad", "angry", "anxious", "fear", "fearful", "disgust", "frustrated",
        "stressed", "worried", "nervous", "irritated", "upset", "unhappy",
        "depressed", "panic", "down", "low",
    }
)
_BAD_POSTURES = frozenset({"slouching", "poor", "bad", "unhealthy", "lying", "leaning"})
_TIRED_LEVELS = frozenset(
    {"mild", "tired", "moderate", "fatigued", "weary", "sleepy", "drowsy", "high", "severe", "exhausted"}
)
_ABNORMAL_ENV_LEVELS = frozenset(
    {"high", "low", "dark", "dim", "hot", "cold", "noisy", "loud", "humid", "dry", "bright"}
)


def _resolve_focus(
    context_type: str, current_state: dict, *, user_query: str | None = None
) -> str:
    if context_type == "wellness_care":
        return _infer_wellness_focus(current_state)
    if context_type == "behavior_distraction":
        return "distraction"
    if context_type == "environment_care":
        return _infer_environment_focus(current_state)
    if context_type == "speech":
        return _infer_speech_focus(user_query or "")
    return "none"


def _infer_wellness_focus(current_state: dict) -> str:
    emotion = str(current_state.get("emotion") or "").strip().lower()
    posture = str(current_state.get("posture") or "").strip().lower()
    fatigue = str(current_state.get("fatigue_level") or "").strip().lower()
    if emotion in _NEGATIVE_EMOTIONS:
        return "emotion"
    if fatigue in _TIRED_LEVELS:
        return "fatigue"
    if posture in _BAD_POSTURES:
        return "posture"
    return "fatigue"


def _infer_environment_focus(current_state: dict) -> str:
    for key in ("light_level", "noise_level", "temperature_level", "humidity_level"):
        if str(current_state.get(key) or "").strip().lower() in _ABNORMAL_ENV_LEVELS:
            return key.replace("_level", "")
    return "environment"


# 仅当用户**亲口**提到时才在语音回复里走疲劳/情绪关怀方向；视觉感知不驱动 speech。
_SPEECH_FATIGUE_UTTERANCE = (
    "累", "疲惫", "疲倦", "困", "瞌睡", "打盹", "没精神", "乏力", "撑不住", "想睡", "犯困",
)
_SPEECH_EMOTION_UTTERANCE = (
    "难过", "伤心", "郁闷", "烦", "烦躁", "焦虑", "紧张", "低落", "不开心", "沮丧", "压力大",
    "委屈", "生气", "恼火", "崩溃", "绷着", "难受", "emo",
)


def _infer_speech_focus(user_query: str) -> str:
    text = str(user_query or "").strip().lower()
    if not text:
        return "general"
    if any(kw in text for kw in _SPEECH_EMOTION_UTTERANCE):
        return "emotion"
    if any(kw in text for kw in _SPEECH_FATIGUE_UTTERANCE):
        return "fatigue"
    return "general"


def _filter_candidates_for_context(
    candidates: list[dict[str, Any]], *, context_type: str, focus: str, current_state: dict | None = None
) -> list[dict[str, Any]]:
    """按场景职责边界裁剪候选，避免越权推荐。"""

    current_state = current_state or {}

    if context_type == "environment_care":
        # 环境关怀只围绕环境，不推荐任何放松/运动/兴趣类候选。
        return []
    if context_type == "behavior_distraction":
        # 分心提醒只用专注/工作策略，不推荐娱乐/运动放松。
        kept = [
            c
            for c in candidates
            if c["category"] == "focus_strategy" or c["memory_type"] in {"habit", "work_style"}
        ]
        return _sorted_candidates(kept)
    return _sorted_candidates(candidates)


def _sorted_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 主要按置信度排序；care_strategy 只在**同置信度**时轻微加权，
    # 不再无条件压过更高置信度的候选（否则唯一的低置信 care_strategy 会每轮霸榜）。
    def _key(item: dict[str, Any]) -> float:
        conf = float(item.get("confidence") or 0.0)
        tiebreak = 0.05 if item.get("category") == "care_strategy" else 0.0
        return conf + tiebreak

    return sorted(candidates, key=_key, reverse=True)[:5]


def _pick_recommended_angle(
    candidates: list[dict[str, Any]],
    *,
    context_type: str,
    focus: str,
    recently_used: list[str],
    focus_active: bool = False,
) -> tuple[str, str]:
    """从候选 + 场景里选一个本轮建议方向，尽量避开最近用过的角度。"""

    recent = set(recently_used)

    if context_type == "environment_care":
        return (focus or "environment"), "环境关怀只围绕当前异常的环境因素"

    if context_type == "behavior_distraction":
        for angle in ("refocus", "phone_away", "short_pause_then_focus"):
            if angle not in recent:
                return angle, "分心提醒优先收回注意力，避开最近用过的角度"
        return "refocus", "分心提醒收回注意力"

    # wellness_care：**以关怀本身为主**（rest / empathy / posture），内容候选只是可选点缀，
    # 避免每轮都推同一个爱好（例如反复「讲笑话」）。只有当该关怀角度最近刚用过时，
    # 才换一个未用过的候选类别来增加多样性。
    if context_type == "wellness_care":
        default = {"emotion": "empathy", "posture": "posture", "fatigue": "rest"}.get(focus, "rest")
        if default not in recent:
            return default, "以关怀本身为主（rest/empathy/posture），内容候选仅作可选点缀，不要每轮重复"
        for cand in candidates:
            angle = cand["category"]
            if angle == "focus_strategy" and not focus_active:
                continue
            if angle and angle != "unknown" and angle not in recent:
                return angle, f"该关怀角度近期已用，改用候选 {angle} 增加多样性"
        return default, "以关怀本身为主"

    # speech：优先用候选类别作为角度，尽量避开最近用过的。
    if candidates:
        for cand in candidates:
            angle = cand["category"]
            if angle and angle != "unknown" and angle not in recent:
                return angle, f"结合用户记忆/资料中的 {angle}（{cand['source']}）"
        first = candidates[0]
        return first["category"], f"结合用户记忆/资料：{first['label'][:20]}"

    return "general", "普通对话，无需强个性化"


# ---- 等级 / 语气 / 角度历史 --------------------------------------------------
def _personalization_level(
    *,
    context_type: str,
    use_memory: bool,
    candidates: list[dict[str, Any]],
    personalization_candidates: list[dict[str, Any]] | None = None,
) -> str:
    if not use_memory or context_type == "environment_care":
        return "none"
    active = personalization_candidates if context_type == "wellness_care" else candidates
    if not active:
        return "none"
    if any(float(c.get("confidence") or 0.0) >= 0.7 for c in active):
        return "strong"
    return "light"


def _preferred_tone(
    context_type: str,
    focus: str,
    avoid_patterns: list[dict[str, Any]],
    style_hints: list[str],
) -> str:
    short = bool(avoid_patterns) or context_type in {"behavior_distraction", "environment_care"}
    short = short or focus == "posture"
    gentle = any("温和" in h or "朋友" in h for h in style_hints) or focus == "emotion"
    if short and gentle:
        return "gentle_short"
    if short:
        return "short"
    if gentle:
        return "gentle"
    return "natural"


_REASON_TO_ANGLE = {
    "rest_reminder": "rest",
    "emotion_reminder": "empathy",
    "posture_reminder": "posture",
    "distraction_reminder": "refocus",
    "environment_warning": "environment",
}


def _recent_angles(runtime: dict) -> list[str]:
    angles: list[str] = []
    for item in _as_list(runtime.get("recent_reminders")):
        if not isinstance(item, dict):
            continue
        angle = _REASON_TO_ANGLE.get(str(item.get("reason") or ""))
        if angle and angle not in angles:
            angles.append(angle)
    return angles


# ---- 归类与工具 --------------------------------------------------------------
def _categorize(text: str, tags: list[str]) -> str:
    """优先用 tags 归类，其次用极少量通用语义关键词兜底；都不命中则 unknown。"""

    haystack = " ".join([str(text or "")] + [str(t) for t in tags]).lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if _contains_any(haystack, keywords):
            return category
    return "unknown"


def _contains_any(haystack: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in haystack for kw in keywords)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]
