from __future__ import annotations

"""长期行为记忆流水线。

MemoryPipeline 统一串联：
Event/Action -> BehaviorExtractor -> BehaviorUpdater -> InsightExtractor
-> MemoryPolicy -> UserProfileService。

它不是 God Class：具体提取、统计、画像抽取、写入策略都委托给独立组件。
"""

from src.agent.action import Action
from src.agent.event import Event
from src.agent.memory.behavior.behavior_stats import BehaviorStats
from src.agent.memory.behavior.behavior_updater import BehaviorUpdater
from src.agent.memory.extractor.behavior_extractor import BehaviorExtractor, BehaviorSignal
from src.agent.memory.extractor.insight_extractor import InsightExtractor
from src.agent.memory.policy.memory_policy import MemoryPolicy
from src.agent.memory.policy.personalization_policy import PersonalizationPolicy, PersonalizedPolicy
from src.agent.state import AgentState
from src.services.user_profile_service import UserProfileService
from src.storage.behavior_store import BehaviorStore


class MemoryPipeline:
    """长期行为建模流水线入口。"""

    def __init__(
        self,
        store: BehaviorStore,
        profile_service: UserProfileService,
        *,
        behavior_extractor: BehaviorExtractor | None = None,
        behavior_updater: BehaviorUpdater | None = None,
        insight_extractor: InsightExtractor | None = None,
        memory_policy: MemoryPolicy | None = None,
        personalization_policy: PersonalizationPolicy | None = None,
    ) -> None:
        self.store = store
        self.profile_service = profile_service
        self.behavior_extractor = behavior_extractor or BehaviorExtractor()
        self.behavior_updater = behavior_updater or BehaviorUpdater()
        self.insight_extractor = insight_extractor or InsightExtractor()
        self.memory_policy = memory_policy or MemoryPolicy()
        self.personalization_policy = personalization_policy or PersonalizationPolicy(profile_service)
        self.stats_by_user = {
            user_id: BehaviorStats.from_dict(raw_stats)
            for user_id, raw_stats in self.store.load_stats().items()
        }

    def process_event(self, user_id: str | None, event: Event, state: AgentState) -> None:
        """处理一条标准事件，更新长期行为统计并尝试刷新画像。"""
        normalized_user_id = self.profile_service.ensure_user_id(user_id)
        signals = self.behavior_extractor.extract_event(event, state)
        self._process_signals(normalized_user_id, signals, timestamp=event.timestamp)

    def process_actions(self, user_id: str | None, actions: list[Action], timestamp: int) -> None:
        """处理本轮动作，记录 Agent 给出的休息建议等交互事实。"""
        normalized_user_id = self.profile_service.ensure_user_id(user_id)
        signals: list[BehaviorSignal] = []
        seen_suggestion_keys: set[tuple[str, object]] = set()
        for action in actions:
            for signal in self.behavior_extractor.extract_action(action, timestamp):
                # 同一轮提醒通常同时生成 speak 和 display，长期统计只计一次建议。
                key = (signal.type, signal.payload.get("content_type"))
                if signal.type == "break_suggestion_shown" and key in seen_suggestion_keys:
                    continue
                seen_suggestion_keys.add(key)
                signals.append(signal)
        self._process_signals(normalized_user_id, signals, timestamp=timestamp)

    def build_personalized_policy(self, user_id: str | None) -> PersonalizedPolicy:
        """生成当前用户的个性化策略快照，供 Planner/Realizer 使用。"""
        return self.personalization_policy.build(user_id)

    def get_stats(self, user_id: str | None) -> BehaviorStats:
        """读取某个用户的长期行为统计，不存在时自动创建空统计。"""
        normalized_user_id = self.profile_service.ensure_user_id(user_id)
        return self._ensure_stats(normalized_user_id)

    def _process_signals(self, user_id: str, signals: list[BehaviorSignal], *, timestamp: int) -> None:
        """统一处理行为信号，避免事件和动作两条入口重复代码。"""
        if not signals:
            return

        stats = self._ensure_stats(user_id)
        changed = False
        for signal in signals:
            changed = self.behavior_updater.update(stats, signal) or changed

        if not changed:
            return

        self._save_stats()
        self._refresh_insights(user_id, stats, timestamp=timestamp)

    def _refresh_insights(self, user_id: str, stats: BehaviorStats, *, timestamp: int) -> None:
        """从统计中抽取画像候选，并通过 MemoryPolicy 后写入 profile。"""
        profile = self.profile_service.get_user(user_id)

        decayed, decay_changed = self.memory_policy.decay_insights(profile.insights, now=float(timestamp))
        if decay_changed:
            self.profile_service.replace_insights(user_id, decayed, timestamp=timestamp)
            profile = self.profile_service.get_user(user_id)

        for candidate in self.insight_extractor.extract(stats):
            decision = self.memory_policy.evaluate(candidate, profile.insights)
            if not decision.allow_write:
                continue

            if decision.contradicted_contents:
                remaining = [
                    insight
                    for insight in profile.insights
                    if not (
                        insight.insight_type == candidate.insight_type
                        and insight.content in decision.contradicted_contents
                    )
                ]
                self.profile_service.replace_insights(user_id, remaining, timestamp=timestamp)
                profile = self.profile_service.get_user(user_id)

            self.profile_service.upsert_insight(
                user_id,
                insight_type=candidate.insight_type,
                content=candidate.content,
                confidence=candidate.confidence,
                evidence_count=candidate.evidence_count,
                timestamp=timestamp,
            )
            profile = self.profile_service.get_user(user_id)

    def _ensure_stats(self, user_id: str) -> BehaviorStats:
        """确保某个用户有独立的长期行为统计对象。"""
        stats = self.stats_by_user.get(user_id)
        if stats is None:
            stats = BehaviorStats()
            self.stats_by_user[user_id] = stats
            self._save_stats()
        return stats

    def _save_stats(self) -> None:
        """统一保存所有用户的长期行为统计。"""
        self.store.save_stats({
            user_id: stats.to_dict()
            for user_id, stats in self.stats_by_user.items()
        })
