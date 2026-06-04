"""用户认知层公开入口。

`agent/user/` 收敛三类用户相关材料，边界不同、但都围绕用户建模：

1. UserProfile / UserInfo / UserPreference：
   用户明确声明或系统明确配置的静态权威资料（姓名、年龄、显式偏好、展示设置）。
   权威来源是 UserProfileStore。

2. PersonalContext：
   面向决策层的**动态个性化上下文快照**，由 UserProfile + LongTermMemory +
   RuntimeHistory 三个来源组合生成。它不是用户资料本身，而是"这一轮决策需要
   知道哪些关于用户的信息"的只读快照。

3. PersonalContextBuilder：
   PersonalContext 的唯一构建入口。只读各来源，做冲突检测、优先级融合和 prompt
   压缩，不直接写入 store。

UserProfileService 位于 src/services/，UserProfileStore 位于 src/storage/。
本目录只保留用户认知层的模型和上下文构建器。

名字说明：
取名 `user/` 而非 `personalization/` 或 `context/`，是出于就近原则——
PersonalContext 的核心输入是 UserProfile，放在 user/ 附近便于维护来源关系。
如未来 PersonalContext 演化出更独立的语义，可迁移到 personalization/。
"""

from src.agent.user.personal_context import PersonalContext
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.agent.user.user_profile import UserInfo, UserPreference, UserProfile

__all__ = [
    "PersonalContext",
    "PersonalContextBuilder",
    "UserInfo",
    "UserPreference",
    "UserProfile",
]
