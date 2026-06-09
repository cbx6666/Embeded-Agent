# Memory

## 职责

结构化用户记忆：由 LLM **异步**从 `speech_recognized` 言谈中抽取多维度用户画像。

- `memory_model.py`：`MemoryItem`（id/user_id/type/content/evidence/confidence/tags/
  source_event/created_at/updated_at/last_used_at）。`type` 闭集见 `MEMORY_TYPES`
  （preference / hobby / habit / emotion_pattern / work_style / interaction_style /
  care_strategy / dislike / fact）。
- `memory_extractor.py`：`MemoryExtractor` 只做**一次** `memory_extract` LLM 调用 +
  schema 校验，无 critic/consolidator/validator 多阶段链路。
- `prompts/memory_extract.md`：抽取 prompt（只抽长期有用信息、保留 evidence、跳过琐事/
  敏感隐私，输出稳定 JSON）。
- `memory_service.py`：
  - **写入**：`submit_speech_memory` 入后台队列，主链路不等待 memory LLM；后台线程
    抽取 → 按类型/内容/标签合并去重 → 落盘，超 `max_memories_per_user`（默认 100）按最旧淘汰。
  - **读取**：`retrieve_user_context(user_id, query, context_type, top_k)` 用轻量打分
    （类型权重 + 标签命中 + 内容命中 + 最近使用 + confidence）返回结构化、分组后的记忆。

存储：`data/memory/user_memory.json`（策略见 `MemoryPolicy`，含各 context_type 的类型权重）。

## context_type 检索优先级

只为真实主链路 context_type 配置权重（见 `MemoryPolicy.type_weights`）：

- `speech`：preference / interaction_style / work_style / hobby / care_strategy / dislike / fact
- `wellness_care`：care_strategy / emotion_pattern / hobby / preference / habit / work_style / dislike
- `behavior_distraction`：work_style / habit / dislike / interaction_style / care_strategy
- `environment_care`：preference / dislike（环境相关）/ habit / work_style / interaction_style
- `sensor_status_report`：不检索 Memory、不调用 LLM、不生成 memory_usage_hints。

## memory_usage_hints

每次 LLM 调用前，`src/agent/context/memory_usage_hints.py` 把 profile + memory + runtime +
当前状态临时整理成「本次如何使用记忆」的策略（**不落盘**），由 `AgentCore._user_context()`
注入 `user_context["memory_usage_hints"]`。它只做整理/筛选/归类/限制，不写死爱好枚举，
不生成最终文案。

关键字段：

- `recommended_angle`：本轮主方向。wellness 以**关怀本身**为主（rest/empathy/posture），
  内容兴趣只作点缀，避免每轮都推同一个。
- `recommended_content`：本轮要点缀的**那一个兴趣**，由 `rotation_seed` 在所有兴趣候选间**轮换**
  （讲笑话→打篮球→听相声…）。`rotation_seed` 即 `state.interaction.care_rotation_index`，每次带兴趣的
  关怀/回复后自增；姿态/环境场景或无兴趣记忆时为 `null`。
- prompt 要求把第三人称记忆改写成第二人称口语，且只用本轮 `recommended_content`、不自行换、不重复上一轮。

专注结束（`timer_finished` → `complete_focus`）也复用同一套：`AgentCore` 用 `focus_complete_care.md`
生成一句带轮换兴趣的个性化关怀，LLM 失败时回退到默认完成文案。

测试：`pytest tests/test_memory_usage_hints.py tests/test_memory_llm.py`；完整流程见 `docs/testing/agent_functional_test_guide.md` §5。

## 不负责

- 旧四角色决策链 / 旧 observe/extract/critic/consolidate 多阶段记忆管线（已删除）
- 同步阻塞主响应链路（memory LLM 永远异步）
- 敏感隐私、医疗/政治/宗教等属性
