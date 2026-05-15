# Embeded-Agent

Embeded-Agent 是一个面向嵌入式交互场景的 LLM-centered Agent Runtime Prototype。当前实现把 LLM 放在语义认知层，把 Python 放在 deterministic runtime boundary：LLM 负责理解、规划、审查、表达和长期记忆候选；runtime 负责 state、validator、guard、store、action、trace 和 replay。

## 核心链路

```text
Event
-> Reducer
-> RuntimeHistoryService
-> LongTermMemoryPipeline
-> PersonalContextBuilder
-> AgentContextBuilder
-> LLMAgentOrchestrator
-> IntentPlanValidator
-> DeterministicGuard
-> ActionRealizer
-> DeviceAdapter
-> ActionResult
-> RuntimeTrace
```

## 核心能力

- 统一建模 `Event / State / RuntimeHistory / LongTermMemory / PersonalContext / Intent / Action`。
- 四角色 LLM Agent：`SituationAnalyst`、`IntentPlanner`、`SafetyCritic`、`ResponseWriter`。
- 长期记忆管线：observe、extract、critic、consolidate、validate、store。
- Deterministic boundary：schema validation、registered intent、presence/cooldown guard、action realization、device adapter。
- 个性化上下文：显式 `UserProfile`、证据化 `LongTermMemory` 和短期 `RuntimeHistory` 收口到 `PersonalContext`。
- 可观测性：轻量 `RuntimeTrace` 支持 debug print、JSON dump 和测试断言。
- 长时间运行验证：`tests/scenarios/`、`tests/replay/` 和 `scripts/runtime_experiments/`。

## 项目结构

```text
src/
  main.py
  adapters/
  agent/
  services/
  storage/

tests/
  scenarios/
  replay/

scripts/
  runtime_experiments/
  debug/

docs/
  design/
  integration/
  requirements/
  shared/
```

## 文档入口

- `docs/design/src_architecture_design.md`：当前 `src/` 架构设计文档。
- `docs/design/storage_layout.md`：本地 `data/` 目录的数据域布局说明。
- `docs/requirements/agent_requirements.md`：Agent 需求与目标说明。
- `docs/integration/`：行为、显示、环境、语音、情绪等适配说明。
- `src/agent/README.md`：Agent 内核目录说明。

## LLM 配置

生产 `LLMService` 使用 DeepSeek Chat Completions，并读取 `.env` 或环境变量：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

未配置 API 时，生产链路会抛出配置错误。测试和实验通过显式 fake/double 提供 deterministic LLM 行为，不由 `LLMService` 内置 mock。

## 运行

```bash
python -m src.main
```

可选视觉适配器：

```bash
python -m src.main --vision
```

## 测试

```bash
python -m pytest -q
```

## 运行实验

```bash
python scripts/runtime_experiments/study_session_experiment.py
python scripts/runtime_experiments/long_term_memory_experiment.py
python scripts/runtime_experiments/multi_user_isolation_experiment.py
python scripts/runtime_experiments/hallucination_resistance_experiment.py
python scripts/runtime_experiments/retrieval_quality_experiment.py
```

实验输出默认写入：

- runtime experiments：`data/experiments/runtime/`
- retrieval experiment：`data/experiments/retrieval/`

## Debug CLI

```bash
python scripts/debug/inspect_memory.py --memory data/memory/long_term_memory.json
python scripts/debug/inspect_personal_context.py --user default --text "gentle reminder"
python scripts/debug/inspect_trace.py data/experiments/runtime/study_session_experiment/trace_logs.json
python scripts/debug/replay_events.py data/experiments/runtime/study_session_experiment/events.json
```
