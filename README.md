# Embeded-Agent

Embeded-Agent 是一个面向嵌入式交互场景的 LLM-centered Agent 原型。当前架构把 LLM 放在认知核心，把确定性代码放在验证、边界、执行和持久化位置。

## 核心能力

- 统一建模 `Event / State / Intent / Action`
- 四角色 LLM Agent：SituationAnalyst、IntentPlanner、SafetyCritic、ResponseWriter
- LLM-managed Memory：观察、提取、审查、整合，再由代码验证和持久化
- Deterministic Boundary：schema validation、Guard、ActionRealizer、DeviceAdapter
- 面向专注辅助场景的计时、提醒、状态反馈和 trace 调试

## 主链路

```text
Event
-> Reducer
-> ProfileSnapshot / Memory
-> AgentContextBuilder
-> LLMAgentOrchestrator
-> IntentPlanValidator
-> DeterministicGuard
-> ActionRealizer
-> DeviceAdapter
```

LLM 负责理解、推理、规划、记忆候选和表达；代码负责验证、边界、状态更新、动作落地、设备执行和持久化。

## 项目结构

- `src/agent/`：Agent 主链路、状态、动作、决策、记忆和运行时。
- `src/adapters/`：CLI、语音、显示、视觉/情绪等输入输出适配。
- `src/services/`：LLM、短期 memory、计时器、用户 profile 服务。
- `src/storage/`：JSON 持久化。
- `docs/`：架构、集成和需求文档。
- `tests/`：新架构测试。

## 文档入口

- `docs/llm_centered_architecture.md`
- `docs/llm_memory_architecture.md`
- `src/agent/README.md`

## LLM 配置

`LLMService` 支持 OpenAI 风格 Chat Completions 接口，并读取 `.env` 或环境变量：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

未配置密钥时会使用本地 mock，保证测试和离线开发可运行。mock 不是主要智能来源，所有输出仍会经过同一套 validator 和 guard。

## 运行

```bash
python -m src.main
python -m src.agent_lab
```

## 测试

```bash
python -m pytest -q
```
