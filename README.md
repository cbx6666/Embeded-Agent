# Embeded-Agent

Embeded-Agent 是一个面向嵌入式交互场景的事件驱动 Agent 原型工程。  
当前版本围绕专注辅助、人机交互和状态驱动决策展开，核心目标是建立一条可控、可解释、可扩展的 agent 闭环。

## 核心能力

- 统一建模 `Event / State / Intent / Action`
- 基于状态机与规则的决策链路
- 受约束的 LLM 辅助意图判断与文本生成
- 动作执行结果回流形成闭环
- 面向专注场景的提醒、计时与状态维护能力

## 项目结构

- `src/agent/`：Agent 核心决策与闭环运行层
- `src/adapters/`：输入输出适配层
- `src/services/`：LLM、记忆、计时等服务层
- `src/storage/`：状态持久化
- `tests/`：单元测试

## Agent 工作流

系统当前采用如下处理流程：

```text
Event
-> State 更新
-> 候选 Intent 生成
-> （可选）LLM 辅助意图选择
-> IntentGuard 校验
-> Action 生成
-> Action 执行
-> ActionResult 回流
```

该设计确保：

- LLM 不直接修改状态
- LLM 不直接生成动作
- 规则层始终负责安全边界
- 闭环具备明确停止条件

## LLM 配置

当前 `LLMService` 默认支持 DeepSeek 的 OpenAI 兼容接口，并支持从项目根目录 `.env` 读取配置。  
可参考 [`.env.example`](/d:/Homework/embed/project/.env.example:1)：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

如果未配置密钥，系统会自动回退到本地 mock 逻辑，以保证测试和离线开发可继续进行。

## 运行

- 基础 CLI：`python -m src.main`
- 自主化验证终端：`python -m src.agent_lab`

## 测试

当前测试可通过以下命令执行：

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```
