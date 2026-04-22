# 接入说明（给不负责内核的同学）

本文说明：**你不负责内核业务逻辑时**，要改哪里、交什么文档，方便**写内核的同学**以后在 `reducer` / `policy` / `core` 里接你的契约。  
整体思路一句话：**用 Adapter 把真实世界变成标准 `Event`，或把标准 `Action` 变成真实输出；联调时再调用 `handle_event`（内核未接好前，事件可能暂时不改变状态，属预期）。**

## 现阶段边界（重要）

- **请你们专注**：**`Event` / `Action` 契约**（`src/agent/event/types.py`、`src/agent/action/types.py`、可选 **`factories.py`**）、**`src/adapters/`**、**`docs/<主题>_integration.md`**，以及只测适配器/纯函数的用例。  
- **现阶段请不要改内核**：不要在本阶段 PR 里修改 **`src/agent/reducer.py`**、**`policy.py`**、**`core.py`**、**`state/`**、**`services/memory_service.py`** 等（除非内核负责人明确开口让你改）。新增状态字段、怎么归约、发什么 `Action`，由内核同学根据你们文档里的 **payload 与业务说明** 统一写。  
- **目的**：先把 **event / action 长什么样、adapter 怎么发** 定清楚，内核同学只对付一块业务逻辑，避免多人同时改内核冲突。

多人同时改 **`types.py`** 时仍要约定合并顺序或命名，避免重复 `type` 字符串。

**与内核的后续衔接**：你们合入契约与适配器后，内核同学再开 PR 补齐 **`reducer` / `policy` / `core` / 记忆** 等，把事件真正吃进状态机。

### 分层原则：检测怎么做的，要对内核「不可见」

- **底层（适配器 / 设备侧）**：摄像头、MediaPipe、EAR、PERCLOS、ResNet 推理、阈值、防抖、线程等——**全部放在 `src/adapters/`**（及你们自有的辅助模块），**不要写进** `reducer` / `policy` / `core`。  
- **对上只暴露契约**：通过标准 **`Event`**（`type` + `timestamp` + `payload`）上报**已经抽象好的事实**（例如 `user_fatigue_updated` 里的 `fatigue_level`、`perclos` 等）；内核**不得**依赖「从哪一帧怎么算出 EAR」这类实现细节。  
- **内核**：只根据**事件语义**更新状态、决定 **Action**；把「怎么检测」留在适配器，有利于换人、换模型、换阈值而不动内核。

### 与主分支 `main` 对齐内核时（建议）

在功能分支上执行 **`git diff main -- src/agent/reducer.py src/agent/policy.py src/agent/core.py src/agent/state/ src/services/memory_service.py`**，确认哪些差异是**有意提交**、哪些应回退或与 main 保持一致。  
（示例：若 **`reducer.py` 相对 `main` 无 diff**，则「事件→状态」的归约规则与 `main` 一致；情绪流记忆、`user_fatigue_updated` 字面量等通常在其它文件，以你本地 `git diff` 为准。）

---

## 一、你要做的事（建议顺序）

1. 拉取团队约定的主分支（及子模块若有）。  
2. 新建分支：`feature/<模块>-<简述>`。  
3. 在本分支里**写全你方负责的部分**：**新增或沿用的 `Event.type` / `Action.type`**（`event/types.py`、`action/types.py`）、**可选工厂**（`event/factories.py`、`action/factories.py`）、**输入/输出 Adapter**（`src/adapters/`）。**不要改** `reducer` / `policy` / `core` / `state` / `memory_service`（见上文「现阶段边界」）。  
4. **禁止**在适配器里直接改 `AgentCore.state`；应构造 **`Event`** 并调用 **`AgentCore.handle_event(event)`**（在本地或联调脚本中验证时亦如此）。  
5. 在 **`docs/`** 下新增 **`docs/<主题>_integration.md`**，正文结构必须与 **第三节模板**一致（把【】换成真实内容）。  
6. 自测：能跑则执行 `python -m unittest discover -s tests -v`；若当前主干尚未接好你方事件的内核逻辑、测试失败属预期，请在你方文档第六节写明**联调方式与示例事件**。  
7. 提 PR 时写清：涉及哪些 Event/Action、新增文件列表、**payload 约定**，并 **@ 或通知内核同学** 便于其做整合 PR。  
8. 提交前过一遍 **第四节 PR 自检**。

---

## 二、代码分工（你主要动哪里）

| 内容 | 路径 | 一般由谁写 |
|------|------|------------|
| 事件类型字面量 | `src/agent/event/types.py` | **你方**（多人时注意命名不冲突；冲突由内核或组长在合并时裁定） |
| 事件工厂（可选） | `src/agent/event/factories.py` | 你方 |
| 动作类型字面量 | `src/agent/action/types.py` | **你方**（同上） |
| 动作工厂（可选） | `src/agent/action/factories.py` | 你方 |
| 状态归约、策略、调度、记忆 rollup | `reducer.py`、`policy.py`、`core.py`、`memory_service.py`、`state/` | **仅内核同学**（你们只把契约与文档交给对方） |
| 输入 / 输出适配 | `src/adapters/*.py` | **你方** |
| 单测 | `tests/` | 你方可只测适配器/纯函数；**勿改**依赖完整状态机的用例，除非与内核同学同步 |

**禁止**：在适配器里使用**未写入** `src/agent/event/types.py` / `src/agent/action/types.py` 的 `type` 字符串。

---

## 三、说明文档模板（复制到 `docs/<主题>_integration.md`）

将 `【】` 换成实际内容，**不要删掉下面各级标题**。

```markdown
# 【业务线名称】：我们要负责的 Event、Action 和 Adapter

本文只讲**和「【一句话业务范围】」**相关的内容。  
【更细的字段、闭集、接口说明可写在本文件后续小节，或写「见 PR / 代码注释」。】

---

## 一、我们这条线要负责什么

| 类别 | 我们关心的部分 |
|------|----------------|
| **Event（事件）** | 【】 |
| **Action（动作）** | 【】 |
| **Adapter（适配器）** | 【】 |

**边界**：【算法、阈值、采样、重试等放哪一层；哪些不归你们管】

---

## 二、白话：我们在做什么、逻辑是什么

1. 【】 …（建议写 3～6 条）

整体逻辑一句话：【Adapter → Event → 内核 → Action → Adapter】

---

## 三、Event 有哪些（与本业务线直接相关）

| Event（`type`） | 作用（白话） | 典型谁产生 |
|-----------------|-------------|------------|
| 【】 | 【】 | 【】 |

（仓库里已有类型以 `src/agent/event/types.py` 为准。）

---

## 四、Action 有哪些（本业务线场景下会怎么用）

| Action（`type`） | 作用（白话） | 在本业务线中 |
|------------------|-------------|--------------|
| 【】 | 【】 | 【】 |

（仓库里已有类型以 `src/agent/action/types.py` 为准。）

---

## 五、Adapter 有哪些（作用是什么）

| Adapter | 作用（白话） | 与本业务线的关系 |
|---------|-------------|------------------|
| 【】 | 【】 | 【】 |

---

## 六、与项目组 / 内核的衔接

- 本 PR 是否**新增或修改** `Event` / `Action` 类型：【是 / 否】；若是，**列出类型名与 payload 要点**，供内核同学整合 `reducer` / `policy` / `core` 时对照。  
- **交付给内核/联调方**：关键 Event 的 payload 示例（JSON 或代码片段）、推荐上报频率（如每秒几次、是否防抖）。
```

---

## 四、PR 自检（协作者）

- [ ] 已提交 **`docs/<主题>_integration.md`**，且章节与 **第三节模板**一致  
- [ ] 所有用到的 `Event.type` / `Action.type` 已在对应 **`types.py`** 中定义  
- [ ] **本 PR 未修改** `reducer.py`、`policy.py`、`core.py`、`state/`、`memory_service.py` 等内核文件（有例外须内核同学确认）  
- [ ] 未直接改 `core.state`，仅通过 **`handle_event`** 驱动状态变更  
- [ ] 已运行测试或在本模块文档中写明手动验证步骤  

---

## 五、你需要交付给联调方的信息（建议写进 `*_integration.md`）

- 每个你方发出的 **Event**：payload 里哪些字段必填、取值范围、典型频率。  
- 每个你方消费的 **Action**：依赖哪些 `payload` 字段、执行失败时行为（丢弃 / 重试等）。
