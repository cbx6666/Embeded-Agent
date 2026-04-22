# 姿势识别：我们要负责的 Event、Action 和 Adapter

本文说明姿势识别模块需要提交的事件契约、适配器实现要点和联调说明，遵循团队接入规范 `docs/team_integration_guide.md`。

---

## 一、我们这条线要负责什么

| 类别 | 我们关心的部分 |
|------|----------------|
| **Event（事件）** | `user_posture_updated` |
| **Action（动作）** | 无新增 Action（使用现有 `speak` / `display` 产生提醒） |
| **Adapter（适配器）** | `src/adapters/posture_adapter.py`（摄像头/模型输出 -> Event） |

边界：所有模型、阈值、帧率、防抖、短期统计等均放在适配器内部；内核只消费标准 Event。

---

## 二、白话：我们在做什么、逻辑是什么

1. 摄像头/视觉模型输出姿势分类（例如 `upright` / `slouch` / `lean_left` / `lean_right` 等）。
2. 适配器在本地做置信度过滤与去抖（防抖），构造 `user_posture_updated` 事件并调用 `AgentCore.handle_event`。  
3. 内核（reducer/policy）接收事件后决定是否更新 state 与产生命令（如提醒）。

整体逻辑一句话：Adapter → Event → 内核 → Action → Adapter

---

## 三、Event（与本业务线直接相关）

| Event（`type`） | 作用（白话） | 典型谁产生 |
|-----------------|-------------|------------|
| `user_posture_updated` | 上报当前识别到的姿势与置信度（便于内核记录与决策） | 摄像头适配器 / mock 命令 |
| `user_posture_summary` | 累计不良姿势达阈值后的汇总/告警事件（例如连续 slouch 超过 120s） | 适配器本地汇总后发出 |

示例事件（JSON）： 

```json
{
  "type": "user_posture_updated",
  "timestamp": 1680000000,
  "payload": {
    "posture": "slouch",
    "confidence": 0.92,
    "duration_sec": 2.0,
    "person_id": "user123",
    "keypoints_summary": {"torso_angle_deg": 18.5},
    "severity": "mild",
    "source": "camera_v1",
    "frame_id": 12345
  }
}
```

---

## 四、Action（本业务线场景下会怎么用）

| Action（`type`） | 作用（白话） | 在本业务线中 |
|------------------|-------------|--------------|
| `speak` / `display` | 提醒用户调整姿势或休息 | 内核决定何时触发（例如长时间 slouch） |

---

## 五、Adapter 有哪些（作用是什么）

| Adapter | 作用（白话） | 与本业务线的关系 |
|---------|-------------|------------------|
| `posture_adapter.py` | 把模型输出（posture + confidence）转为 `user_posture_updated` 并调用 `handle_event` | 必须实现阈值与去抖策略 |

---

## 六、与项目组 / 内核的衔接

 - 本 PR 是否新增 Event 类型：是（`user_posture_updated`、`user_posture_summary`），已在 `src/agent/event/types.py` 中声明。  
 - 交付给内核：payload 示例（见上文）、推荐上报频率/防抖策略：仅在姿势变化或同一姿势持续超过 2s 时上报；置信度门限 0.6（可调）。  
 - 建议新增 state 字段：在 reducer 中把 posture 写入 `state.user.posture`（字符串），并可保留 `state.user.posture_since` 或 `state.user.posture_history` 供 policy 使用。  
 - 建议 reducer 伪码：

```py
def reduce_state(state, event):
  if event.type == "user_posture_updated":
    state.user.posture = event.payload.get("posture")
    state.user.posture_confidence = event.payload.get("confidence")
    state.user.posture_since = event.timestamp
    # 可选：记录 duration 或其他字段到 memory
  if event.type == "user_posture_summary":
    # 记录为 incident/summary
    state.memory.recent_events.append({
      "type": event.type,
      "timestamp": event.timestamp,
      "payload": event.payload,
    })
  return state
```

- 建议 policy 伪码：

```py
def decide_actions(prev_state, state, event, llm_service):
  actions = []
  if event.type == "user_posture_updated":
    if event.payload.get("posture") in {"slouch", "lying"} and event.payload.get("duration_sec", 0) >= 30:
      actions.append(Action(type="speak", payload={"text": "你刚趴了下去，起来活动一下吧"}))
  if event.type == "user_posture_summary":
    actions.append(Action(type="speak", payload={"text": f"检测到你已连续{event.payload.get('accumulated_sec')}秒不良姿势，建议休息。"}))
  return actions
```

联调示例（Python）：

联调示例（Python）：

```py
from src.agent.core import build_default_core
from src.adapters.posture_adapter import PostureAdapter

core = build_default_core()
adapter = PostureAdapter(core)
adapter.publish_posture("slouch", confidence=0.9, frame_id=1)
```

---

调参与调试建议：
- posture 值闭集（建议）： [`upright`, `slouch`, `lean_left`, `lean_right`, `head_down`, `lying`, `unknown`]。
- 默认参数建议：置信度门限 0.6、去抖 2s、summary 阈值 120s、上报频率上限 1Hz。
- debug 模式：适配器输出 keypoints/角度可视化并保存示例帧以便人工标注与阈值调参。
- 隐私：仅上报事件/汇总，不上传原始图像；提供开关以关闭摄像头检测。

PR 描述模板（复制到 GitHub PR body）：

```
feat(posture): add posture adapter + events + docs

变更文件：
- src/agent/event/types.py (新增 user_posture_summary)
- src/agent/event/factories.py (新增 posture factory fields + summary factory)
- src/adapters/posture_adapter.py (适配器实现：阈值/去抖/summary)
- docs/posture_integration.md (集成文档)
- tests/test_posture_adapter.py (单测)

Event / payload 约定：见 docs/posture_integration.md（包含 posture 闭集、confidence、duration_sec、person_id、keypoints_summary、severity 等字段）。

联调步骤：
1. 在本地运行 `python -m unittest discover -s tests -v`。
2. 在本地启动 core：
  from src.agent.core import build_default_core
  core = build_default_core()
  from src.adapters.posture_adapter import PostureAdapter
  adapter = PostureAdapter(core)
  adapter.publish_posture("slouch", confidence=0.9, frame_id=1)

注意：本 PR 仅新增事件契约与适配器，不修改 reducer/policy/core/state/memory_service；若内核需接入请在 PR 中 @ 内核同学并参考文档中提供的 reducer/policy 伪码。
```

---

PR 自检请覆盖 `docs/team_integration_guide.md` 的第 4 节清单：types 已声明、docs 已添加、适配器与 tests 包含、未直接改内核文件等。
