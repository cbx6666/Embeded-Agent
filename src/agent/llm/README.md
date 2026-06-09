# LLM

## 职责

`llm/` 提供 LLM 调用封装与 prompt 构建。

| 文件 | 职责 |
|------|------|
| `client.py` | `LLMClient.complete_json(role, prompt)` |
| `prompt_builder.py` | `build_speech_prompt`、`build_behavior_distraction_prompt`、`build_wellness_prompt`、`build_environment_care_prompt` |

## 决策 role

- `speech_recognized`：用户语音理解
- `behavior_distraction_check`：玩手机分心提醒措辞
- `wellness_care_check`：疲劳 / 负面情绪 / 姿态关怀
- `environment_care_check`：环境（光照 / 温度 / 湿度 / 噪声）关怀
- `sensor_status_report`：传感器数值播报由 `SensorStatusHandler` 确定性拼接，不调 LLM

另有后台 `memory_extract`（异步记忆抽取，不阻塞主链路）。

模板在 `../prompts/`。

## 测试

注入 `tests/fakes/fake_llm_service.FakeLLMService` 作为底层 `llm_service`。
