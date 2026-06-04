# WuJie 情绪 OM（Ascend NPU）

| 文件 | 用途 |
|------|------|
| `wujie_vgg19_static.om` | FER2013 VGG19 静态图，全栈默认情绪后端 `wujie-om` |

运行前需 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并确认 ACL 可导入。

默认路径见 `src/adapters/vision_affect/config.py` 中的 `DEFAULT_WUJIE_OM_MODEL`；也可用环境变量 `WUJIE_OM_MODEL` 或 `--wujie-om` 覆盖。
