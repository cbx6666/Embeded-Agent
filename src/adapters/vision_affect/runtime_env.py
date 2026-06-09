from __future__ import annotations

"""MediaPipe / TensorFlow 运行时环境配置与健康探活。

在部分 ARM 嵌入式板（如 Ascend davinci-mini）上，导入 MediaPipe / TensorFlow 时
可能在 C 层触发 ``CollectiveRegistry::Register`` 并 ``Aborted (core dumped)``。
这是 SIGABRT，Python 的 try/except 无法捕获，会直接杀死整个进程。

因此这里不在本进程内试加载，而是用 **独立子进程** 探活：子进程崩溃只影响它自己，
主 Agent（语音 / 行为 / LLM）据此跳过视觉、继续运行。
"""

import os
import subprocess
import sys

_RUNTIME_ENV_OVERRIDES = {
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "GLOG_minloglevel": "2",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "MEDIAPIPE_DISABLE_GPU": "1",
}

_PROBE_OK_MARKER = "VISION_RUNTIME_OK"

_PROBE_SCRIPT = (
    "import mediapipe as mp\n"
    "fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)\n"
    "fm.close()\n"
    f"print('{_PROBE_OK_MARKER}')\n"
)


def configure_ml_runtime_env() -> None:
    """进程级 ML 运行时环境（可安全多次调用）。"""

    for key, value in _RUNTIME_ENV_OVERRIDES.items():
        os.environ.setdefault(key, value)


def vision_runtime_healthy(timeout_sec: float = 90.0) -> tuple[bool, str]:
    """在子进程中试加载 MediaPipe FaceMesh，判断视觉运行时是否可用。

    返回 ``(healthy, detail)``；``detail`` 在失败时给出原因，便于打印提示。
    子进程若 SIGABRT（如 TF CollectiveRegistry 崩溃），只会让子进程退出，
    不影响主进程。
    """

    configure_ml_runtime_env()
    env = dict(os.environ)
    env.update(_RUNTIME_ENV_OVERRIDES)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"探活超时（>{timeout_sec:.0f}s）"
    except Exception as exc:  # noqa: BLE001 - 探活本身失败也按不可用处理
        return False, f"探活子进程启动失败：{exc}"

    if proc.returncode == 0 and _PROBE_OK_MARKER in (proc.stdout or ""):
        return True, "ok"

    if proc.returncode < 0:
        signal_num = -proc.returncode
        return False, (
            f"探活子进程被信号 {signal_num} 终止（疑似 TensorFlow/MediaPipe 在板端崩溃，"
            "如 CollectiveRegistry Aborted）"
        )

    tail = (proc.stderr or "").strip().splitlines()[-3:]
    return False, f"探活子进程退出码={proc.returncode}；stderr: {' | '.join(tail)}"
