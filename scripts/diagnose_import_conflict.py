"""诊断视觉运行时崩溃：逐个在独立子进程里跑“真实场景”，定位崩溃触发点。

纯 import 组合已全部 OK，所以这里测更接近应用的动作：创建 FaceMesh、后台线程、
pygame 初始化、tensorflow 与 mediapipe 同进程等。某场景崩溃（SIGABRT/core dump）
只影响它自己的子进程，脚本继续测下一项，最后汇总。

用法（板端 (shared) 环境）：
    python scripts/diagnose_import_conflict.py
"""

from __future__ import annotations

import os
import subprocess
import sys

# 每个场景：(名称, 代码片段)。片段结尾必须 print('OK') 才算通过。
_FACEMESH = (
    "import mediapipe as mp\n"
    "fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)\n"
    "fm.close()\n"
)

SCENARIOS: list[tuple[str, str]] = [
    ("tensorflow 然后 import mediapipe", "import tensorflow\nimport mediapipe\nprint('OK')\n"),
    ("mediapipe 然后 import tensorflow", "import mediapipe\nimport tensorflow\nprint('OK')\n"),
    ("创建 FaceMesh", _FACEMESH + "print('OK')\n"),
    (
        "创建 FaceMesh + 跑一帧",
        _FACEMESH.replace("fm.close()\n", "")
        + "import numpy as np\n"
        + "fm.process(np.zeros((480,640,3),dtype=np.uint8))\n"
        + "fm.close()\nprint('OK')\n",
    ),
    (
        "后台线程里创建 FaceMesh（贴近应用）",
        "import threading\n"
        "err = {}\n"
        "def work():\n"
        "    try:\n"
        "        import mediapipe as mp\n"
        "        fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)\n"
        "        fm.close()\n"
        "    except BaseException as e:\n"
        "        err['e'] = repr(e)\n"
        "t = threading.Thread(target=work); t.start(); t.join()\n"
        "print('OK' if not err else 'THREAD_ERR:' + err['e'])\n",
    ),
    (
        "onnxruntime + 后台线程 FaceMesh",
        "import onnxruntime\n"
        "import threading\n"
        "def work():\n"
        "    import mediapipe as mp\n"
        "    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)\n"
        "    fm.close()\n"
        "t = threading.Thread(target=work); t.start(); t.join()\n"
        "print('OK')\n",
    ),
    (
        "torch + 后台线程 FaceMesh",
        "import torch\n"
        "import threading\n"
        "def work():\n"
        "    import mediapipe as mp\n"
        "    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)\n"
        "    fm.close()\n"
        "t = threading.Thread(target=work); t.start(); t.join()\n"
        "print('OK')\n",
    ),
    (
        "pygame 显示 + 创建 FaceMesh",
        "import os\n"
        "os.environ.setdefault('SDL_AUDIODRIVER','dummy')\n"
        "import pygame\n"
        "pygame.init()\n"
        "try:\n"
        "    pygame.display.set_mode((320,240))\n"
        "except Exception as e:\n"
        "    print('display skip', e)\n"
        + _FACEMESH
        + "print('OK')\n",
    ),
    (
        "应用真实顺序: onnxruntime->torch->线程FaceMesh",
        "import onnxruntime\n"
        "import torch\n"
        "import threading\n"
        "def work():\n"
        "    import mediapipe as mp\n"
        "    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)\n"
        "    import numpy as np\n"
        "    fm.process(np.zeros((480,640,3),dtype=np.uint8))\n"
        "    fm.close()\n"
        "t = threading.Thread(target=work); t.start(); t.join()\n"
        "print('OK')\n",
    ),
    (
        "refine_landmarks=True 创建+跑一帧（疑似真凶）",
        "import mediapipe as mp\n"
        "import numpy as np\n"
        "fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)\n"
        "for _ in range(5):\n"
        "    fm.process(np.zeros((480,640,3),dtype=np.uint8))\n"
        "fm.close()\nprint('OK')\n",
    ),
    (
        "应用真实顺序 + refine_landmarks=True（最贴近应用）",
        "import onnxruntime\n"
        "import torch\n"
        "import threading\n"
        "def work():\n"
        "    import mediapipe as mp\n"
        "    import numpy as np\n"
        "    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)\n"
        "    for _ in range(5):\n"
        "        fm.process(np.zeros((480,640,3),dtype=np.uint8))\n"
        "    fm.close()\n"
        "t = threading.Thread(target=work); t.start(); t.join()\n"
        "print('OK')\n",
    ),
]

_RUNTIME_ENV = {
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "GLOG_minloglevel": "2",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "MEDIAPIPE_DISABLE_GPU": "1",
}


def run_scenario(code: str) -> tuple[str, str]:
    env = dict(os.environ)
    env.update(_RUNTIME_ENV)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "超时(>180s)"
    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and out.endswith("OK"):
        return "OK", ""
    if proc.returncode < 0:
        return "CRASH", f"被信号 {-proc.returncode} 终止(core dump/abort)"
    tail = (proc.stderr or "").strip().splitlines()[-2:]
    extra = f" | stdout={out}" if out and "OK" not in out else ""
    return f"FAIL({proc.returncode})", " | ".join(tail) + extra


def main() -> None:
    print(f"python = {sys.executable}")
    print(f"version = {sys.version.split()[0]}")
    print("=" * 72)
    results: list[tuple[str, str, str]] = []
    for name, code in SCENARIOS:
        status, detail = run_scenario(code)
        flag = "✓" if status == "OK" else "✗"
        print(f"[{flag}] {name:<42} {status}  {detail}")
        results.append((name, status, detail))

    print("=" * 72)
    bad = [r for r in results if r[1] != "OK"]
    if not bad:
        print("结论：所有场景都 OK —— 崩溃可能还需 acl OM 设备上下文等更多状态，请把输出发回。")
    else:
        print("以下场景异常（即崩溃触发点）：")
        for name, status, detail in bad:
            print(f"  - {name}  ({status}) {detail}")


if __name__ == "__main__":
    main()
