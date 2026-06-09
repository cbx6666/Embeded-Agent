"""一锤定音：验证 MediaPipe(TF) 与 Ascend ACL OM 的加载顺序谁先谁后导致崩溃。

之前的诊断只 `import acl`、从不真正 `load` OM，所以测不出。本脚本用应用自己的
`acl_runtime.shared_om_session` 真正 `load_from_file` 一个 .om（占用 NPU 设备上下文），
再分别用两种顺序加载 MediaPipe FaceMesh，定位崩溃触发点。

每个场景跑在独立子进程：崩溃（SIGABRT/core dump）只影响该子进程，脚本继续。

用法（板端 (shared) 环境，仓库根目录）：
    python scripts/diagnose_om_mediapipe_order.py
"""

from __future__ import annotations

import os
import subprocess
import sys

_ENV = {
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "GLOG_minloglevel": "2",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "MEDIAPIPE_DISABLE_GPU": "1",
}

# 在子进程里：发现第一个 .om 并打印路径（供其它场景使用）
_FIND_OM = (
    "import glob, sys\n"
    "oms = sorted(glob.glob('models/**/*.om', recursive=True))\n"
    "print(oms[0] if oms else '')\n"
)

# 公共：真正 load 一个 OM（占用 Ascend 设备上下文）
_LOAD_OM = (
    "from src.adapters.vision_common.acl_runtime import shared_om_session\n"
    "sess = shared_om_session(OM_PATH, device_id=0)\n"
    "assert sess.load(), 'OM load 失败'\n"
    "print('OM_LOADED', OM_PATH)\n"
)

# 公共：创建 FaceMesh 并跑一帧
_FACEMESH = (
    "import numpy as np, mediapipe as mp\n"
    "fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)\n"
    "fm.process(np.zeros((480,640,3), dtype=np.uint8))\n"
    "fm.close()\n"
    "print('FACEMESH_OK')\n"
)


def _scn_order_a(om: str) -> str:
    # 应用当前顺序：先 load OM，再 MediaPipe（预期崩溃）
    return f"OM_PATH = {om!r}\n" + _LOAD_OM + _FACEMESH + "print('OK')\n"


def _scn_order_b(om: str) -> str:
    # 修复顺序：先 MediaPipe，再 load OM（预期 OK）
    return f"OM_PATH = {om!r}\n" + _FACEMESH + _LOAD_OM + "print('OK')\n"


def run(code: str, timeout: int = 180) -> tuple[str, str]:
    env = dict(os.environ)
    env.update(_ENV)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "超时"
    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and out.endswith("OK"):
        return "OK", out.replace("\n", " | ")
    if proc.returncode < 0:
        return "CRASH", f"被信号 {-proc.returncode} 终止(core dump/abort) | stdout={out}"
    tail = (proc.stderr or "").strip().splitlines()[-3:]
    return f"FAIL({proc.returncode})", " | ".join(tail) + f" | stdout={out}"


def main() -> None:
    print(f"python = {sys.executable}")
    print(f"version = {sys.version.split()[0]}")
    print("=" * 72)

    env = dict(os.environ)
    env.update(_ENV)
    found = subprocess.run(
        [sys.executable, "-c", _FIND_OM], capture_output=True, text=True, env=env, cwd=os.getcwd()
    )
    om = (found.stdout or "").strip().splitlines()[-1].strip() if found.stdout.strip() else ""
    if not om:
        print("[✗] 未在 models/**/*.om 找到任何 OM 模型，无法测试 OM+MediaPipe 顺序。")
        print("    请确认板上已有 .om（如 models/wujie/wujie_vgg19_static.om）。")
        return
    print(f"使用 OM 模型：{om}")
    print("=" * 72)

    scenarios = [
        ("仅 MediaPipe FaceMesh（基线）", _FACEMESH + "print('OK')\n"),
        (f"仅 load OM（基线）", f"OM_PATH = {om!r}\n" + _LOAD_OM + "print('OK')\n"),
        ("顺序A 应用现状: 先 OM 后 MediaPipe（疑似崩）", _scn_order_a(om)),
        ("顺序B 修复方案: 先 MediaPipe 后 OM（疑似OK）", _scn_order_b(om)),
    ]
    results = []
    for name, code in scenarios:
        status, detail = run(code)
        flag = "✓" if status == "OK" else "✗"
        print(f"[{flag}] {name:<42} {status}")
        if status != "OK":
            print(f"      └─ {detail}")
        results.append((name, status))

    print("=" * 72)
    a = next((s for n, s in results if n.startswith("顺序A")), None)
    b = next((s for n, s in results if n.startswith("顺序B")), None)
    if a == "CRASH" and b == "OK":
        print("结论：确认！先 OM 后 MediaPipe 崩，先 MediaPipe 后 OM 正常。")
        print("      => 修复方向正确：main.py 已在 OM 预载前主线程 warmup MediaPipe。")
    elif a == "OK" and b == "OK":
        print("结论：两种顺序都 OK —— 崩溃触发点不在‘OM vs MediaPipe 顺序’，需继续排查。")
    elif a == "CRASH" and b == "CRASH":
        print("结论：两种顺序都崩 —— 同进程内 TF 与 ACL OM 无法共存，需进程隔离视觉模块。")
    else:
        print(f"结论：A={a}, B={b}，请把完整输出发回分析。")


if __name__ == "__main__":
    main()
