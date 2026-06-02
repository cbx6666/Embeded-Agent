#!/usr/bin/env bash
# YOLO26 .pt -> ONNX -> .om (Huawei ATC)
# Run on Ascend board with CANN + ultralytics installed.
set -e
set -u
set -o pipefail 2>/dev/null || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SHARED_PY="${SHARED_PY:-/opt/ai-envs/shared/bin/python}"
IMGSZ="${IMGSZ:-320}"
SOC="${SOC_VERSION:-Ascend310B1}"
OUT_DIR="${OUT_DIR:-$ROOT/models/yolo26}"

pick_python() {
  if [ -x "$SHARED_PY" ] && "$SHARED_PY" -c "from ultralytics import YOLO" 2>/dev/null; then
    echo "$SHARED_PY"
    return 0
  fi
  if python3 -c "from ultralytics import YOLO" 2>/dev/null; then
    echo "python3"
    return 0
  fi
  echo "错误: 无法 import ultralytics。" >&2
  echo "  请使用共享环境: $SHARED_PY" >&2
  echo "  或修复当前 python3 的 numpy<2 / matplotlib 冲突（勿用 base conda 的 python3）。" >&2
  return 1
}

PYTHON="$(pick_python)" || exit 1
SHARED_ROOT="$(cd "$(dirname "$PYTHON")/.." && pwd)"
SHARED_SITE="${SHARED_ROOT}/lib/python3.10/site-packages"
echo "使用 Python: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.executable)'))"

setup_ascend_env() {
  if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  fi
}

setup_atc_python() {
  # ATC 内部会调 python3；conda(base) 的 NumPy 2.x 会触发 np.float_ 报错
  export PATH="${SHARED_ROOT}/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
  export PYTHONPATH="${SHARED_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE 2>/dev/null || true
  hash -r 2>/dev/null || true
  echo "ATC 使用的 python3: $(command -v python3)"
  python3 -c "import numpy; print('ATC numpy', numpy.__version__)"
  if ! python3 -c 'import numpy, sys; sys.exit(0 if int(numpy.__version__.split(".")[0]) < 2 else 1)'; then
    echo "错误: ATC 仍加载到 NumPy>=2，请先: deactivate" >&2
    echo "  并确认: which python3 => ${SHARED_ROOT}/bin/python3" >&2
    return 1
  fi
}

mkdir -p "$OUT_DIR"

if [ "${SKIP_ONNX:-0}" = "1" ]; then
  echo "==> 跳过 ONNX 导出 (SKIP_ONNX=1)，仅执行 ATC"
else
  echo "==> Export ONNX (imgsz=$IMGSZ)"
"$PYTHON" <<PY
from pathlib import Path
from ultralytics import YOLO

out_dir = Path("$OUT_DIR")
imgsz = int("$IMGSZ")
for name in ("yolo26n.pt", "yolo26n-pose.pt"):
    pt = out_dir / name
    if not pt.is_file():
        pt = Path(name)
    m = YOLO(str(pt))
    # nms=False：保留原始检测头，便于 OM + 我们自写后处理（与 yolo_ultralytics_ops 一致）
    m.export(format="onnx", imgsz=imgsz, simplify=True, opset=11, nms=False)
PY
fi

setup_ascend_env
setup_atc_python || exit 1

echo "==> ATC -> OM (soc=$SOC)"
for stem in yolo26n yolo26n-pose; do
  onnx="$OUT_DIR/${stem}.onnx"
  if [ ! -f "$onnx" ]; then
    onnx="${stem}.onnx"
  fi
  if ! command -v atc >/dev/null 2>&1; then
    echo "错误: 未找到 atc，请先 source Ascend set_env.sh" >&2
    exit 1
  fi
  atc --model="$onnx" \
    --framework=5 \
    --output="$OUT_DIR/${stem}" \
    --input_format=NCHW \
    --input_shape="images:1,3,${IMGSZ},${IMGSZ}" \
    --soc_version="$SOC"
  echo "OK: $OUT_DIR/${stem}.om"
done

echo "Done. Test: python scripts/test_phone_hand_detection.py --backend om --imgsz $IMGSZ"
