#!/usr/bin/env bash
# 始终用共享环境 Python 运行，避免项目 .venv 覆盖 PATH。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_PY="/opt/ai-envs/shared/bin/python"
if [[ ! -x "$SHARED_PY" ]]; then
  echo "未找到共享环境: $SHARED_PY" >&2
  exit 1
fi
cd "$ROOT"
exec "$SHARED_PY" "$@"
