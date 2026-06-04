from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect runtime trace JSON.")
    parser.add_argument("trace_path", help="Path to trace JSON, usually trace_logs.json or one trace object.")
    parser.add_argument("--stage", default=None, help="Optional stage filter.")
    parser.add_argument("--json", action="store_true", help="Print filtered raw JSON.")
    args = parser.parse_args()

    data = json.loads(Path(args.trace_path).read_text(encoding="utf-8"))
    traces = _normalize_traces(data)
    rows = []
    for trace_index, trace in enumerate(traces, start=1):
        for event in trace.get("events", []):
            if not isinstance(event, dict):
                continue
            if args.stage and event.get("stage") != args.stage:
                continue
            rows.append({"trace_index": trace_index, **event})

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"trace_path={args.trace_path} rows={len(rows)} stage={args.stage or '*'}")
    for row in rows:
        payload = row.get("payload", {})
        keys = ", ".join(sorted(payload)) if isinstance(payload, dict) else "-"
        print(
            f"- trace={row['trace_index']} seq={row.get('sequence')} "
            f"{row.get('stage')}:{row.get('label')} payload_keys=[{keys}]"
        )
        if row.get("stage") in {"validator", "guard", "action"}:
            print(f"  {json.dumps(payload, ensure_ascii=False, sort_keys=True)[:500]}")


def _normalize_traces(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        return [data]
    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("trace"), dict):
                result.append(item["trace"])
            elif isinstance(item, dict) and isinstance(item.get("events"), list):
                result.append(item)
        return result
    return []


if __name__ == "__main__":
    main()
