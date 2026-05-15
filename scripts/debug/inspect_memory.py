from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LongTermMemoryStore contents.")
    parser.add_argument("--memory", default="data/memory/long_term_memory.json", help="Path to long-term memory JSON.")
    parser.add_argument("--user", default=None, help="Optional user_id filter.")
    parser.add_argument("--include-inactive", action="store_true", help="Show contradicted/inactive memories.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    args = parser.parse_args()

    payload = _load_memory_payload(Path(args.memory), user_id=args.user, include_inactive=args.include_inactive)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"memory_path={args.memory} user={args.user or '*'} count={len(payload)}")
    for memory in payload:
        conflicts = []
        if memory.get("contradiction_of"):
            conflicts.append(f"contradiction_of={memory['contradiction_of']}")
        metadata = memory.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("contradicted_by"):
            conflicts.append(f"contradicted_by={metadata['contradicted_by']}")
        if isinstance(metadata, dict) and metadata.get("contradicts"):
            conflicts.append(f"contradicts={metadata['contradicts']}")
        conflict_text = f" {' '.join(conflicts)}" if conflicts else ""
        print(
            f"- {memory['id']} user={memory['user_id']} status={memory['status']} "
            f"type={memory['memory_type']} confidence={memory['confidence']:.3f} "
            f"decay={memory['decay']:.3f} evidence={len(memory['evidence'])}{conflict_text}"
        )
        print(f"  {memory['content']}")


def _load_memory_payload(path: Path, *, user_id: str | None, include_inactive: bool) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw = data.get("memories", []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return []
    memories = [dict(item) for item in raw if isinstance(item, dict)]
    if user_id is not None:
        memories = [item for item in memories if str(item.get("user_id", "")) == user_id]
    if not include_inactive:
        memories = [item for item in memories if str(item.get("status", "active")) == "active"]
    return memories


if __name__ == "__main__":
    main()
