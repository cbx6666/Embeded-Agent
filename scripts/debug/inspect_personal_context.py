from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.event import Event
from src.agent.state import AgentState
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.services.user_profile_service import UserProfileService
from src.storage.json_store import JsonStore
from src.storage.long_term_memory_store import LongTermMemoryStore
from src.storage.user_profile_store import UserProfileStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PersonalContext and retrieval for one user/event.")
    parser.add_argument("--state", default="data/runtime/runtime_store.json", help="Runtime state JSON.")
    parser.add_argument("--memory", default="data/memory/long_term_memory.json", help="Long-term memory JSON.")
    parser.add_argument("--profiles", default="data/user/user_profiles.json", help="User profile JSON.")
    parser.add_argument("--user", default="default", help="User id.")
    parser.add_argument("--event-type", default="user_text_input", help="Event type for retrieval scoring.")
    parser.add_argument("--text", default="", help="Text query for retrieval.")
    parser.add_argument("--timestamp", type=int, default=0, help="Synthetic event timestamp.")
    parser.add_argument("--limit", type=int, default=8, help="Retrieval limit.")
    parser.add_argument("--json", action="store_true", help="Print full JSON.")
    args = parser.parse_args()

    state_dict = JsonStore(args.state).load_state_dict()
    state = AgentState.from_dict(state_dict)
    state.current_user_id = args.user
    event = Event(type=args.event_type, timestamp=args.timestamp, payload={"text": args.text})
    builder = PersonalContextBuilder(
        long_term_memory_store=LongTermMemoryStore(args.memory),
        user_profile_service=UserProfileService(UserProfileStore(args.profiles), now_fn=lambda: 0),
    )
    personal_context = builder.build(user_id=args.user, state=state, event=event)
    retrieved = personal_context.retrieve_relevant(event_type=args.event_type, text=args.text, limit=args.limit)
    payload = {"personal_context": personal_context.to_dict(), "retrieved": retrieved}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"user={args.user} event_type={args.event_type} text={args.text!r}")
    print(f"profile_items={len(personal_context.profile_items)}")
    print(f"behavior_preferences={len(personal_context.behavior_preferences)}")
    print(f"active_constraints={len(personal_context.active_constraints)}")
    print(f"uncertain_memories={len(personal_context.uncertain_memories)}")
    print(f"compression={personal_context.compression}")
    print("retrieved:")
    for item in retrieved:
        print(
            f"- source={item.get('source')} type={item.get('memory_type') or item.get('item_type')} "
            f"effective_confidence={item.get('effective_confidence')} content={item.get('content')}"
        )


if __name__ == "__main__":
    main()
