#!/usr/bin/env python3
"""
Validate and prepare LocalStoryChat Gold training examples.

The Gold files use chat-style JSON:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Some early files contain literal newline control characters inside JSON
strings. Python's json parser can read these with strict=False. This script
then writes every Gold file back as canonical JSON and builds a JSONL dataset.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
GOLD_DIR = ROOT / "gold"
DATASET = ROOT / "gold_dataset.jsonl"


def load_gold(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw, strict=False)

    if not isinstance(data, dict):
        raise ValueError("root must be an object")

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing or empty 'messages' list")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] is not an object")
        if msg.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{i}] has invalid role")
        if not isinstance(msg.get("content"), str):
            raise ValueError(f"messages[{i}] content is not a string")

    return data


def main() -> int:
    files = sorted(GOLD_DIR.glob("*.json"))
    print(f"Found {len(files)} Gold files")

    if not files:
        print("ERROR: no Gold files found", file=sys.stderr)
        return 1

    records = []
    errors = []

    for path in files:
        try:
            data = load_gold(path)

            # Rewrite the source file as valid, canonical JSON.
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            records.append(data)
            print(f"OK   {path.name}")
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    print()

    if errors:
        print("ERRORS:")
        for error in errors:
            print(f" - {error}")
        return 1

    with DATASET.open("w", encoding="utf-8") as out:
        for data in records:
            out.write(json.dumps(data, ensure_ascii=False) + "\n")

    user_messages = sum(
        1
        for data in records
        for msg in data["messages"]
        if msg["role"] == "user"
    )
    assistant_messages = sum(
        1
        for data in records
        for msg in data["messages"]
        if msg["role"] == "assistant"
    )
    response_words = sum(
        len(msg["content"].split())
        for data in records
        for msg in data["messages"]
        if msg["role"] == "assistant"
    )

    print(f"All {len(records)} Gold files passed validation.")
    print(f"Built: {DATASET}")
    print(f"Examples: {len(records)}")
    print(f"User messages: {user_messages}")
    print(f"Assistant messages: {assistant_messages}")
    print(f"Assistant response words: {response_words}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
