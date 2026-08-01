#!/usr/bin/env python3
"""Validate the local JSONL data shape and split identifiers."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 36, "valid": 8, "test": 12}
HEADINGS = ("Documented observations", "Care or access items", "Information to confirm", "Human-review note")


def main() -> None:
    all_ids: set[str] = set()
    for split, expected_count in EXPECTED.items():
        rows = [json.loads(line) for line in (ROOT / "data" / f"{split}.jsonl").read_text().splitlines()]
        assert len(rows) == expected_count, f"{split}: expected {expected_count}, found {len(rows)}"
        for row in rows:
            messages = row["messages"]
            assert [message["role"] for message in messages] == ["system", "user", "assistant"]
            case_id = messages[1]["content"].splitlines()[0]
            assert case_id not in all_ids, f"duplicate case id: {case_id}"
            all_ids.add(case_id)
            target = messages[2]["content"]
            assert all(heading in target for heading in HEADINGS), f"missing heading in {case_id}"
    print(f"Validated {len(all_ids)} fully synthetic examples across three splits.")


if __name__ == "__main__":
    main()
