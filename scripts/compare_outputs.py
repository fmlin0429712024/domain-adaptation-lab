#!/usr/bin/env python3
"""Create a compact, reviewer-friendly baseline versus fine-tuned comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADINGS = ("Documented observations", "Care or access items", "Information to confirm", "Human-review note")


def score(text: str) -> int:
    return sum(bool(re.search(rf"(?m)^{re.escape(heading)}$", text)) for heading in HEADINGS)


def has_thinking(text: str) -> bool:
    if "<think>" not in text:
        return False
    content = text.split("<think>", maxsplit=1)[1].split("</think>", maxsplit=1)[0]
    return bool(content.strip())


def main() -> None:
    baseline = json.loads((ROOT / "outputs" / "baseline.json").read_text())["cases"]
    fine_tuned = json.loads((ROOT / "outputs" / "fine_tuned.json").read_text())["cases"]
    rows = []
    for before, after in zip(baseline, fine_tuned, strict=True):
        rows.append(
            {
                "case_id": before["case_id"],
                "required_sections_before": score(before["generated"]),
                "required_sections_after": score(after["generated"]),
                "thinking_before": has_thinking(before["generated"]),
                "thinking_after": has_thinking(after["generated"]),
                "baseline": before["generated"],
                "fine_tuned": after["generated"],
                "expected": after["expected"],
            }
        )
    summary = {
        "cases": len(rows),
        "mean_required_sections_before": sum(row["required_sections_before"] for row in rows) / len(rows),
        "mean_required_sections_after": sum(row["required_sections_after"] for row in rows) / len(rows),
        "outputs_with_thinking_before": sum(row["thinking_before"] for row in rows),
        "outputs_with_thinking_after": sum(row["thinking_after"] for row in rows),
        "results": rows,
    }
    (ROOT / "outputs" / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in summary if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
