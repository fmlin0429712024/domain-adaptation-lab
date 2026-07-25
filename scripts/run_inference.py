#!/usr/bin/env python3
"""Generate and save baseline or adapter-backed outputs for the held-out test set."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("baseline", "fine_tuned"), required=True)
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args()
    if args.label == "fine_tuned" and args.adapter_path is None:
        parser.error("--adapter-path is required for fine_tuned inference")

    model, tokenizer = load(MODEL_ID, adapter_path=str(args.adapter_path) if args.adapter_path else None)
    rows = [json.loads(line) for line in (ROOT / "data" / "test.jsonl").read_text().splitlines()]
    outputs = []
    for row in rows:
        messages = row["messages"][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        outputs.append(
            {
                "case_id": messages[1]["content"].splitlines()[0],
                "generated": generated,
                "expected": row["messages"][-1]["content"],
            }
        )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    payload = {
        "label": args.label,
        "model": MODEL_ID,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": outputs,
    }
    (output_dir / f"{args.label}.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
