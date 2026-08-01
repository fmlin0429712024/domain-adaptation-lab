#!/usr/bin/env python3
"""Promotion gate and version registry for the ESKD QLoRA adapter.

Implements steps 7-9 of the nine-step lifecycle in fine-tuning-advanced-reference.md
(register/approve, deploy, operate/rollback) as a small, dependency-free CLI, in
place of a managed pipeline tool (Vertex Pipelines / Kubeflow). See "Why a CLI
registry instead of Vertex Pipelines" in that document for the reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "outputs" / "registry.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"active_version": None, "versions": [], "events": []}


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def next_version(registry: dict) -> str:
    return f"v{len(registry['versions']) + 1}"


def run_gate(comparison: dict, thresholds: dict) -> dict:
    checks = {
        "min_mean_required_sections_after": comparison["mean_required_sections_after"]
        >= thresholds["min_mean_required_sections_after"],
        "max_outputs_with_thinking_after": comparison["outputs_with_thinking_after"]
        <= thresholds["max_outputs_with_thinking_after"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def cmd_promote(args: argparse.Namespace) -> None:
    evaluation = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())["evaluation"]
    comparison = json.loads((ROOT / "outputs" / "comparison.json").read_text())
    gate = run_gate(comparison, evaluation["promotion_gate"])

    adapter_path = ROOT / "outputs" / "adapters" / "adapters.safetensors"
    registry = load_registry()
    version = next_version(registry)
    entry = {
        "version": version,
        "status": "approved" if gate["passed"] else "rejected",
        "created_at": datetime.now(UTC).isoformat(),
        "adapter_path": str(adapter_path.relative_to(ROOT)),
        "adapter_sha256_16": sha256_file(adapter_path),
        "base_model": yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())["model"]["id"],
        "training_config": yaml.safe_load((ROOT / "configs" / "training.yaml").read_text()),
        "data_sha256_16": {name: sha256_file(ROOT / "data" / f"{name}.jsonl") for name in ("train", "valid", "test")},
        "eval_summary": {key: value for key, value in comparison.items() if key != "results"},
        "gate": gate,
        "git_commit": git_commit(),
        "notes": args.notes,
    }
    registry["versions"].append(entry)
    registry["events"].append({"type": "promote", "version": version, "at": entry["created_at"], "result": entry["status"]})
    if gate["passed"]:
        registry["active_version"] = version
    save_registry(registry)

    print(json.dumps(entry, indent=2))
    if not gate["passed"]:
        raise SystemExit(f"\nPromotion gate FAILED for {version}; active version unchanged. See 'gate.checks' above.")


def cmd_rollback(args: argparse.Namespace) -> None:
    registry = load_registry()
    target = next((v for v in registry["versions"] if v["version"] == args.to), None)
    if target is None:
        raise SystemExit(f"Unknown version: {args.to}")
    if target["status"] != "approved":
        raise SystemExit(f"{args.to} was never approved by the gate; cannot roll back to it.")

    previous = registry["active_version"]
    registry["active_version"] = args.to
    registry["events"].append(
        {
            "type": "rollback",
            "from": previous,
            "to": args.to,
            "at": datetime.now(UTC).isoformat(),
            "reason": args.reason,
        }
    )
    save_registry(registry)
    print(f"Active version rolled back: {previous} -> {args.to}")


def cmd_show(_args: argparse.Namespace) -> None:
    registry = load_registry()
    summary = [
        {"version": v["version"], "status": v["status"], "created_at": v["created_at"], "notes": v["notes"]}
        for v in registry["versions"]
    ]
    print(json.dumps({"active_version": registry["active_version"], "versions": summary, "events": registry["events"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser("promote", help="Run the promotion gate against outputs/comparison.json and register the result.")
    promote.add_argument("--notes", default="")
    promote.set_defaults(func=cmd_promote)

    rollback = subparsers.add_parser("rollback", help="Roll the active version back to a prior approved version.")
    rollback.add_argument("--to", required=True, help="Version id to roll back to, e.g. v1")
    rollback.add_argument("--reason", default="")
    rollback.set_defaults(func=cmd_rollback)

    show = subparsers.add_parser("show", help="Print the active version and full promotion/rollback history.")
    show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
