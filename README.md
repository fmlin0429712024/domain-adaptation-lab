# ESKD Fine-Tuning & MLOps Lab

A local, synthetic-data demonstration of the full small-model lifecycle: fine-tune it, prove it
works, register and gate the result, serve it, and know precisely what a managed pipeline platform
(Vertex AI Pipelines / Kubeflow) would automate on top of that. No client data, no proprietary
logic, not a clinical system.

```mermaid
flowchart LR
    A[Train + evaluate<br/>Part 1] --> B[Register + gate<br/>Part 1]
    B --> C[Serve<br/>Part 2]
    C --> D[Automate + monitor<br/>Part 3 — conceptual]
    D -.feedback.-> A
```

| Part | Answers | Status |
|---|---|---|
| [1 — Training](training/README.md) | Can a small model learn a narrow task? | **Built and run** — real QLoRA adapter, held-out before/after comparison, promotion gate + version registry |
| [2 — Serving](serving/README.md) | Can the trained model be served like a product? | **Documented** — Ollama serving mechanics and the local/production maturity ladder |
| [3 — Automation](serving/automation-pipeline.md) | What would a managed pipeline platform add on top of Parts 1 and 2? | **Conceptual mapping** — every hand-built piece named against its Vertex Pipelines/Kubeflow equivalent, deliberately not run |

## Why three parts, one repo

Each part answers a different question in the same lifecycle, so they live together instead of as
separate projects. Part 1 is real, executed work. Part 2 and 3 are honestly labeled by how far they
go — Part 2 is designed but not yet run end to end; Part 3 is intentionally conceptual, because a
single-adapter, single-machine demo has none of the scale problems a managed orchestrator exists to
solve. See [training/docs/concept-review.md](training/docs/concept-review.md) for the author's own
recall notes on the underlying design decisions.
