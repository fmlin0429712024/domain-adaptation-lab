# Domain Adaptation Lab

A reference lab for deciding when prompting, retrieval-augmented generation (RAG), or supervised fine-tuning provides the most appropriate solution for a domain-specific task.

## Purpose

The lab uses a synthetic longitudinal-care review case study to compare three approaches:

- **Prompting:** baseline behavior from a general-purpose model.
- **RAG:** current, citable knowledge such as policies, procedures, and guidelines.
- **Supervised fine-tuning:** stable domain language, approved review patterns, and structured output behavior.

The objective is not to claim clinical performance or autonomous decision-making. It is to establish a reproducible decision framework, a small synthetic experiment, and clear evaluation criteria.

## Reference use case

A reviewer receives longitudinal observations, approved risk signals, and summarized notes. The system produces a structured review brief that highlights evidence, identifies missing information, and requires human review for any decision.

## Scope

- Synthetic data only
- Small-model SFT experiment with a held-out evaluation set
- RAG baseline using versioned reference documents
- Base-model, RAG, and SFT comparison
- Structured outputs, evidence grounding, abstention, and human-review criteria

## Out of scope

- Real patient or client data
- Autonomous clinical decisions or recommendations
- Production deployment or clinical-performance claims
- GraphRAG and complex agent orchestration

## Documentation

- [Project brief](docs/project-brief.md)
