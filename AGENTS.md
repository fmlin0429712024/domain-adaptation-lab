# Domain Adaptation Lab Instructions

## Project objective

Build a small, public-safe reference lab that explains and demonstrates the decision boundary between prompting, RAG, and supervised fine-tuning (SFT). The initial case study is a synthetic longitudinal-care review workflow.

## Current priority

Start with documentation, data design, and evaluation design. Do not begin model training, dependency installation, deployment, or complex UI work unless the user explicitly asks.

## Required design principles

- Use synthetic data only. Never add client, patient, proprietary, or personally identifiable data.
- Describe the work as an experiment or reference lab; never claim clinical validity, production readiness, HIPAA compliance, or autonomous clinical decision-making.
- Treat SFT as a hypothesis. A small synthetic experiment may demonstrate the training and evaluation pipeline, but cannot prove real domain performance.
- Keep RAG for dynamic, citable knowledge. Keep SFT for stable, approved output behavior and domain language. Keep deterministic predictive models separate from SFT.
- Require human review for all case outputs.
- Prefer a few cohesive documents over a large documentation tree.

## Reference projects

- `../LLM-Playground/`: model-platform architecture, enterprise security, cost, reliability, evaluation, and delivery concepts.
- `../industrial-operations-ai-poc/`: synthetic predictive-maintenance ML and governed human-review workflow; use as a contrast case, not as a reason to force SFT.
- `../lhs/sft-eskd-poc/`: private SFT decision material; use only for conceptual reference. Do not copy confidential names, data claims, or client-specific content.

## Proposed stages

1. Define the use case, decision boundary, and safety constraints.
2. Design synthetic training, validation, and held-out evaluation data.
3. Implement a prompt baseline and a small RAG baseline.
4. Run a minimal small-model LoRA/QLoRA SFT experiment only when approved.
5. Compare format validity, evidence grounding, abstention, consistency, cost, and latency.

## Expected response behavior

When discussing an implementation choice, explain why prompting, RAG, SFT, or conventional ML is appropriate. Avoid recommending fine-tuning merely because it is available.
