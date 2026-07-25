# ESKD Nursing Note Standardizer Instructions

## Project objective

Build a small, public-safe supervised fine-tuning (SFT) showcase for a synthetic ESKD nursing-note standardization workflow.

## Current priority

Keep the project focused on the local SFT before/after comparison. Do not add retrieval features, deployment, or complex UI work unless the user explicitly asks.

## Required design principles

- Use synthetic data only. Never add client, patient, proprietary, or personally identifiable data.
- Describe the work as an experiment or reference lab; never claim clinical validity, production readiness, HIPAA compliance, or autonomous clinical decision-making.
- Treat SFT as a hypothesis. A small synthetic experiment may demonstrate the training and evaluation pipeline, but cannot prove real domain performance.
- Require human review for all case outputs.
- Prefer a few cohesive documents over a large documentation tree.

## Reference projects

- `../lhs/sft-eskd-poc/`: private SFT decision material; use only for conceptual reference. Do not copy confidential names, data claims, or client-specific content.

## Proposed stages

1. Define the synthetic nursing-note use case and output format.
2. Design synthetic training, validation, and held-out evaluation data.
3. Run the selected small model before fine-tuning.
4. Run a minimal local QLoRA experiment.
5. Compare format validity, evidence support, missing-information handling, and consistency.

## Expected response behavior

When discussing an implementation choice, focus on why SFT is suitable for this narrow, repeatable formatting task. Avoid claims of clinical validity or general domain performance.
