# Project Brief

## Decision framework

| Need | Preferred approach |
|---|---|
| General capability or a small bounded task | Prompting |
| Current, changing, or citable knowledge | RAG |
| Stable domain language, approved review behavior, or repeated structured output | Supervised fine-tuning |
| Numeric prediction from historical structured signals | Conventional supervised ML |

## Case-study boundary

The case study assembles a synthetic evidence packet containing longitudinal observations, summarized notes, and an approved risk signal. It produces a structured review brief for a human reviewer. The output must distinguish observed evidence from uncertainty and must request review when evidence is insufficient.

## Evaluation questions

The comparison should answer:

1. Does the output match the required structure and approved taxonomy?
2. Does it ground its statements in the supplied evidence?
3. Does it correctly abstain when the case lacks sufficient evidence?
4. Does RAG use current reference material with traceable citations?
5. Does SFT improve consistency or efficiency relative to the same base model and a prompt/RAG baseline?

## Success boundary

The lab succeeds when it demonstrates a reproducible and honest technology-selection process. It does not need to establish clinical efficacy or replace human expertise.
