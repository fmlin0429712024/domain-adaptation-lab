# Fine-Tuning — Advanced Scope Reference

## What this lab already proves

This project has completed the right **foundation** for a public-safe fine-tuning showcase:

- a narrow, repeatable task: standardizing synthetic ESKD nursing-note handoffs;
- separated train, validation, and held-out test data;
- local QLoRA / LoRA adapter training on a small base model;
- baseline versus tuned comparison on unseen synthetic examples;
- an explicit human-review and non-clinical boundary.

This is a real training and evaluation workflow. It is not merely a conceptual diagram.

## The concrete MLOps story for this lab

This lab uses one mainstream fine-tuning pattern: **QLoRA**. It keeps the 1.7B base-model weights unchanged and trains a small adapter. The adapter plus the base model is the approved fine-tuned model version.

```mermaid
flowchart LR
    A[Versioned synthetic data<br/>data-v1] --> B[Training configuration<br/>run-001]
    B --> C[QLoRA training job]
    C --> D[ESKD adapter<br/>adapter-v1]
    D --> E[Held-out evaluation]
    E --> F{Promotion gate}
    F -->|Approved| G[Register<br/>eskd-handoff:v1]
    F -->|Rejected| H[Revise data or configuration]
    G --> I[One serving API<br/>POST /summarize]
    I --> J[Human review, logs,<br/>monitoring, rollback]
```

The API consumer calls one approved endpoint. It does not need to know whether the service internally uses a base model, an adapter, or a merged model file.

### The deployment mental model

```text
Before fine-tuning
  eskd-handoff:v0 = base model only

After evaluation and approval
  eskd-handoff:v1 = same base model + ESKD adapter v1

Client behavior
  POST /summarize   (unchanged)
```

The service is updated to load `v1`. The base-model files may remain cached, but the **effective model version** changes because the adapter changes the model's behavior. If `v1` fails its monitoring or human-review criteria, the service can return to `v0` or the previous approved version.

## A nine-step MLOps teaching framework

There is no single official nine-step standard. This is a practical framework for explaining the complete lifecycle of this fine-tuning use case from business need through operation.

| Step | Question | Fine-tuning artifact or decision |
|---:|---|---|
| 1. Define the business task | What repeated decision or output needs improvement? | Synthetic ESKD handoff standardization; human review remains required. |
| 2. Select the base model | What model fits capability, cost, licensing, and runtime constraints? | Small 1.7B Qwen base model for a local demonstration. |
| 3. Prepare and version data | What examples teach the task, and which examples remain unseen? | Separate synthetic train, validation, and test splits. |
| 4. Define the training recipe | Exactly how will the model be adapted? | QLoRA method, adapter settings, model revision, seed, and run configuration. |
| 5. Run and record training | What happened during the training job? | Run ID, data version, configuration, metrics, adapter artifact, and logs. |
| 6. Evaluate | Is the candidate better on held-out examples and within defined boundaries? | Before/after comparison, format checks, evidence support review, and unsafe-addition checks. |
| 7. Register and approve | Which exact artifact is permitted for use? | `eskd-handoff:v1` links base model, adapter, evaluation record, and approval. |
| 8. Deploy or serve | How do approved clients obtain the result? | One stable `/summarize` API loads the approved base-model-plus-adapter version. |
| 9. Operate and improve | What is observed after release, and what triggers a new version or rollback? | Quality signals, cost/latency, human feedback, incidents, rollback, and approved new data. |

The first six steps create evidence for a release. Steps seven through nine govern that release after training has finished.

## Inside steps 5 and 6: what one iteration does, and how a full run is monitored

The table above describes step 5 ("run and record training") and step 6 ("evaluate") at the level of an MLOps checklist. This section drops one level down: what a single training iteration actually does, and what the full run looks like end to end, using this project's real configuration in `outputs/adapters/adapter_config.json`.

### One iteration, mechanically

Each iteration processes one training example (`batch_size: 1` in this run) through five steps:

1. **Load a batch** — one nursing-note example from `data/train.jsonl`.
2. **Forward pass** — the model (frozen base weights plus the current adapter) generates its guess at the four-section summary.
3. **Compute loss** — the guess is compared, token by token, against the target summary. Because `mask_prompt: true`, loss is computed only on the completion tokens, not the input notes.
4. **Backward pass** — gradients are computed only for the adapter's rank-8 `A`/`B` matrices on the last 8 of 28 layers. The frozen base weights are excluded from this step entirely; they never receive a gradient.
5. **Optimizer step** — the adapter's weights are nudged by a small amount, sized by the learning rate (`1e-5` in this run).

This same five-step loop repeats once per iteration. Nothing changes about the loop between iteration 1 and iteration 250 except the current values of the adapter weights and which example is in the batch.

### What the full 250-iteration run looks like

| Config value | What it controls | This run's number |
|---|---|---|
| `iters` | total iterations | 250 |
| `batch_size` | examples per iteration | 1 |
| training set size | — | 36 examples |
| `steps_per_report` | how often training loss is printed | every 10 steps |
| `save_every` | how often an adapter checkpoint is written | every 100 steps |
| `steps_per_eval` | how often the validation set is scored | every 200 steps |

With `batch_size: 1` and 36 training examples, one epoch (one full pass over the training set) is 36 iterations. 250 iterations therefore covers **250 ÷ 36 ≈ 6.9 epochs** — not 250 epochs. The two checkpoint files in `outputs/adapters/` (`0000100_adapters.safetensors`, `0000200_adapters.safetensors`) are not two different adapters; they are two snapshots of the same rank-8 `A`/`B` matrices at iteration 100 and iteration 200.

### Picking a checkpoint: training loss is not the selection criterion

Training loss measures how well the current adapter reproduces the training examples it has already seen, repeatedly. It is expected to trend down, with noise, since `batch_size: 1` means every step's gradient direction comes from a single example. A falling training loss does not by itself confirm the adapter generalizes — with only 36 examples, it is entirely possible for training loss to keep falling while held-out performance plateaus or regresses.

The signal that actually matters for checkpoint selection is the held-out score in `outputs/comparison.json`, computed on `data/test.jsonl` — data the adapter never trains on — not the training loss curve. As noted earlier in this document, `compare_outputs.py` automates two of the five checks listed in `configs/evaluation.yaml` (required-section presence, absence of leaked `<think>` reasoning); the remaining three are reviewed manually on the examples shown in the README. Checkpoint selection follows that held-out signal, not training loss.

## One product, three operational lifecycles

A modern AI application can change at more than one layer. MLOps, LLMOps, and AgenticOps are useful labels for governing those different layers. They overlap in practice; the important point is to know **which asset changed and which evaluation gate applies**.

| Lifecycle | Primary changing asset | Typical examples | Primary question |
|---|---|---|---|
| **MLOps** | Training data, model weights, adapter, model artifact | QLoRA adapter v1 → adapter v2 | Does the changed model perform better and remain safe enough to promote? |
| **LLMOps** | Prompt, model selection, generation settings, evaluation policy | New system prompt; model-provider change; temperature limit | Does the model call remain accurate, reliable, safe, and cost-effective? |
| **AgenticOps** | Workflow, skills, tool permissions, routing, Markdown/YAML instructions, harness code | New agent skill; changed approval route; new tool policy | Does the workflow complete the business task with the right controls and human oversight? |

```mermaid
flowchart TB
    A[Business workflow and user feedback] --> B{What changed or failed?}
    B -->|Task behavior / domain format| C[MLOps\nData + adapter + model version]
    B -->|Prompt / model call quality| D[LLMOps\nPrompt + model selection + evaluation]
    B -->|Workflow / skill / tool behavior| E[AgenticOps\nAgent definition + harness + controls]
    C --> F[Approved application release]
    D --> F
    E --> F
    F --> G[Production signals and human feedback]
    G --> A
```

### The flywheel is not automatically model retraining

The flywheel means collecting evidence from real use, reviewing it, and improving the correct layer through an approved change process.

| Signal from use | Likely improvement path |
|---|---|
| Summaries consistently omit a domain-specific pattern | Curate approved examples and consider a new adapter version through **MLOps**. |
| Output is too verbose or ignores a formatting instruction | Improve the prompt or generation policy through **LLMOps**. |
| The system calls the wrong tool or skips a required human approval | Change the workflow, skill, or policy through **AgenticOps**. |
| New knowledge or a changing policy is missing | Update the approved knowledge source or retrieval layer; do not automatically retrain the model. |

For a traditional enterprise workflow, AgenticOps and LLMOps may be the primary focus because the business process, tools, approval path, and prompts change most often. For an AI product with repeated domain behavior, MLOps becomes more central when new approved data justifies a new model or adapter version. The same product may need all three, but each change should be evaluated and promoted through the lifecycle that matches the asset being changed.

> **Interview explanation:** I view MLOps, LLMOps, and AgenticOps as connected operational lifecycles rather than separate technologies. MLOps governs changes to data and model artifacts; LLMOps governs model calls and prompts; AgenticOps governs the workflow, skills, and tool controls. Feedback from production should be triaged to the right layer. In this lab, the central artifact is a QLoRA adapter, so the detailed lifecycle is MLOps: versioned data, training configuration, evaluation, registration, approval, serving, monitoring, and rollback.

## Scope: what this lab does and does not claim

The point is not to claim every form of fine-tuning or every production capability. The point is to be able to explain which method fits the task, what artifact it produces, how it is evaluated, how it is governed after training — and to be precise about what is adjacent but distinct.

| Topic | What it is | Status in this lab | Why distinct / next step |
|---|---|---|---|
| **QLoRA / LoRA** | A small adapter is trained while base weights remain frozen. | **Completed in this lab.** | Extend with hosted MLOps promotion controls. |
| **Instruction fine-tuning** | Examples teach a desired task or response style. | This is the task style used by the ESKD lab. | Continue improving synthetic-data diversity and evaluation. |
| **Full fine-tuning** | Some or all base-model weights are updated. | Not implemented. | Learn conceptually; use only when adapter tuning cannot meet a justified need. |
| **Embedding fine-tuning** | Train a retriever/embedding model so relevant documents rank nearer to a query. | Not implemented; a separate problem type. | Improves retrieval quality, not response format — build a separate retrieval-focused lab, with retrieval metrics. |
| **Continued pre-training / foundation-model training** | General model weights learn from a huge corpus, at large-scale compute. | Not implemented and out of scope. | Usually a model-provider responsibility; understand the category and vendor-selection implications. |
| **Real clinical-data validation** | Validate a clinical use case with approved data, clinical experts, protocol, and safety monitoring. | Not implemented. | Synthetic before/after output cannot establish clinical quality, safety, or workflow effectiveness. |
| **Hosted MLOps fine-tuning pipeline** | Reproducible cloud training, evaluation gates, registry, deployment, monitoring, and rollback. | Lifecycle contract implemented locally (`scripts/promote.py`); the managed cloud version is not implemented, only mapped. | See [../serving/automation-pipeline.md](../serving/automation-pipeline.md) for which Vertex/Kubeflow primitive replaces each local step. |

These are complementary layers, not a single maturity ladder. A strong enterprise system may use several of them, but does not need all of them.

> **Interview explanation:** This lab proves the local SFT foundation: curated splits, QLoRA adaptation, and held-out before/after evaluation on a narrow synthetic task. I distinguish that from embedding fine-tuning for retrieval, from managed MLOps for governed promotion, and from clinical validation, which needs approved real-world data and expert review. I do not collapse those claims into one.

## Recommended next learning order

Steps 7-9 of the table above (register, deploy, operate) are implemented as a promotion gate and
version registry in `scripts/promote.py` / `outputs/registry.json`. The managed-platform equivalent
(Vertex AI Pipelines / Kubeflow) is mapped primitive-by-primitive in
[../serving/automation-pipeline.md](../serving/automation-pipeline.md) — not duplicated here.

### 1. Embedding fine-tuning — learn next

Use this only when a retrieval problem is demonstrated:

```text
question
  -> embedding model
  -> vector search / reranking
  -> relevant evidence
  -> LLM answer with citations
```

The training data is usually a set of query–relevant-document pairs, often with hard negatives. The evaluation is retrieval-oriented: Recall@k, MRR, nDCG, or task success—not merely fluent generated text.

### 2. Real clinical validation — understand the boundary, do not simulate it

For a real healthcare implementation, the work would require approved data access, de-identification/minimum-necessary review, a clinical ground truth or review protocol, subgroup/safety analysis, human oversight, and post-deployment monitoring. This lab should continue to say: **synthetic data only; not clinical validation**.

### 3. Foundation-model training — know the category, not the implementation

You should be able to distinguish it from fine-tuning:

> Fine-tuning adapts an existing model to a bounded task using curated examples. Foundation-model training builds general capability from an enormous corpus. In enterprise delivery, I would typically select and govern a foundation model rather than train one from scratch.
