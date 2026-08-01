# Automation / Pipeline Platform Layer — Part 3 of 3

> **Conceptual mapping only.** No GCP project was provisioned, no Kubeflow Pipelines (KFP) spec was
> compiled, and no Kubernetes/Argo runtime was used. This document assumes basic GCP familiarity
> (GCS, IAM, Pub/Sub, Cloud Scheduler are not re-explained) and exists to precisely name which
> managed-platform primitive would replace which manual step in [Part 1](../training/README.md) and
> [Part 2](README.md) — not to claim hands-on operation of that platform.

## The core fact these two products share

Vertex AI Pipelines and Kubeflow are two **runtimes for the same pipeline-definition language** —
the Kubeflow Pipelines SDK (KFP). Vertex Pipelines is Google's managed, serverless runtime for
KFP-defined (or TFX) pipelines: you don't operate a Kubernetes cluster. Self-managed Kubeflow is
the same KFP SDK, but you install Kubeflow Pipelines on your own GKE cluster and operate the
execution infrastructure yourself. The pipeline code is largely identical either way; what differs
is who runs the cluster underneath it.

## Three layers of the same lifecycle

1. **Concept layer** — train → deploy → monitor → adjust → retrain. Anyone can describe this.
2. **Manual practice layer** — what Parts 1 and 2 actually do: a human runs each script in order,
   reads the result, and decides whether to proceed. This is real, hands-on lifecycle experience.
3. **Platform/automation layer** — the same steps, but defined as pipeline code that a managed
   system executes, tracks, and can trigger without a human starting each run. This is what "Vertex
   Pipelines / Kubeflow experience" specifically asks about, and it's the layer this project
   deliberately does not implement — see "Why not implemented" below.

## What we hand-built, mapped to the real platform primitive

| Hand-built in this project | KFP / Vertex primitive | What it actually is |
|---|---|---|
| `train.sh`, `run_inference.py`, `compare_outputs.py`, `promote.py` as separate scripts run in order | `@dsl.component` (a KFP pipeline component) | Each script becomes a containerized, versioned function with typed inputs and outputs |
| Manually running those scripts in the right order | `@dsl.pipeline`, compiled to a pipeline spec and submitted as a run | The DAG connecting components; the platform executes them in dependency order, in parallel where possible |
| `training/outputs/registry.json` (Part 1) | Vertex AI Model Registry | Versioned model resources; supports aliases (e.g. "default") so serving always points at whichever version is aliased, without a config change |
| `adapter_sha256_16` / `git_commit` fields in the registry entry | Vertex ML Metadata (automatic lineage tracking) | Every artifact in a pipeline run — dataset, model, metric — is automatically linked to the inputs that produced it; we approximated this by hand with hashes |
| `promote.py`'s pass/fail gate check | A conditional component (`dsl.If` in KFP v2) | The pipeline only proceeds to "register" if the evaluation component's output metric clears the threshold |
| Manually reloading the adapter in Ollama (Part 2) | Vertex AI Endpoint deploying a Model Registry entry | Deployment is itself a pipeline component / API call, not a separate manual act |
| Manually re-running `promote.py` when convenient | Cloud Scheduler (time-based) or a Pub/Sub event trigger (e.g. new data lands in GCS → Cloud Function → pipeline run) | This is the flywheel's actual "automatic" part — a signal or schedule starts a new pipeline run, not a person |
| *(not built — no second model version to justify it)* | Vertex AI Model Monitoring | Watches a deployed endpoint for training/serving skew and drift in production traffic; can itself publish the alert that triggers the retraining run above |

## Why this stays conceptual, not implemented

A single-adapter, single-machine demonstration has no parallel components, no GPU cluster
scheduling, and no second model version to manage — the problems a managed orchestrator exists to
solve. Standing up a real GCP project and Vertex Pipelines run for one adapter would cost real money
and time to prove a scale problem this project doesn't have. The judgment call itself — knowing
*when* a managed orchestrator earns its cost, versus when a hand-rolled script gives the same
lifecycle guarantees for free — is the actual signal an interviewer is checking for, more often than
literal hands-on pipeline-authoring experience.

## Interview statement

> I didn't reach for Vertex Pipelines here because a single-adapter, single-Mac experiment doesn't
> have the parallelism or multi-version problem that a managed orchestrator solves. I built the same
> lifecycle contract by hand instead — a deterministic evaluation gate, an immutable versioned
> registry, and a rollback path in Part 1 — and I can point to exactly which Vertex/Kubeflow
> component each piece stands in for. I know the KFP SDK pattern — `@dsl.component`-decorated
> Python functions composed into a `@dsl.pipeline`, submitted to Vertex AI Pipelines or a
> self-managed Kubeflow install on GKE — I haven't operated it hands-on, and that's a deliberate
> scope choice, not a gap I'm unaware of.
