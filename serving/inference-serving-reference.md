# Inference Serving — Reference

## The maturity ladder

Ollama and vLLM are not combined in a typical deployment — they sit at different rungs of the
same ladder, each matched to a different deployment target. You pick the rung that matches
where you're deploying, not both at once.

```text
1. Local experimentation       transformers / raw Python scripts
                                (Part 1 of this project: mlx_lm.generate())
                                        │
2. Local / single-user serving  Ollama, LM Studio, llama.cpp server
                                ← this lab
                                        │
3. Production / multi-tenant    vLLM, TGI, SGLang, TensorRT-LLM
   serving                      ← data-center GPUs, many concurrent users
                                        │
4. Multi-model management       Triton, BentoML, LiteLLM Proxy, KServe
                                ← governs many instances of rung 3
```

Popular content demonstrates one rung at a time, matched to its audience: hobbyist "run an LLM
on my laptop" videos show rung 2; MLOps/infra content shows rung 3; enterprise platform content
shows rung 4. Seeing all of them side by side in one demo is not how it's actually done in
practice.

## Ollama and vLLM: same value proposition, different scale

Both answer the identical question — *load a model, expose an API so it can be called* — for
different environments:

| | Ollama | vLLM |
|---|---|---|
| Runtime | llama.cpp, GGUF quantized models | Custom CUDA kernels, PagedAttention |
| Target | Single machine, single/few users | GPU cluster, many concurrent users |
| Continuous batching / multi-tenant optimization | Not the point | The entire reason it exists |
| Apple Silicon | Native, well supported | Unofficial/experimental; no meaningful throughput story on CPU |

This lab runs Ollama because it matches the actual hardware (a CPU Mac). Running vLLM here
would not demonstrate its value proposition — that value is entirely about GPU-scale
concurrency, which doesn't exist in a single-machine CPU demo. Choosing the tool that matches
the deployment target, rather than the tool that sounds more advanced, is the point.

## What Ollama's serving mechanics actually do

- **Prefill vs. decode**: the prompt is processed once in parallel (compute-bound); each output
  token is then generated one at a time, re-reading the growing KV-cache (memory-bandwidth-bound).
- **KV-cache**: every attention layer caches prior tokens' key/value projections so decode
  doesn't recompute them. Per-token size = `2 × num_layers × num_kv_heads × head_dim ×
  dtype_bytes`; total footprint scales with sequence length. At the single-request scale this
  lab runs at, cache size is small and not a constraint — it becomes the limiting resource
  only under many concurrent requests, which is rung 3's problem, not rung 2's.

## What vLLM adds at production scale (not run here — documented for interview fluency)

- **PagedAttention**: treats the KV-cache like OS virtual memory — fixed-size pages, a
  per-request page table, pages freed and reused immediately when a request finishes. Removes
  the fragmentation that comes from reserving one contiguous worst-case block per request.
- **Continuous (in-flight) batching**: at every decode step, finished requests are evicted and
  new ones admitted into the same running batch, instead of waiting for a fixed batch to fully
  finish before starting the next one. This is what keeps GPU utilization high under real,
  variable traffic.
- **Multi-LoRA serving**: one base model can serve several LoRA adapters concurrently, routing
  each request to its adapter with near-zero switching cost. This is the most direct bridge to
  Part 1's QLoRA adapter — the adapter trained there is exactly the kind of artifact this
  feature exists to serve, if this were deployed at production scale.

## Model management: Triton isn't the only option, and isn't chosen yet

Model management tools solve a different problem than serving engines: not "run this model,"
but "which model/version is running, and how is that governed." This lab doesn't need one yet
— there is exactly one adapter, so there's nothing to manage across. The landscape, for when a
second model or version exists:

| Tool | Core value | Weight / PoC-friendliness |
|---|---|---|
| **Triton Inference Server** | One repository managing multiple model *frameworks* (TensorRT, ONNX, PyTorch, Python backend — including a vLLM backend); versioned model directories, dynamic batching config | Heavier: Docker, a defined `model_repository/` layout, `config.pbtxt` per model. Its management plane doesn't require a GPU, only whichever backend is plugged in does. |
| **BentoML** | Packaging and serving multiple models with versioning | Lighter: pure Python, runs directly on a laptop |
| **LiteLLM Proxy** | One unified (often OpenAI-compatible) API in front of many existing model endpoints/providers; routing, fallback | Lightest: `pip install` + a YAML config; increasingly popular as a gateway layer |

Triton is the name most associated with "model management" in enterprise job descriptions, and
is the right one to know deeply for that reason. Whether it, or a lighter alternative, actually
gets implemented in this lab depends on what the second model/version turns out to be — that
decision is deliberately deferred, not skipped.

## Interview statement

> I treat inference tooling as a ladder matched to deployment scale, not a toolbox where more
> tools shown means more expertise. I run Ollama here because it's the correct tool for a
> single-machine demo; I can explain PagedAttention and continuous batching in detail without
> having run vLLM myself, because that value proposition only exists at GPU-cluster scale. The
> same discipline applies to model management: Triton is the name most enterprises use, but I'd
> pick the tool that matches the actual number and diversity of models in production, not the
> most recognizable name.
