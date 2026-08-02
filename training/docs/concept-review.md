# Concept Review — daily recall, not project documentation

This page is a personal study aid: short enough to re-read in a minute, meant to be
repeated daily until it's automatic. It is deliberately separate from README.md,
which tells the project story for anyone evaluating the work. This page is scoped
to what a confident, honest verbal answer needs — not full technical depth. Items
explicitly marked "skip unless pushed" are background only; don't volunteer them.

## 30-second answer

> I ran a real local QLoRA fine-tune — took a small Qwen3-1.7B model and taught it
> to turn messy synthetic ESKD nursing notes into a consistent 4-section handoff
> summary. I versioned train/valid/test data, trained a LoRA adapter, and did a
> before/after comparison on held-out test cases — went from 0/4 required sections
> to 4/4. My main production experience is agentic/prompting-based, not fine-tuning
> — this was a deliberate exercise to make sure I actually understand the full
> training lifecycle, not just call an API.

## Quick-recall Q&A

**Why QLoRA, not full fine-tuning?**
Full fine-tuning updates all 1.7B params — needs 4x+ memory for weights + gradients
+ optimizer state, not feasible on a laptop. QLoRA freezes and 4-bit-compresses the
base model, trains only a small low-rank correction: 2.49M params, 0.145% of the
model, in my actual run.

**Why rank=8, only the last 8 of 28 layers?**
Later layers shape output format/style — what I'm changing. Earlier layers hold
general language ability — I want to keep those untouched. Rank=8 is deliberately
small because 36 training examples can't responsibly support a higher-capacity
adapter; more rank on that little data raises overfitting risk, not quality.
*(Skip unless pushed: "later = style, earlier = general" is a common heuristic, not
a proven law — the real evidence is the held-out eval, not the theory.)*

**36 examples, ~6.9 epochs — how do you know it's not memorizing?**
The 12 held-out test examples never appeared in training and use different wording.
A model that only memorized couldn't correctly reformat notes phrased in ways it's
never seen — but it produced all 4 required sections on 12/12 unseen cases, with
content grounded in each specific note, not copy-pasted. That's generalization
evidence, not proof of memorization avoidance.
Say the caveat out loud, don't skip it: 36 synthetic examples proves the training
and evaluation *mechanics* are sound — it is not a claim of broad real-world
generalization.

**Why a small local model, not a large hosted LLM?**
Two reasons, both worth saying, not just the first one: (1) practical — this is a laptop-scale
demo, no GPU cluster or API budget assumed. (2) architectural — that constraint mirrors a real
production pattern: for a narrow, repeated task, a small fine-tuned model is often cheaper and
faster per call than paying per-token for a large general model at volume, so this isn't just a
workaround, it's a legitimate choice class. The MLOps lifecycle I'm demonstrating — versioned
data, gate, registry, rollback — is identical whether the base model is 1.7B or 70B; in a resourced
environment I'd evaluate base-model size as a cost/latency/quality tradeoff against the specific
task, not default to "biggest available."

## Depth control — do not volunteer unless directly pushed

- Adam optimizer internals (why gradients + optimizer state cost extra memory)
- Why 4-bit specifically vs. 8-bit quantization
- LoRA's "intrinsic rank" theory / the original paper's math
- Exact epoch math beyond "~7 passes over 36 examples"

If asked, it's fine to say "I know that's a layer deeper than I've gone — happy to
reason through it live" rather than fake precision.
