#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs
uv run mlx_lm.lora \
  --model mlx-community/Qwen3-1.7B-4bit \
  --train \
  --data data \
  --iters 250 \
  --batch-size 1 \
  --num-layers 8 \
  --mask-prompt \
  --adapter-path outputs/adapters
