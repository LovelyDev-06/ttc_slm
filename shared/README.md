# Shared Core

Reference implementations shared structurally across Math, Code, and Reasoning:
model loading, FLOP accounting, checkpointing, Hub utilities, strategy interfaces,
learned latent router (`384 -> 128 -> 32 -> 4`, L2-normed embeddings, `lr 0.002`,
`epochs 200`, `MAX_WEIGHT 2.0`, `80/20 train/val`), and inference-masking support.

Canonical paper setup (one model + one dataset per track):
reasoning `llama1b + arc_challenge`, math `llama1b + gsm8k`, code `qwen1_5b + mbpp`.

Domain-specific dataset loading, answer/code/reasoning validation, prompts, and
verifiers remain inside each track. Code uses prompt-only embeddings (no choices
field in MBPP); reasoning/math append `prompt + choices` — same normalization.
