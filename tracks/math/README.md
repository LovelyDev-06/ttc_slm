# Math Track — Test-Time Compute Scaling for Small LMs

Domain: **Mathematics** — Llama-3.2-1B-Instruct on **GSM8K** (canonical paper setup: one model + one dataset). Hub: `swasa26/ttc-slm-math`.

Implements Greedy, Best-of-N, Self-Consistency, Tree Search, a domain verifier, and a learned latent router with FLOP accounting, checkpointing, and inference masking.

Router (same method, all tracks): L2-normed MiniLM embedding → `128` hidden → `32` latent → 4 strategies, `lr 0.002, epochs 200, MAX_WEIGHT 2.0`, `80/20 train/val split seed 42`, `--seed` sampling, no dropout. Logs report `majority/random` baselines plus `train_acc/train_bal/val_acc/val_bal`; best checkpoint picked on `val_acc`.

```bash
python scripts/train_router.py --model llama1b --dataset gsm8k --limit 200 --seed 42 --fresh_net
```

Math-specific utilities normalize final answers (including `\\boxed{...}` forms) while the router and strategy framework remain structurally aligned with the Code and Reasoning tracks.
