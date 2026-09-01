# Reasoning Track — Test-Time Compute Scaling for Small LMs

Partner C deliverable for the six-strategy evaluation pipeline.

**Domain:** general reasoning and multiple-choice reasoning using **Llama-3.2-1B-Instruct** and **Qwen2.5-1.5B-Instruct** on **ARC-Challenge** and **MMLU STEM**.

The repository mirrors the shared Code Track architecture but replaces code execution/AST validation with reasoning-specific answer extraction, exact multiple-choice correctness, reasoning-quality signals, and step-level tree-search judging. The assignment guide specifies that Partner C owns an independent six-strategy pipeline for these two models and reasoning benchmarks. See the provided guide for the independent-track requirement and the six strategy definitions.

## Strategies
1. **Greedy** — one deterministic reasoning generation.
2. **Best-of-N** — multiple sampled reasoning candidates; evaluation selects a correct candidate when one is available.
3. **Self-Consistency** — multiple reasoning paths, final-answer consensus voting.
4. **Tree Search** — branching partial reasoning paths with LLM step judging and pruning.
5. **Verifier** — reasoning-oriented answer correctness and quality scoring.
6. **Learned Latent Router** — sentence embedding → hidden layer → latent bottleneck → strategy classifier.

## Domain-specific design
- `src/reasoning_utils.py`: extracts final option labels and checks exact benchmark answers.
- `src/strategies/verifier.py`: provides reasoning-chain quality signals and final answer verification.
- `src/strategies/self_consistency.py`: votes primarily on the final multiple-choice answer to avoid paraphrase fragmentation.
- `src/strategies/tree_search.py`: explores partial reasoning paths and uses the LM itself as a Yes/No step judge.
- `src/strategies/router.py`: same learned latent-router architecture as the shared code repo, including inference masking for unseen strategy classes.

## Dataset commands
```bash
python scripts/run_greedy.py --model llama1b --dataset arc_challenge --limit 20
python scripts/run_greedy.py --model qwen1_5b --dataset mmlu_stem --limit 20
```

Run order: greedy → best-of-N → self-consistency → tree search → train router → run router.

## Important research note
For router training, inspect the printed strategy-label distribution. If a strategy has **zero positive labels**, place it in `router.inference_mask_unseen` before router inference. This prevents a randomly initialized/untrained output logit from selecting an unseen class by chance.

## Shared infrastructure retained
`flop_utils.py`, checkpointing, Hugging Face Hub storage, model loading, CSV logging, and the learned latent-router implementation remain structurally aligned with the shared Code Track so results can be compared consistently during paper integration.
