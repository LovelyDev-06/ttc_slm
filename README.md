# Adaptive Inference Routing — Three-Track Research Repository

A consolidated research repository containing three independently runnable tracks:

- **Math:** Llama-3.2 1B / 3B on GSM8K and MATH
- **Code:** Qwen2.5 1.5B / 7B on HumanEval and MBPP
- **Reasoning:** Llama-3.2 1B and Qwen2.5 1.5B on ARC-Challenge and MMLU STEM

Each track implements the same research framework:
1. Greedy decoding
2. Best-of-N
3. Self-consistency
4. Tree search
5. Domain verifier
6. Learned latent router

## Router safety

The router is a learned neural classifier with a latent bottleneck. Strategies that
receive zero positive training labels can be placed in `router.inference_mask_unseen`.
The corresponding logits are masked before strategy selection, preventing an unseen
class from being selected due to an untrained/random logit.

## Layout

`tracks/math`, `tracks/code`, and `tracks/reasoning` are independently runnable.
`shared/` contains the common reference implementations. Domain-specific evaluation
remains separate to preserve benchmark correctness and reproducibility.

## Recommended workflow

Run and validate each individual strategy first, then train the verifier/router on the
track's training data, inspect router-label distributions, configure inference masking
only for zero-label classes, and finally evaluate the router on held-out test data.
Aggregate results only after all per-track runs are complete.
