# Math Track — Test-Time Compute Scaling for Small LMs

Domain: **Mathematics** — Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct on **GSM8K** and **Hendrycks MATH**.

Implements Greedy, Best-of-N, Self-Consistency, Tree Search, a domain verifier, and a learned latent router with FLOP accounting, checkpointing, and inference masking.

Math-specific utilities normalize final answers (including `\\boxed{...}` forms) while the router and strategy framework remain structurally aligned with the Code and Reasoning tracks.
