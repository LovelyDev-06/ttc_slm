# Code Track — Test-Time Compute Scaling for Small LMs

Partner B deliverable for the "Test-Time Compute Scaling for Small Language Models" project.
Domain: **Code** — Qwen2.5-1.5B-Instruct & Qwen2.5-7B-Instruct on **HumanEval** and **MBPP**.

This repo implements the six shared strategies end-to-end, independently of the Math
(Partner A) and Reasoning (Partner C) tracks, per the Team Work Assignment Guide — plus
three extras: **AST normalization**, **FLOP/token accounting**, and a **learned latent
router** (a small trained network, not a hand-written if/else).

If you've never worked with a repo like this before, read section **"0. If you're new to
this"** below before anything else — it explains what every piece is actually doing.

---

## 0. If you're new to this

You don't need to understand transformers internals to run this. Here's the mental model:

- **The model** (Qwen2.5) is a program that reads a coding problem and writes Python code.
  It's not deterministic by default — ask it twice and you can get two different answers
  (that's `temperature` — 0 = always the single most likely answer, >0 = some randomness).
- **A "strategy"** is just a different way of *using* that model to get a better answer:
  ask once (greedy), ask N times and pick the best (best-of-n), ask N times and go with
  whatever most of them agree on (self-consistency), build the answer piece-by-piece and
  backtrack from dead ends (tree search), and so on.
- **The verifier** is how we decide if a piece of code is "good" — for code, that's
  unambiguous: does it parse, and does it pass the benchmark's unit tests?
- **The router** decides, per-problem, which strategy to spend compute on. Easy problems
  don't need expensive tree search; hard ones might. We're training a small neural net to
  make that call automatically instead of hand-picking rules.
- **FLOPs** (floating point operations) is just "how much compute did this cost". More
  candidates / deeper search = more FLOPs = (hopefully) higher accuracy. The whole point of
  this project is to plot accuracy against FLOPs and see which strategy gives the best
  accuracy per unit of compute.

Every script below is a normal Python CLI program: you run it from a terminal (or a
Kaggle/Colab notebook cell with a `!` prefix), it prints progress, and it writes a results
file. Nothing here requires you to write new code to get started — just follow Quickstart.

---

## Strategies implemented

| # | Strategy         | Script                                                   | Code-domain adaptation |
| - | ---------------- | --------------------------------------------------------- | ----------------------- |
| 1 | Greedy Decoding  | `scripts/run_greedy.py`                                    | single deterministic generation |
| 2 | Best-of-N        | `scripts/run_best_of_n.py`                                  | picks the candidate that passes tests (falls back to valid-AST candidate) |
| 3 | Tree Search      | `scripts/run_tree_search.py`                                | expands code search paths and prunes with a lightweight ranking signal |
| 4 | Router           | `scripts/train_router.py` + `scripts/run_router.py`          | **learned latent router**: MLP with a latent bottleneck over problem embeddings; labels trade correctness against FLOP cost |

## Repo layout

```
code-track-ttc-scaling/
├── configs/config.yaml        # model names, sampling params, paths, router/runtime verification settings
├── src/
│   ├── data_utils.py          # HumanEval / MBPP loading + prompt formatting
│   ├── code_utils.py          # code extraction, AST normalization, subprocess test execution
│   ├── model_utils.py         # model/tokenizer loading (HF transformers)
│   ├── hub_utils.py           # push checkpoints/results to HF Hub
│   ├── flop_utils.py          # FLOP/token accounting (used by every strategy)
│   └── strategies/
│       ├── greedy.py
│       ├── best_of_n.py
│       ├── self_consistency.py
│       ├── tree_search.py
│       ├── verifier.py        # heuristic score + LearnedVerifier (logistic regression)
│       └── router.py          # LatentRouterNet (MLP) + dispatch logic
├── scripts/                   # thin CLI entry points, one per strategy, plus:
│   ├── train_verifier.py      # trains the pass-probability classifier
│   ├── train_router.py        # labels problems by running all strategies, trains the router
│   └── evaluate_all.py        # aggregates logs -> results table + FLOPs-vs-accuracy plot
└── logs/, checkpoints/        # created at runtime, gitignored
```

## What you can reuse directly from Partner A's math-track repo

Not everything here needs to be written from scratch — some files are genuinely
domain-agnostic:

| File | Reuse from Partner A? |
| --- | --- |
| `src/hub_utils.py` | **Yes, copy verbatim.** Pure HF Hub upload plumbing, no math/code logic at all. |
| `src/flop_utils.py` | **Yes, copy verbatim** (and consider pulling it into a shared package — all three tracks should compute FLOPs the same way for the paper's plots to be comparable). |
| `src/model_utils.py` | Structurally identical — copy it and just swap the model name dict for Qwen. |
| `scripts/evaluate_all.py` | Mostly reusable — the plotting/aggregation logic doesn't care about the domain, only the filename regex needs to match your model/dataset names. |
| `configs/config.yaml` | Same top-level shape (`models:`, `paths:`, `hub:`) — copy the skeleton, replace the domain-specific sections. |
| `src/data_utils.py`, `src/code_utils.py` (≈ their `answer_utils.py`), all of `src/strategies/` | **No — these are genuinely domain-specific** (AST parsing and code execution vs. numeric answer extraction and regex checking) and are written fresh in this repo. |

---

## Quickstart on Kaggle

1. New Kaggle Notebook → enable a GPU accelerator (T4 x2 or P100).
2. Add your Hugging Face token as a Kaggle **Secret** named `HF_TOKEN` (needed for pushing
   checkpoints/results; Qwen2.5 is ungated so you don't need it for downloading the model).
3. Clone this repo into the notebook:

```
!git clone https://github.com/<your-username>/code-track-ttc-scaling.git
%cd code-track-ttc-scaling
!pip install -r requirements.txt -q
```

4. Log in to HF Hub:

```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(UserSecretsClient().get_secret("HF_TOKEN"))
```

5. Edit `configs/config.yaml`: change `hub.repo_id` to your own HF username/repo.

6. Run stage by stage, cheapest first (this also validates your setup early):

```
!python scripts/run_greedy.py --model qwen1_5b --dataset humaneval --limit 20
```

Drop `--limit` once you've confirmed it runs, and repeat for `qwen7b`, `mbpp`, etc.

7. Each run writes results to `logs/<strategy>_<model>_<dataset>.jsonl` and pushes to your
   HF Hub repo (see `configs/config.yaml` → `hub:`).

## Order of operations (recommended)

1. `run_greedy.py` — cheapest, gives your baseline pass-rate + FLOPs-per-problem reference.
2. `run_best_of_n.py` and `run_self_consistency.py` — both just need repeated sampling, no
   training required first.
3. `train_verifier.py` — trains the pass-probability classifier on a **train** split
   (MBPP has one; see the docstring in that script for why HumanEval doesn't and what to do
   instead). This is needed for better tree-search ranking and feeds the router's cost/benefit
   labeling.
4. `run_tree_search.py` — most expensive per-problem, do this once your compute budget allows.
   Works without a trained verifier too (falls back to a cheap heuristic score) — just less
   accurately ranked.
5. `train_router.py` — the big one: runs greedy, best_of_n, AND tree_search on a small
   training subset (default 60 problems — keep this small, it's expensive) to label each
   problem with "which strategy was best for its cost", then trains the latent router MLP
   on problem embeddings to predict that label.
6. `run_router.py` — uses the trained router to route each **test** problem to a strategy
   automatically, no manual strategy selection.
7. `scripts/evaluate_all.py` — aggregates all logs into one results table + a
   FLOPs-vs-accuracy plot for the paper.

## Compute budgeting knobs

All in `configs/config.yaml`:

- `best_of_n.n_candidates`
- `self_consistency.n_samples`
- `tree_search.branching_factor`, `tree_search.beam_width`, `tree_search.max_depth`
- `models.max_new_tokens`
- `router.cost_penalty_lambda` — higher = router favors cheap strategies more aggressively

Start small (`n=4`, `limit=20`) to sanity-check correctness before scaling up to full
HumanEval (164 problems) / MBPP test split — these runs are what generate your
FLOPs-vs-accuracy curve for the paper.

## The three extras, in one paragraph each

**AST normalization** (`src/code_utils.py::normalize_ast`) — two code candidates that only
differ in variable names or comments are "the same solution" for voting purposes. We parse
each candidate into a Python AST, rename local variables to `v0, v1, v2...` in order of
appearance, strip docstrings, and hash the result. Self-consistency (strategy 3) votes on
this hash, not on raw generated text.

**FLOP/token accounting** (`src/flop_utils.py`) — every strategy logs a `FlopRecord` per
problem: prompt tokens, generated tokens, number of generations, and any extra forward
passes (e.g. verifier re-scoring). FLOPs are estimated as `2 × num_params × total_tokens`
(the standard inference-FLOPs approximation for a dense transformer). `evaluate_all.py`
turns this into the compute-vs-accuracy plot everyone will want in the paper.

**Learned latent router** (`src/strategies/router.py`) — rather than a hand-written
"if function looks hard, use tree search" rule, `train_router.py` actually runs every
candidate strategy on a small labeled set, picks whichever won on accuracy-minus-cost, and
trains a small MLP (sentence-embedding → hidden layer → **latent bottleneck** → strategy
classifier) to predict that label from the problem text alone. At inference the net has
never seen the strategy outputs — it's predicting difficulty from the problem statement,
the same way you'd eyeball a problem and guess "this one needs more thought."

## Definition of done (per the assignment guide)

- [ ] All six strategies run successfully on both models (1.5B, 7B) and both benchmarks (HumanEval, MBPP)
- [ ] AST-based execution validation works reliably (checked via `run_tests` returning consistent pass/fail)
- [ ] Learned verifier trained and evaluated (validation accuracy reported by `train_verifier.py`)
- [ ] Learned latent router trained and evaluated (routing distribution + accuracy reported by `run_router.py`)
- [ ] FLOP/token accounting recorded for every strategy run
- [ ] Checkpoints (verifier, router) and results pushed to your own HF Hub repo and recoverable
- [ ] `evaluate_all.py` produces a results table + FLOPs-vs-accuracy plot ready for paper integration
