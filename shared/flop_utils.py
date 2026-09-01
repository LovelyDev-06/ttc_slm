"""
flop_utils.py

Test-time-compute papers live or die by their compute-vs-accuracy curves.
This module gives every strategy a single, consistent way to report "how
much compute did this candidate/problem/strategy actually cost".

REUSE NOTE: this has ZERO code-domain logic in it. It's the same for math,
code, and reasoning. Strongly recommend copying this file into all three
repos (or better: pulling it into a 4th tiny shared package) so the final
paper's FLOPs-vs-accuracy plot is computed the same way for every track.

--------------------------------------------------------------------------
Why this specific formula
--------------------------------------------------------------------------
For a dense transformer doing a forward pass (inference, not training),
FLOPs-per-token is commonly approximated as:

    FLOPs_per_token ≈ 2 * N_params

(each parameter is touched by ~1 multiply + 1 add per token, ignoring
attention's quadratic term, which is a fine approximation for the model
sizes here — 1.5B/7B — at typical sequence lengths). Training uses 6*N
instead of 2*N because of the backward pass; we're doing pure inference
here so we use 2*N.

Total FLOPs for a generation = 2 * N_params * (prompt_tokens + generated_tokens)

We also track prompt and generated tokens SEPARATELY, because strategies
differ wildly in how they spend tokens (best-of-n spends it on parallel
generations, tree search spends it on prompt re-encoding of partial paths,
etc). This lets the final plot break down "where did the compute go" per
strategy, not just report a single number.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FlopRecord:
    strategy: str
    problem_id: str
    num_params: int
    prompt_tokens: int
    generated_tokens: int
    num_generations: int = 1        # e.g. best_of_n with n=8 -> 8
    extra_forward_passes: int = 0   # number of extra forward calls
    extra_forward_tokens: int = 0   # tokens processed by extra forward calls (e.g. LLM step judge)

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens + self.generated_tokens) * self.num_generations

    @property
    def flops(self) -> float:
        base = 2 * self.num_params * self.total_tokens
        extra = 2 * self.num_params * (self.extra_forward_passes + self.extra_forward_tokens)
        return base + extra

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "problem_id": self.problem_id,
            "num_params": self.num_params,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "num_generations": self.num_generations,
            "extra_forward_passes": self.extra_forward_passes,
            "extra_forward_tokens": self.extra_forward_tokens,
            "total_tokens": self.total_tokens,
            "estimated_flops": self.flops,
        }


class FlopLedger:
    """Accumulates FlopRecords for a whole run so scripts can dump a
    per-problem AND an aggregate (sum/mean) compute report."""

    def __init__(self):
        self.records: List[FlopRecord] = []

    def log(self, record: FlopRecord):
        self.records.append(record)

    def total_flops(self) -> float:
        return sum(r.flops for r in self.records)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def mean_flops_per_problem(self) -> float:
        if not self.records:
            return 0.0
        return self.total_flops() / len(self.records)

    def as_dicts(self) -> List[dict]:
        return [r.to_dict() for r in self.records]

    def summary(self) -> dict:
        return {
            "n_problems": len(self.records),
            "total_flops": self.total_flops(),
            "total_tokens": self.total_tokens(),
            "mean_flops_per_problem": self.mean_flops_per_problem(),
        }


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
