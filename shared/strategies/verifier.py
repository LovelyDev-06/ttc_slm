"""
Strategy 5 — Verifier.

Per the assignment guide: "Use AST parsing to determine whether generated
code is structurally valid, followed by execution against the relevant
test cases to determine functional correctness."

This module provides TWO things:

1. `heuristic_score()` — a fast, rule-based quality score (no training
   needed) used by tree search to rank/prune partial code paths where
   running the full test suite isn't possible yet (the function isn't
   finished). Combines: AST validity, presence of a return statement,
   no obvious placeholder text ("...", "pass  # TODO"), and length
   sanity.

2. `LearnedVerifier` — a small trained classifier (logistic regression
   over hand-crafted code features) that predicts P(candidate passes the
   hidden tests) WITHOUT running them. This is what train_verifier.py
   trains, and what the learned router (router.py) uses as one of its
   input features — a "genuinely learned", not hand-coded, quality signal.

Both stage-1 (AST) and stage-2 (execution) checks referenced in the guide
live in src/code_utils.py (`is_valid_python`, `run_tests`); this file is
about turning those raw signals into a single SCORE for ranking, not just
a boolean.
"""

import pickle
import numpy as np
from src.code_utils import is_valid_python, run_tests


PLACEHOLDER_MARKERS = ["...", "# TODO", "pass  # implement", "NotImplementedError"]


def heuristic_score(code: str) -> float:
    """Returns a score in [0, 1]. Cheap, no execution — used for ranking
    PARTIAL (incomplete) code during tree search expansion."""
    if not code.strip():
        return 0.0

    score = 0.0
    if is_valid_python(code):
        score += 0.5
    else:
        # partial code is often syntactically incomplete (e.g. mid-statement)
        # by design during tree search — give partial credit if it at least
        # doesn't contain placeholder junk
        score += 0.1

    if "return" in code:
        score += 0.2

    if not any(marker in code for marker in PLACEHOLDER_MARKERS):
        score += 0.2

    # mild length penalty for absurdly short or absurdly long partials
    n_lines = code.count("\n") + 1
    if 1 <= n_lines <= 60:
        score += 0.1

    return min(score, 1.0)


def extract_features(code: str) -> np.ndarray:
    """Hand-crafted features for the learned verifier. Kept simple and
    interpretable on purpose — this is a small logistic regression, not
    a neural net, so feature engineering matters more than model capacity."""
    valid_ast = 1.0 if is_valid_python(code) else 0.0
    n_lines = code.count("\n") + 1
    has_return = 1.0 if "return" in code else 0.0
    has_placeholder = 1.0 if any(m in code for m in PLACEHOLDER_MARKERS) else 0.0
    n_ifs = code.count("if ")
    n_loops = code.count("for ") + code.count("while ")
    n_defs = code.count("def ")
    length_chars = len(code)

    return np.array([
        valid_ast, n_lines, has_return, has_placeholder,
        n_ifs, n_loops, n_defs, length_chars,
    ], dtype=np.float32)


class LearnedVerifier:
    """Thin wrapper around a scikit-learn LogisticRegression trained by
    scripts/train_verifier.py. Falls back to heuristic_score if no trained
    model has been loaded yet (so the pipeline still runs before you've
    trained anything)."""

    def __init__(self, sklearn_model=None):
        self.model = sklearn_model

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(sklearn_model=model)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def score(self, code: str) -> float:
        if self.model is None:
            return heuristic_score(code)
        feats = extract_features(code).reshape(1, -1)
        # probability of class "1" = passes tests
        return float(self.model.predict_proba(feats)[0, 1])


def verify_final(code: str, test: str, entry_point: str, timeout_s: int) -> dict:
    """The 'real' ground-truth verifier used at the END of a strategy
    (not for ranking partials): stage 1 AST validity, stage 2 execution.
    This is just a thin re-export of code_utils.run_tests for readability
    at call sites, e.g. `from src.strategies.verifier import verify_final`."""
    return run_tests(code, test, entry_point, timeout_s=timeout_s)
