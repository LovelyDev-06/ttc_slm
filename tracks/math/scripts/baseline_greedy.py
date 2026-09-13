#!/usr/bin/env python
"""
Standalone greedy-decoding baseline over a dataset sample, for comparing
against the router's accuracy on the identical problems.

The repo's existing scripts/run_greedy.py has the Reasoning track's model
and dataset choices (llama1b/qwen1_5b, arc_challenge/mmlu_stem), not math's
-- likely mixed up during the multi-track consolidation -- so it can't run
GSM8K/MATH correctly. This script reuses the same underlying pieces
(src/strategies/greedy.py, src/data_utils.py, src/model_utils.py) that
scripts/train_router.py already exercises successfully, so it's known-good.

Example:
    python scripts/baseline_greedy.py --model llama1b --dataset gsm8k --limit 200 --seed 42
"""
import argparse, csv, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_dataset
from src.model_utils import load_model_and_tokenizer
from src.strategies.greedy import run_greedy
from src.flop_utils import FlopLedger

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llama1b", "llama3b"])
    p.add_argument("--dataset", required=True, choices=["gsm8k", "math"])
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default="configs/config.yaml")
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)

    problems = load_dataset(a.dataset, split=a.split, limit=a.limit, seed=a.seed)
    model, tokenizer, num_params = load_model_and_tokenizer(a.model, cfg)

    ledger = FlopLedger()
    results = []
    for i, problem in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {problem['problem_id']}: greedy")
        r = run_greedy(model, tokenizer, num_params, [problem], cfg, ledger)[0]
        results.append(r)

    n_correct = sum(bool(r["passed"]) for r in results)
    acc = n_correct / max(len(results), 1)
    print(f"\nGreedy baseline | {a.model} | {a.dataset} | acc={acc:.2%} ({n_correct}/{len(results)})")
    print(f"FLOP summary: {ledger.summary()}")

    tag = len(problems)
    out_csv = os.path.join(cfg["paths"]["logs_dir"], f"greedy_baseline_{a.model}_{a.dataset}_{a.split}_limit{tag}.csv")
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
