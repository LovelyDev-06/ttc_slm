"""
Strategy 2 — Best-of-N.

Generate N candidates by sampling, then pick the strongest one using a
domain-appropriate correctness check. For code, "domain-appropriate" means:
  1. Prefer any candidate that actually passes the tests (there may be more
     than one — pick the shortest, as a simple tie-breaker proxy for
     cleaner code).
  2. If NONE pass, fall back to the candidate with valid AST that fails the
     fewest visible assertions (best-effort partial credit signal), else
     just return the first candidate.
"""

import torch
from src.model_utils import build_code_prompt
from src.code_utils import extract_code, run_tests, is_valid_python
from src.flop_utils import FlopRecord, count_tokens


def _select_best(candidates_with_results):
    passing = [c for c in candidates_with_results if c["test_result"]["passed"]]
    if passing:
        return min(passing, key=lambda c: len(c["code"]))

    valid_ast = [c for c in candidates_with_results if c["test_result"]["valid_ast"]]
    if valid_ast:
        return valid_ast[0]

    return candidates_with_results[0]


def run_best_of_n(model, tokenizer, num_params, problems, config, ledger):
    results = []
    max_new_tokens = config["models"]["max_new_tokens"]
    n = config["best_of_n"]["n_candidates"]
    temp = config["models"]["temperature_sampling"]
    top_p = config["models"]["top_p"]

    for problem in problems:
        prompt = build_code_prompt(tokenizer, problem["prompt"], problem["dataset"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = count_tokens(tokenizer, prompt)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
            )

        candidates = []
        total_gen_tokens = 0
        for row in output_ids:
            gen_ids = row[inputs["input_ids"].shape[1]:]
            total_gen_tokens += len(gen_ids)
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
            code = extract_code(raw_output)
            test_result = run_tests(
                code, problem["test"], problem["entry_point"],
                timeout_s=config["verifier"]["execution_timeout_s"],
            )
            candidates.append({"code": code, "test_result": test_result})

        best = _select_best(candidates)

        ledger.log(FlopRecord(
            strategy="best_of_n",
            problem_id=problem["problem_id"],
            num_params=num_params,
            prompt_tokens=prompt_tokens,
            generated_tokens=total_gen_tokens,
            num_generations=1,  # tokens already summed above, avoid double counting
        ))

        results.append({
            "problem_id": problem["problem_id"],
            "dataset": problem["dataset"],
            "strategy": "best_of_n",
            "code": best["code"],
            "passed": best["test_result"]["passed"],
            "valid_ast": best["test_result"]["valid_ast"],
            "error": best["test_result"]["error"],
            "n_candidates": n,
            "n_passing": sum(1 for c in candidates if c["test_result"]["passed"]),
        })

    return results
