"""
Strategy 1 — Greedy Decoding.

Single best next-token choice, no sampling. This is the cheapest strategy
and establishes the baseline accuracy + baseline FLOPs-per-problem that
every other strategy is compared against.
"""

import torch
from src.model_utils import build_code_prompt
from src.code_utils import extract_code, run_tests
from src.flop_utils import FlopRecord, count_tokens


def run_greedy(model, tokenizer, num_params, problems, config, ledger):
    results = []
    max_new_tokens = config["models"]["max_new_tokens"]

    for problem in problems:
        prompt = build_code_prompt(tokenizer, problem["prompt"], problem["dataset"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
        code = extract_code(raw_output)

        test_result = run_tests(
            code, problem["test"], problem["entry_point"],
            timeout_s=config["verifier"]["execution_timeout_s"],
        )

        ledger.log(FlopRecord(
            strategy="greedy",
            problem_id=problem["problem_id"],
            num_params=num_params,
            prompt_tokens=count_tokens(tokenizer, prompt),
            generated_tokens=len(gen_ids),
            num_generations=1,
        ))

        results.append({
            "problem_id": problem["problem_id"],
            "dataset": problem["dataset"],
            "strategy": "greedy",
            "code": code,
            "passed": test_result["passed"],
            "valid_ast": test_result["valid_ast"],
            "error": test_result["error"],
        })

    return results
