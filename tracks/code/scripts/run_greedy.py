#!/usr/bin/env python
"""Run greedy decoding with per-problem JSON checkpoints and a final CSV."""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_dataset
from src.model_utils import load_model_and_tokenizer
from src.strategies.greedy import run_greedy
from src.flop_utils import FlopLedger, FlopRecord
from src.checkpoint_utils import atomic_json_save, load_json_checkpoint, make_run_checkpoint, write_results_csv
from src.hub_utils import push_file, download_file


def ledger_from_checkpoint(records):
    ledger = FlopLedger()
    for r in records:
        ledger.log(FlopRecord(
            strategy=r["strategy"], problem_id=r["problem_id"], num_params=int(r["num_params"]),
            prompt_tokens=int(r["prompt_tokens"]), generated_tokens=int(r["generated_tokens"]),
            num_generations=int(r.get("num_generations", 1)),
            extra_forward_passes=int(r.get("extra_forward_passes", 0)),
        ))
    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen1_5b", "qwen7b"])
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no_push", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
    os.makedirs(config["paths"]["checkpoints_dir"], exist_ok=True)

    problems = load_dataset(args.dataset, split=args.split, limit=args.limit)
    limit_tag = len(problems)
    stem = f"greedy_{args.model}_{args.dataset}_{args.split}_limit{limit_tag}"
    checkpoint_path = os.path.join(config["paths"]["checkpoints_dir"], stem + ".json")
    csv_path = os.path.join(config["paths"]["logs_dir"], stem + ".csv")
    hub_checkpoint = f"checkpoints/{stem}.json"

    if not os.path.exists(checkpoint_path) and not args.no_push:
        download_file(config, hub_checkpoint, checkpoint_path)

    checkpoint = load_json_checkpoint(checkpoint_path) if os.path.exists(checkpoint_path) else {"results": [], "ledger_records": []}
    results = checkpoint.get("results", [])
    completed = {r["problem_id"] for r in results}
    ledger = ledger_from_checkpoint(checkpoint.get("ledger_records", []))

    print(f"{len(problems)} problems loaded. Already completed: {len(completed)}")
    model, tokenizer, num_params = load_model_and_tokenizer(args.model, config)

    for i, problem in enumerate(problems, 1):
        pid = problem["problem_id"]
        if pid in completed:
            print(f"[{i}/{len(problems)}] {pid}: already complete, skipping")
            continue
        print(f"[{i}/{len(problems)}] {pid}: running greedy")
        local_ledger = FlopLedger()
        new_results = run_greedy(model, tokenizer, num_params, [problem], config, local_ledger)
        r = new_results[0]
        flop_record = local_ledger.as_dicts()[0]
        r["flops"] = flop_record
        results.append(r)
        ledger.log(FlopRecord(
            strategy=flop_record["strategy"], problem_id=flop_record["problem_id"], num_params=flop_record["num_params"],
            prompt_tokens=flop_record["prompt_tokens"], generated_tokens=flop_record["generated_tokens"],
            num_generations=flop_record["num_generations"], extra_forward_passes=flop_record["extra_forward_passes"],
        ))
        payload = make_run_checkpoint(results, ledger.as_dicts(), {"model": args.model, "dataset": args.dataset, "split": args.split, "limit": limit_tag})
        atomic_json_save(payload, checkpoint_path)
        if not args.no_push and i % config["hub"].get("push_every_n_problems", 1) == 0:
            push_file(checkpoint_path, config, hub_checkpoint)

    write_results_csv(results, csv_path)
    print(f"Greedy pass rate: {sum(bool(r['passed']) for r in results)}/{len(results)} = {sum(bool(r['passed']) for r in results)/max(len(results),1):.2%}")
    print(f"FLOP summary: {ledger.summary()}")
    print(f"Wrote {csv_path}")
    if not args.no_push:
        push_file(csv_path, config, f"logs/{os.path.basename(csv_path)}")


if __name__ == "__main__":
    main()
