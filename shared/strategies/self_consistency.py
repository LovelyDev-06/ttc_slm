"""Strategy 3 — Self-Consistency.
Sample multiple independent solutions, normalize their AST when possible, vote by
normalized program, then verify the winning candidates against the benchmark tests.
"""
import ast, torch
from collections import Counter, defaultdict
from src.model_utils import build_code_prompt
from src.code_utils import extract_code, run_tests
from src.flop_utils import FlopRecord, count_tokens


def _normalize_code(code):
    try:
        return ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    except SyntaxError:
        return "RAW:" + code.strip()


def run_self_consistency(model, tokenizer, num_params, problems, config, ledger):
    results=[]; n=config["self_consistency"]["n_samples"]; max_new=config["models"]["max_new_tokens"]
    temp=config["models"]["temperature_sampling"]; top_p=config["models"]["top_p"]
    for problem in problems:
        prompt=build_code_prompt(tokenizer,problem["prompt"],problem["dataset"])
        inputs=tokenizer(prompt,return_tensors="pt").to(model.device); prompt_tokens=count_tokens(tokenizer,prompt)
        with torch.no_grad():
            out=model.generate(**inputs,max_new_tokens=max_new,do_sample=True,temperature=temp,top_p=top_p,num_return_sequences=n,pad_token_id=tokenizer.pad_token_id)
        groups=defaultdict(list); total_gen=0
        for row in out:
            ids=row[inputs["input_ids"].shape[1]:]; total_gen+=len(ids)
            code=extract_code(tokenizer.decode(ids,skip_special_tokens=True)); groups[_normalize_code(code)].append(code)
        ordered=sorted(groups.values(),key=len,reverse=True)
        candidates=[g[0] for g in ordered]
        tested=[]
        for code in candidates:
            tr=run_tests(code,problem["test"],problem["entry_point"],timeout_s=config["verifier"]["execution_timeout_s"])
            tested.append((code,tr,len(groups[_normalize_code(code)])))
        passing=[x for x in tested if x[1]["passed"]]
        best=max(passing,key=lambda x:x[2]) if passing else max(tested,key=lambda x:(x[2],x[1]["valid_ast"]))
        ledger.log(FlopRecord("self_consistency",problem["problem_id"],num_params,prompt_tokens,total_gen,1))
        results.append({"problem_id":problem["problem_id"],"dataset":problem["dataset"],"strategy":"self_consistency","code":best[0],"passed":best[1]["passed"],"valid_ast":best[1]["valid_ast"],"error":best[1]["error"],"n_samples":n,"consensus_count":best[2]})
    return results
