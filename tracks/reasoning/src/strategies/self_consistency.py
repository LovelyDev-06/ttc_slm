import torch
from collections import defaultdict
from src.model_utils import build_reasoning_prompt
from src.reasoning_utils import normalize_reasoning, evaluate_output
from src.flop_utils import FlopRecord, count_tokens

def run_self_consistency(model,tokenizer,num_params,problems,config,ledger):
    results=[]; n=config["self_consistency"]["n_samples"]
    for p in problems:
        prompt=build_reasoning_prompt(tokenizer,p["prompt"],p["dataset"]); inp=tokenizer(prompt,return_tensors="pt").to(model.device)
        with torch.no_grad(): out=model.generate(**inp,max_new_tokens=config["models"]["max_new_tokens"],do_sample=True,temperature=config["models"]["temperature_sampling"],top_p=config["models"]["top_p"],num_return_sequences=n,pad_token_id=tokenizer.pad_token_id)
        groups=defaultdict(list); total=0
        for row in out:
            ids=row[inp["input_ids"].shape[1]:]; total+=len(ids); raw=tokenizer.decode(ids,skip_special_tokens=True); groups[normalize_reasoning(raw,p.get("choices"))].append(raw)
        # Primary vote is by final answer to avoid paraphrase-fragmentation of chains.
        answer_groups=defaultdict(list)
        for raws in groups.values():
            for raw in raws:
                ev=evaluate_output(raw,p); answer_groups[ev["predicted_answer"]].append(raw)
        best_answer=max(answer_groups.values(),key=len) if answer_groups else [next(iter(groups.values()))[0]]
        raw=best_answer[0]; ev=evaluate_output(raw,p)
        ledger.log(FlopRecord("self_consistency",p["problem_id"],num_params,count_tokens(tokenizer,prompt),total,1))
        results.append({"problem_id":p["problem_id"],"dataset":p["dataset"],"subset":p.get("subset"),"strategy":"self_consistency","response":raw,**ev,"n_samples":n,"consensus_count":len(best_answer)})
    return results
