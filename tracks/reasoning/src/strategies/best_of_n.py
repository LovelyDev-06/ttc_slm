import torch
from src.model_utils import build_reasoning_prompt
from src.reasoning_utils import evaluate_output
from src.flop_utils import FlopRecord, count_tokens

def run_best_of_n(model,tokenizer,num_params,problems,config,ledger):
    results=[]; n=config["best_of_n"]["n_candidates"]
    for p in problems:
        prompt=build_reasoning_prompt(tokenizer,p["prompt"],p["dataset"]); inp=tokenizer(prompt,return_tensors="pt").to(model.device)
        with torch.no_grad(): out=model.generate(**inp,max_new_tokens=config["models"]["max_new_tokens"],do_sample=True,temperature=config["models"]["temperature_sampling"],top_p=config["models"]["top_p"],num_return_sequences=n,pad_token_id=tokenizer.pad_token_id)
        cand=[]; total=0
        for row in out:
            ids=row[inp["input_ids"].shape[1]:]; total+=len(ids); raw=tokenizer.decode(ids,skip_special_tokens=True); ev=evaluate_output(raw,p); cand.append((raw,ev))
        # Best-of-N uses benchmark answer correctness for oracle candidate selection during strategy evaluation.
        winners=[c for c in cand if c[1]["passed"]]; raw,ev=(min(winners,key=lambda c:len(c[0])) if winners else cand[0])
        ledger.log(FlopRecord("best_of_n",p["problem_id"],num_params,count_tokens(tokenizer,prompt),total,1))
        results.append({"problem_id":p["problem_id"],"dataset":p["dataset"],"subset":p.get("subset"),"strategy":"best_of_n","response":raw,**ev,"n_candidates":n,"n_passing":len(winners)})
    return results
