import torch
from src.model_utils import build_reasoning_prompt
from src.reasoning_utils import evaluate_output
from src.flop_utils import FlopRecord, count_tokens

def run_greedy(model,tokenizer,num_params,problems,config,ledger):
    results=[]
    for p in problems:
        prompt=build_reasoning_prompt(tokenizer,p["prompt"],p["dataset"]); inp=tokenizer(prompt,return_tensors="pt").to(model.device)
        with torch.no_grad(): out=model.generate(**inp,max_new_tokens=config["models"]["max_new_tokens"],do_sample=False,pad_token_id=tokenizer.pad_token_id)
        ids=out[0][inp["input_ids"].shape[1]:]; raw=tokenizer.decode(ids,skip_special_tokens=True); ev=evaluate_output(raw,p)
        ledger.log(FlopRecord("greedy",p["problem_id"],num_params,count_tokens(tokenizer,prompt),len(ids),1))
        results.append({"problem_id":p["problem_id"],"dataset":p["dataset"],"subset":p.get("subset"),"strategy":"greedy","response":raw,**ev})
    return results
