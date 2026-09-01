import torch
from src.model_utils import build_reasoning_prompt
from src.strategies.verifier import heuristic_score, verify_final
from src.flop_utils import FlopRecord, count_tokens
CHUNK_MAX_NEW_TOKENS=64

def _gen(model,tok,prompt,n,temp,top_p):
    inp=tok(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad(): out=model.generate(**inp,max_new_tokens=CHUNK_MAX_NEW_TOKENS,do_sample=True,temperature=temp,top_p=top_p,num_return_sequences=n,pad_token_id=tok.pad_token_id)
    vals=[]; total=0
    for row in out:
        ids=row[inp["input_ids"].shape[1]:]; total+=len(ids); vals.append(tok.decode(ids,skip_special_tokens=True))
    return vals,total,count_tokens(tok,prompt)

def _judge(model,tok,problem,partial):
    prompt=("You are judging an intermediate reasoning path for a multiple-choice problem. "
            "Is it coherent, relevant, and likely to reach the correct answer? Answer Yes or No.\n\nProblem:\n"+problem["prompt"]+"\n\nReasoning so far:\n"+partial+"\n\nAnswer:")
    inp=tok(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad(): logits=model(**inp).logits[0,-1]
    yi=tok.encode(" Yes",add_special_tokens=False); ni=tok.encode(" No",add_special_tokens=False)
    yes=logits[yi[0]] if yi else torch.tensor(0.,device=logits.device); no=logits[ni[0]] if ni else torch.tensor(0.,device=logits.device)
    return float(torch.softmax(torch.stack([yes,no]),0)[0]), count_tokens(tok,prompt)

def run_tree_search(model,tokenizer,num_params,problems,config,ledger,verifier=None):
    results=[]; c=config["tree_search"]
    for p in problems:
        base=build_reasoning_prompt(tokenizer,p["prompt"],p["dataset"]); beams=[{"text":"","score":0.0}]; total=0; base_tok=count_tokens(tokenizer,base); jc=jt=0
        for _ in range(c["max_depth"]):
            cand=[]
            for b in beams:
                chunks,gen,_=_gen(model,tokenizer,base+"\nReasoning so far:\n"+b["text"],c["branching_factor"],config["models"]["temperature_sampling"],config["models"]["top_p"]); total+=gen
                for ch in chunks:
                    text=b["text"]+ch; score,toks=_judge(model,tokenizer,p,text); jc+=1; jt+=toks
                    cand.append({"text":text,"score":0.7*score+0.3*heuristic_score(text,p)})
            cand.sort(key=lambda x:x["score"],reverse=True); beams=cand[:c["beam_width"]]
        best=max(beams,key=lambda x:x["score"]); ev=verify_final(best["text"],p)
        ledger.log(FlopRecord("tree_search",p["problem_id"],num_params,base_tok,total,1,jc,jt))
        results.append({"problem_id":p["problem_id"],"dataset":p["dataset"],"subset":p.get("subset"),"strategy":"tree_search","response":best["text"],**ev,"final_beam_score":best["score"],"judge_calls":jc})
    return results
