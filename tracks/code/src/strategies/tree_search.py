"""Strategy 4 — Tree Search with an LLM-based step judge.
Each partial program is expanded in small chunks. A lightweight judge prompt is
scored by the same loaded language model to estimate whether the partial path is
coherent and likely to lead to a correct solution. This replaces heuristic-only
intermediate verification; final correctness still uses AST + benchmark tests.
"""
import torch
from src.model_utils import build_code_prompt
from src.code_utils import extract_code
from src.strategies.verifier import verify_final
from src.flop_utils import FlopRecord, count_tokens

CHUNK_MAX_NEW_TOKENS=40

def _generate_chunk(model,tokenizer,prompt,n,temp,top_p):
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**inputs,max_new_tokens=CHUNK_MAX_NEW_TOKENS,do_sample=True,temperature=temp,top_p=top_p,num_return_sequences=n,pad_token_id=tokenizer.pad_token_id)
    chunks=[]; counts=[]
    for row in out:
        ids=row[inputs["input_ids"].shape[1]:]; counts.append(len(ids)); chunks.append(tokenizer.decode(ids,skip_special_tokens=True))
    return chunks,counts,count_tokens(tokenizer,prompt)

def _looks_complete(code): return "return" in code and code.count("def ")>=1

def _judge_score(model,tokenizer,problem,partial_code):
    prompt=("You are a strict code reasoning judge. Given a programming problem and an intermediate Python solution, "
            "estimate whether this intermediate reasoning/code is coherent, relevant, and likely to lead to a correct solution. "
            "Answer Yes or No.\n\nProblem:\n"+problem["prompt"]+"\n\nIntermediate code:\n"+partial_code+"\n\nAnswer:")
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad(): logits=model(**inputs).logits[0,-1]
    yes_ids=tokenizer.encode(" Yes",add_special_tokens=False); no_ids=tokenizer.encode(" No",add_special_tokens=False)
    yes=logits[yes_ids[0]] if yes_ids else torch.tensor(0.,device=logits.device)
    no=logits[no_ids[0]] if no_ids else torch.tensor(0.,device=logits.device)
    score=float(torch.softmax(torch.stack([yes,no]),dim=0)[0].item())
    return score,count_tokens(tokenizer,prompt)

def run_tree_search(model,tokenizer,num_params,problems,config,ledger,verifier=None):
    results=[]; cfg=config["tree_search"]; branching=cfg["branching_factor"]; max_depth=cfg["max_depth"]; beam_width=cfg["beam_width"]
    temp=config["models"]["temperature_sampling"]; top_p=config["models"]["top_p"]
    for problem in problems:
        base=build_code_prompt(tokenizer,problem["prompt"],problem["dataset"]); beams=[{"code":"","score":0.0}]; total_gen=0; base_tokens=count_tokens(tokenizer,base); judge_calls=0; judge_tokens=0
        for _ in range(max_depth):
            candidates=[]
            for beam in beams:
                if _looks_complete(beam["code"]): candidates.append(beam); continue
                chunks,counts,_=_generate_chunk(model,tokenizer,base+beam["code"],branching,temp,top_p); total_gen+=sum(counts)
                for chunk in chunks:
                    code=beam["code"]+chunk; score,toks=_judge_score(model,tokenizer,problem,code); judge_calls+=1; judge_tokens+=toks; candidates.append({"code":code,"score":score})
            candidates.sort(key=lambda c:c["score"],reverse=True); beams=candidates[:beam_width]
            if all(_looks_complete(b["code"]) for b in beams): break
        best=max(beams,key=lambda b:b["score"]); final_code=extract_code(best["code"])
        tr=verify_final(final_code,problem["test"],problem["entry_point"],timeout_s=config["verifier"]["execution_timeout_s"])
        ledger.log(FlopRecord("tree_search",problem["problem_id"],num_params,base_tokens,total_gen,1,judge_calls,judge_tokens))
        results.append({"problem_id":problem["problem_id"],"dataset":problem["dataset"],"strategy":"tree_search","code":final_code,"passed":tr["passed"],"valid_ast":tr["valid_ast"],"error":tr["error"],"final_beam_score":best["score"],"step_judge":"llm","judge_calls":judge_calls})
    return results
