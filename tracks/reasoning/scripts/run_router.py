#!/usr/bin/env python
"""Run the learned router one problem at a time so every result is checkpointed."""
import argparse, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_dataset
from src.model_utils import load_model_and_tokenizer
from src.strategies.router import run_router, load_router, embed_problems
from src.flop_utils import FlopLedger, FlopRecord
from src.checkpoint_utils import atomic_json_save, load_json_checkpoint, make_run_checkpoint, write_results_csv
from src.hub_utils import push_file, download_file

def restore(records):
    l=FlopLedger()
    for r in records: l.log(FlopRecord(r["strategy"],r["problem_id"],int(r["num_params"]),int(r["prompt_tokens"]),int(r["generated_tokens"]),int(r.get("num_generations",1)),int(r.get("extra_forward_passes",0)),int(r.get("extra_forward_tokens",0))))
    return l

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True,choices=["llama1b","qwen1_5b"]); p.add_argument("--dataset",required=True,choices=["arc_challenge","mmlu_stem"]); p.add_argument("--split",default="test"); p.add_argument("--limit",type=int,default=None); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--router_checkpoint",default="checkpoints/router.safetensors"); p.add_argument("--no_push",action="store_true"); a=p.parse_args()
    with open(a.config,encoding="utf-8") as f: cfg=yaml.safe_load(f)
    os.makedirs(cfg["paths"]["logs_dir"],exist_ok=True); os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    if not os.path.exists(a.router_checkpoint) and not a.no_push: download_file(cfg,"checkpoints/router.safetensors",a.router_checkpoint)
    if not os.path.exists(a.router_checkpoint): raise FileNotFoundError(f"No router checkpoint at {a.router_checkpoint}. Run train_router.py first.")
    problems=load_dataset(a.dataset,split=a.split,limit=a.limit); tag=len(problems); stem=f"router_{a.model}_{a.dataset}_{a.split}_limit{tag}"; cp=os.path.join(cfg["paths"]["checkpoints_dir"],stem+".json"); csv=os.path.join(cfg["paths"]["logs_dir"],stem+".csv"); hubcp=f"checkpoints/{stem}.json"
    if not os.path.exists(cp) and not a.no_push: download_file(cfg,hubcp,cp)
    state=load_json_checkpoint(cp) if os.path.exists(cp) else {"results":[],"ledger_records":[]}; results=state.get("results",[]); done={r["problem_id"] for r in results}; ledger=restore(state.get("ledger_records",[]))
    model,tokenizer,num_params=load_model_and_tokenizer(a.model,cfg); embed_dim=embed_problems(problems[:1],cfg).shape[1]; router_net=load_router(a.router_checkpoint,cfg,embedding_dim=embed_dim)
    print(f"{len(problems)} problems loaded. Already completed: {len(done)}")
    for i,problem in enumerate(problems,1):
        pid=problem["problem_id"]
        if pid in done: print(f"[{i}/{len(problems)}] {pid}: already complete, skipping"); continue
        print(f"[{i}/{len(problems)}] {pid}: routing")
        ll=FlopLedger(); r = run_router(model, tokenizer, num_params, [problem], cfg, ll, router_net)[0]; fr=ll.as_dicts()[0]; r["flops"]=fr; results.append(r); ledger.log(FlopRecord(fr["strategy"],fr["problem_id"],int(fr["num_params"]),int(fr["prompt_tokens"]),int(fr["generated_tokens"]),int(fr["num_generations"]),int(fr["extra_forward_passes"]),int(fr.get("extra_forward_tokens",0))))
        atomic_json_save(make_run_checkpoint(results,ledger.as_dicts(),{"model":a.model,"dataset":a.dataset,"split":a.split,"limit":tag}),cp)
        if not a.no_push and i%cfg["hub"].get("push_every_n_problems",1)==0: push_file(cp,cfg,hubcp)
    write_results_csv(results,csv); print(f"Accuracy: {sum(bool(r['passed']) for r in results)}/{len(results)} = {sum(bool(r['passed']) for r in results)/max(len(results),1):.2%}"); print(f"FLOP summary: {ledger.summary()}"); print(f"Wrote {csv}")
    if not a.no_push: push_file(csv,cfg,f"logs/{os.path.basename(csv)}")
if __name__=="__main__": main()
