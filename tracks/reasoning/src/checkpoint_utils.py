import csv, json, os, tempfile
from typing import Any, Dict

def _json_default(v):
    try:
        import numpy as np
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,np.ndarray): return v.tolist()
    except Exception: pass
    raise TypeError(f"Not JSON serializable: {type(v).__name__}")

def atomic_json_save(obj: Dict[str,Any], path: str):
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=".checkpoint_",suffix=".json",dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=_json_default)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.remove(tmp)

def load_json_checkpoint(path: str) -> Dict[str,Any]:
    with open(path,encoding="utf-8") as f: return json.load(f)

def csv_rows_from_results(results):
    rows=[]
    for result in results:
        row=dict(result); flops=row.pop("flops",{}) or {}
        if isinstance(flops,dict):
            for k,v in flops.items(): row[f"flops_{k}"]=v
        rows.append(row)
    return rows

def write_results_csv(results,path):
    rows=csv_rows_from_results(results); os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    if not rows: open(path,"w",encoding="utf-8").close(); return
    fields=[]; seen=set()
    for row in rows:
        for k in row:
            if k not in seen: seen.add(k); fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def make_run_checkpoint(results,ledger_records,metadata=None):
    return {"version":3,"results":results,"ledger_records":ledger_records,"metadata":metadata or {}}
