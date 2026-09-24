"""Math-track dataset loading for GSM8K only (canonical paper setup)."""
from datasets import load_dataset as hf_load_dataset

def _gsm8k(ex, idx):
    raw=str(ex.get("answer",""))
    gold=raw.split("####")[-1].strip() if "####" in raw else raw.strip()
    return {"problem_id":f"gsm8k_{idx}","prompt":str(ex.get("question","")),
            "answer":gold,"reference_solution":raw,"dataset":"gsm8k"}

def load_dataset(name: str, split: str="test", limit: int=None, seed: int=None):
    name=name.lower()
    if name=="gsm8k":
        ds=hf_load_dataset("openai/gsm8k","main",split=split)
        conv=_gsm8k
    else:
        raise ValueError("Unknown dataset. Expected gsm8k.")
    problems=[conv(ex,i) for i,ex in enumerate(ds)]
    if limit is not None and seed is not None:
        import random
        rng=random.Random(seed)
        problems=rng.sample(problems, min(limit, len(problems)))
        problems.sort(key=lambda p: p["problem_id"])
    elif limit is not None:
        problems=problems[:limit]
    return problems