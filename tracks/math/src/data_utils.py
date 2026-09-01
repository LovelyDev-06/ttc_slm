"""Math-track dataset loading for GSM8K and MATH with a common schema."""
from datasets import load_dataset as hf_load_dataset

def _gsm8k(ex, idx):
    raw=str(ex.get("answer",""))
    gold=raw.split("####")[-1].strip() if "####" in raw else raw.strip()
    return {"problem_id":f"gsm8k_{idx}","prompt":str(ex.get("question","")),
            "answer":gold,"reference_solution":raw,"dataset":"gsm8k"}

def _math(ex, idx):
    problem=str(ex.get("problem",""))
    solution=str(ex.get("solution",""))
    # Preserve the official solution; math_utils extracts the final answer from boxed/final form.
    from src.math_utils import extract_final_answer
    gold=extract_final_answer(solution)
    return {"problem_id":str(ex.get("unique_id", f"math_{idx}")),"prompt":problem,
            "answer":gold,"reference_solution":solution,"dataset":"math"}

def load_dataset(name: str, split: str="test", limit: int=None):
    name=name.lower()
    if name=="gsm8k":
        ds=hf_load_dataset("gsm8k","main",split=split)
        conv=_gsm8k
    elif name in {"math","hendrycks_math"}:
        ds=hf_load_dataset("hendrycks/competition_math",split=split)
        conv=_math
    else:
        raise ValueError("Unknown dataset. Expected gsm8k or math.")
    problems=[conv(ex,i) for i,ex in enumerate(ds)]
    return problems[:limit] if limit is not None else problems
