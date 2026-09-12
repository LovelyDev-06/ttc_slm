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
    # Prefer the mirror's own clean pre-extracted answer field when present;
    # only fall back to parsing \boxed{} out of the solution text if it's
    # missing, since the mirror's schema isn't guaranteed identical to the
    # original (now-removed) hendrycks/competition_math repo's.
    gold=str(ex.get("answer","")).strip()
    if not gold:
        from src.math_utils import extract_final_answer
        gold=extract_final_answer(solution)
    return {"problem_id":str(ex.get("unique_id", f"math_{idx}")),"prompt":problem,
            "answer":gold,"reference_solution":solution or problem,"dataset":"math"}

def load_dataset(name: str, split: str="test", limit: int=None):
    name=name.lower()
    if name=="gsm8k":
        ds=hf_load_dataset("gsm8k","main",split=split)
        conv=_gsm8k
    elif name in {"math","hendrycks_math"}:
        # The original hendrycks/competition_math repo was taken down from
        # Hugging Face (copyright takedown) -- switched to a public mirror
        # with the same problems plus a clean pre-extracted answer field.
        # Matches a fix already applied and verified working in the
        # math-track-ttc-scaling repo.
        hf_split="test" if split=="test" else "train"
        ds=hf_load_dataset("nlile/hendrycks-MATH-benchmark",split=hf_split)
        conv=_math
    else:
        raise ValueError("Unknown dataset. Expected gsm8k or math.")
    problems=[conv(ex,i) for i,ex in enumerate(ds)]
    return problems[:limit] if limit is not None else problems