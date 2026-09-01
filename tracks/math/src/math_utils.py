"""Math-domain answer extraction, normalization, and correctness checks."""
import re, hashlib
_BOXED=re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL=re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*([^\n]+)", re.I)

def _canon(s):
    s=str(s or "").strip()
    s=s.replace(",", "").replace("$", "")
    s=re.sub(r"\\\((.*?)\\\)", r"\1", s)
    s=re.sub(r"\\\[(.*?)\\\]", r"\1", s)
    return re.sub(r"\s+", "", s)

def extract_final_answer(text, choices=None):
    if not text: return ""
    boxes=_BOXED.findall(text)
    if boxes: return _canon(boxes[-1])
    matches=_FINAL.findall(text)
    if matches: return _canon(matches[-1])
    lines=[x.strip() for x in str(text).splitlines() if x.strip()]
    return _canon(lines[-1]) if lines else ""

def normalize_math(text, choices=None):
    ans=extract_final_answer(text)
    chain=re.sub(r"\s+"," ",str(text).strip().lower())
    chain=re.sub(r"[^a-z0-9 .,:;()\-+*/=^]","",chain)
    return hashlib.sha256(f"{ans}|{chain[:1200]}".encode()).hexdigest()[:16]

def answer_correct(predicted, gold):
    return _canon(predicted)==_canon(gold)

def evaluate_output(raw_output, problem):
    pred=extract_final_answer(raw_output)
    return {"passed": answer_correct(pred, problem.get("answer","")),
            "predicted_answer":pred, "gold_answer":problem.get("answer","")}
