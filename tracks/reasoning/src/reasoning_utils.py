"""Reasoning-domain answer extraction, normalization, and correctness checks."""
import re, hashlib

_LABEL_RE=re.compile(r"(?:final\s+answer|answer|option|choice)?\s*(?:is|:)?\s*[\(\[]?([A-Z])[\)\]]?", re.I)

def extract_final_answer(text, choices=None):
    if not text: return ""
    labels=[str(c["label"]).upper() for c in (choices or [])]
    matches=_LABEL_RE.findall(text)
    if matches:
        for x in reversed(matches):
            x=x.upper()
            if not labels or x in labels: return x
    # fallback: standalone choice label near the end
    tail=text[-200:].upper()
    for x in reversed(labels):
        if re.search(rf"(?<![A-Z]){re.escape(x)}(?![A-Z])",tail): return x
    return ""

def normalize_reasoning(text, choices=None):
    ans=extract_final_answer(text,choices)
    chain=re.sub(r"\s+"," ",text.strip().lower())
    chain=re.sub(r"[^a-z0-9 .,:;()\-]","",chain)
    # Keep both answer and coarse reasoning form; exact wording is intentionally not required.
    key=f"{ans}|{chain[:1200]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

def answer_correct(predicted, gold):
    return str(predicted).strip().upper()==str(gold).strip().upper()

def evaluate_output(raw_output, problem):
    pred=extract_final_answer(raw_output,problem.get("choices"))
    return {"passed": answer_correct(pred,problem.get("answer","")), "predicted_answer":pred, "gold_answer":problem.get("answer","")}
