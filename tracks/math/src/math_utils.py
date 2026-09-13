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

_NUM_TOKEN=re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?|\\frac\{[^{}]+\}\{[^{}]+\}")

def extract_final_answer(text, choices=None):
    if not text: return ""
    boxes=_BOXED.findall(text)
    if boxes: return _canon(boxes[-1])
    matches=_FINAL.findall(text)
    if matches:
        # BUG (found via a mismatched-prediction spot check on the greedy
        # baseline): the old code did `return _canon(matches[-1])` --
        # _canon() strips ALL whitespace, which is correct for a clean
        # numeric answer but destructive when _FINAL's `([^\n]+)` capture
        # grabs a full prose sentence (happens whenever the word "answer"
        # appears mid-reasoning, not just at a clean "Final Answer: X"
        # marker). A correct answer stated as "...Doctor Jones will have 1
        # hour left..." was getting mangled into the unreadable blob
        # "...DoctorJoneswillhave1hourleft..." and compared against the
        # clean gold answer "1", always failing even though the model was
        # right. Fix: pull out just the leading numeric/mathematical token
        # from the captured text instead of canon-ing the whole sentence;
        # fall back to the old behavior only if no such token is found.
        candidate = matches[-1]
        num_tokens = _NUM_TOKEN.findall(candidate)
        if num_tokens:
            # Take the LAST number in the sentence, not the first -- answers
            # typically appear at the end (e.g. "...is 10 - 5 = 5." should
            # extract "5", not the "10" that appears earlier in the sentence).
            return _canon(num_tokens[-1])
        return _canon(candidate)
    lines=[x.strip() for x in str(text).splitlines() if x.strip()]
    if not lines:
        return ""
    # Same fix as above, applied to the last-line fallback: canon-ing an
    # entire prose sentence (rather than pulling out just the number)
    # mangles a correct answer into an unreadable, unmatchable blob.
    last_line = lines[-1]
    num_tokens = _NUM_TOKEN.findall(last_line)
    if num_tokens:
        return _canon(num_tokens[-1])
    return _canon(last_line)

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
