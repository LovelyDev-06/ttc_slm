"""Reasoning-track dataset loading for ARC-Challenge and MMLU STEM."""
from datasets import load_dataset as hf_load_dataset

MMLU_STEM_SUBSETS = [
    "abstract_algebra","astronomy","college_biology","college_chemistry",
    "college_computer_science","college_mathematics","college_physics",
    "computer_security","conceptual_physics","electrical_engineering",
    "high_school_biology","high_school_chemistry","high_school_computer_science",
    "high_school_mathematics","high_school_physics","high_school_statistics",
    "machine_learning",
]


def _choices(example):
    choices = example.get("choices")
    if isinstance(choices, dict):
        labels = choices.get("label") or [chr(65+i) for i in range(len(choices.get("text", [])))]
        texts = choices.get("text", [])
        return list(zip(labels, texts))
    if isinstance(choices, list):
        return list(zip([chr(65+i) for i in range(len(choices))], choices))
    opts = example.get("options", [])
    return list(zip([chr(65+i) for i in range(len(opts))], opts))


def _normalize(example, idx, dataset, subset=None):
    pairs = _choices(example)
    answer = example.get("answerKey", example.get("answer", example.get("label")))
    if isinstance(answer, int) and pairs:
        answer = pairs[answer][0]
    if isinstance(answer, str) and answer.isdigit() and pairs:
        n=int(answer)
        if 0 <= n < len(pairs): answer=pairs[n][0]
        elif 1 <= n <= len(pairs): answer=pairs[n-1][0]
    question = example.get("question", example.get("input", ""))
    option_text = "\n".join(f"{lab}. {txt}" for lab,txt in pairs)
    prompt = question.strip() + "\n\nChoices:\n" + option_text
    pid = example.get("id") or example.get("task_id") or f"{dataset}_{subset or 'default'}_{idx}"
    return {
        "problem_id": str(pid), "prompt": prompt, "question": question,
        "choices": [{"label":str(l),"text":str(t)} for l,t in pairs],
        "answer": str(answer).strip(), "dataset": dataset,
        "subset": subset,
    }


def load_dataset(name: str, split: str="test", limit: int=None):
    name=name.lower()
    if name in {"arc", "arc_challenge", "arc-challenge"}:
        ds=hf_load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        problems=[_normalize(ex,i,"arc_challenge") for i,ex in enumerate(ds)]
    elif name in {"mmlu", "mmlu_stem", "mmlu-stem"}:
        # Concatenate STEM subjects while preserving the subject name.
        parts=[]
        for subset in MMLU_STEM_SUBSETS:
            try:
                ds=hf_load_dataset("cais/mmlu", subset, split=split)
            except Exception as e:
                raise RuntimeError(f"Could not load MMLU STEM subset '{subset}' split '{split}'.") from e
            parts.extend(_normalize(ex,i,"mmlu_stem",subset) for i,ex in enumerate(ds))
        problems=parts
    else:
        raise ValueError("Unknown dataset. Expected arc_challenge or mmlu_stem.")
    if limit is not None: problems=problems[:limit]
    return problems
