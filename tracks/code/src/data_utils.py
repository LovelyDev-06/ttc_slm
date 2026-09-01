"""
data_utils.py

Loads HumanEval and MBPP and normalizes both into one common problem shape,
so every strategy script only has to deal with ONE format regardless of
which benchmark it's pointed at.

REUSE NOTE: same role as Partner A's data_utils.py (GSM8K/MATH loading).
Structure is intentionally parallel — a `load_dataset(name, split, limit)`
function returning a list of dicts with a common schema. If you want to
merge the two repos later, this is the file to unify first.
"""

from datasets import load_dataset as hf_load_dataset


COMMON_SCHEMA_KEYS = ["problem_id", "prompt", "test", "entry_point", "reference_solution"]


def _humaneval_to_common(example, idx):
    return {
        "problem_id": example.get("task_id", f"humaneval_{idx}"),
        "prompt": example["prompt"],
        "test": example["test"],
        "entry_point": example["entry_point"],
        "reference_solution": example.get("canonical_solution", ""),
        "dataset": "humaneval",
    }


def _mbpp_to_common(example, idx):
    # MBPP tests are a list of individual assert statements
    test_code = "\n".join(example["test_list"])
    return {
        "problem_id": f"mbpp_{example.get('task_id', idx)}",
        "prompt": (example.get("text") or example.get("prompt") or "") + "\n\n" + (example.get("test_list", [""])[0] if example.get("test_list") else ""),
        "test": test_code,
        "entry_point": _guess_entry_point(example["test_list"]),
        "reference_solution": example.get("code", ""),
        "dataset": "mbpp",
    }


def _guess_entry_point(test_list):
    """MBPP doesn't give an explicit function name; infer it from the first
    assert statement, e.g. 'assert similar_elements((3,4),(5,4)) == ...'"""
    if not test_list:
        return None
    first = test_list[0]
    try:
        after_assert = first.split("assert", 1)[1].strip()
        func_name = after_assert.split("(")[0].strip()
        return func_name
    except IndexError:
        return None


def load_dataset(name: str, split: str = "test", limit: int = None):
    """
    name: "humaneval" or "mbpp"
    Returns: list[dict] in the common schema above.
    """
    name = name.lower()
    if name == "humaneval":
        ds = hf_load_dataset("openai_humaneval", split=split)
        converter = _humaneval_to_common
    elif name == "mbpp":
        # sanitized config has cleaner single test_list entries
        ds = hf_load_dataset("mbpp", "sanitized", split=split if split != "test" else "test")
        converter = _mbpp_to_common
    else:
        raise ValueError(f"Unknown dataset '{name}'. Expected 'humaneval' or 'mbpp'.")

    problems = [converter(ex, i) for i, ex in enumerate(ds)]
    if limit:
        problems = problems[:limit]
    return problems
