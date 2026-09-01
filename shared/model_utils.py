"""
model_utils.py

Loads Qwen2.5 models/tokenizers for the code track.

REUSE NOTE: this file is ~95% identical to Partner A's src/model_utils.py.
The only code-track-specific thing is which model names live in config.yaml
under `models:`. If Partner A already has a working model_utils.py, you can
literally copy theirs and just point MODEL_ALIASES at Qwen instead of Llama.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ALIASES = {
    "qwen1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_model_name(alias_or_name: str, config: dict) -> str:
    """Accepts either a short alias ('qwen1_5b') or a full HF repo id."""
    models_cfg = config.get("models", {})
    if alias_or_name in models_cfg:
        return models_cfg[alias_or_name]
    if alias_or_name in MODEL_ALIASES:
        return MODEL_ALIASES[alias_or_name]
    return alias_or_name  # assume it's already a full HF repo id


def load_model_and_tokenizer(alias_or_name: str, config: dict):
    model_name = resolve_model_name(alias_or_name, config)
    dtype = _DTYPE_MAP.get(config["models"].get("dtype", "bfloat16"), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=config["models"].get("device_map", "auto"),
    )
    model.eval()

    # num_params is needed downstream for FLOP accounting (see flop_utils.py)
    num_params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, num_params


def build_code_prompt(tokenizer, problem_prompt: str, dataset: str) -> str:
    """
    Wraps a raw HumanEval/MBPP problem statement in Qwen's chat template.
    Kept separate from data_utils.py so strategies can call it directly
    when they need to re-prompt (e.g. tree search expanding a partial solution).
    """
    if dataset == "humaneval":
        instruction = (
            "Complete the following Python function. Only output the full "
            "function implementation (including the signature), no explanations, "
            "no markdown fences.\n\n" + problem_prompt
        )
    else:  # mbpp
        instruction = (
            "Write a Python function that solves the following problem. "
            "Only output the code, no explanations, no markdown fences.\n\n" + problem_prompt
        )

    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
