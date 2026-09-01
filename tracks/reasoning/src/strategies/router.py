"""
Strategy 6 — Router (learned latent router).

The assignment guide describes the router as "a lightweight classifier
that selects an appropriate strategy based on problem difficulty or
characteristics." You specifically asked for a LEARNED LATENT router
rather than a hand-written if/else on surface features — so this is a
small neural network:

    problem text
       -> frozen sentence embedding (all-MiniLM-L6-v2, 384-dim, CPU-cheap)
       -> Linear + ReLU              (hidden_dim)
       -> Linear                     (latent_dim)   <- the "latent" bottleneck
       -> Linear + softmax           (num_strategies)

The bottleneck (latent_dim, e.g. 16) is the point: the network has to
COMPRESS whatever makes a problem "hard for greedy but fine for tree
search" into a small learned representation, rather than us hand-picking
features like "line count" or "loop count". scripts/train_router.py
trains this end-to-end on (embedding, best_strategy_label) pairs, where
labels come from actually running all strategies on a training split and
picking whichever one was correct at lowest FLOP cost (see
train_router.py's labeling logic, which uses flop_utils + cost_penalty_lambda
from config.yaml).

At inference, run_router() below embeds each problem, forwards it through
the trained net, and dispatches to whichever of the existing strategy
functions (greedy / best_of_n / self_consistency / tree_search) the net picked — reusing
those functions rather than reimplementing generation logic here.
"""

import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file

from src.strategies.greedy import run_greedy
from src.strategies.best_of_n import run_best_of_n
from src.strategies.tree_search import run_tree_search
from src.strategies.self_consistency import run_self_consistency


class LatentRouterNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),   # <- latent bottleneck
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)          # the "latent" representation
        logits = self.classifier(z)
        return logits, z

    def predict(self, x):
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.argmax(logits, dim=-1)


_embedder_cache = {}


def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def embed_problems(problems, config) -> np.ndarray:
    embedder = get_embedder(config["router"]["embedding_model"])
    texts = [p["prompt"] + "\n" + " ".join(f"{c['label']}: {c['text']}" for c in p.get("choices", [])) for p in problems]
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def load_router(checkpoint_path: str, config: dict, embedding_dim: int = 384) -> LatentRouterNet:
    strategies = config["router"]["strategies_available"]
    net = LatentRouterNet(
        input_dim=embedding_dim,
        hidden_dim=config["router"]["hidden_dim"],
        latent_dim=config["router"]["latent_dim"],
        num_classes=len(strategies),
    )
    net.load_state_dict(load_file(checkpoint_path))
    net.eval()
    return net


_STRATEGY_FUNCS = {
    "greedy": run_greedy,
    "best_of_n": run_best_of_n,
    "self_consistency": run_self_consistency,
    "tree_search": run_tree_search,
}


def run_router(model, tokenizer, num_params, problems, config, ledger, router_net):
    strategies = config["router"]["strategies_available"]

    # Inference masking: classes with no positive labels during router
    # training must not be selectable at inference. Otherwise an untrained
    # classifier output can win argmax purely due to random logit values.
    # Configure such classes under router.inference_mask_unseen in config.yaml.
    masked_strategies = config["router"].get("inference_mask_unseen", [])
    unknown = [s for s in masked_strategies if s not in strategies]
    if unknown:
        raise ValueError(
            f"Inference-mask strategies are not in strategies_available: {unknown}"
        )

    allowed = [s for s in strategies if s not in masked_strategies]
    if not allowed:
        raise ValueError("Inference masking removed every available strategy")

    embeddings = embed_problems(problems, config)
    x = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = router_net(x)
        masked_logits = logits.clone()
        for strategy in masked_strategies:
            idx = strategies.index(strategy)
            masked_logits[:, idx] = float("-inf")

        # Compute probabilities AFTER masking so masked classes have exactly
        # zero probability and cannot be selected by argmax.
        probs = torch.softmax(masked_logits, dim=-1)
        predicted_idx = torch.argmax(masked_logits, dim=-1).tolist()
    for problem, idx, prob in zip(problems, predicted_idx, probs.tolist()):
        print(f"Router | {problem['problem_id']} | predicted={idx} | strategy={strategies[idx]} | probabilities={[round(v, 4) for v in prob]}")

    results = []
    # group problems by routed strategy so we can batch-call each strategy fn
    buckets = {s: [] for s in strategies}
    for problem, idx in zip(problems, predicted_idx):
        buckets[strategies[idx]].append(problem)

    for strategy_name, bucket_problems in buckets.items():
        if not bucket_problems:
            continue
        strategy_fn = _STRATEGY_FUNCS[strategy_name]
        bucket_results = strategy_fn(model, tokenizer, num_params, bucket_problems, config, ledger)
        for r in bucket_results:
            r["strategy"] = "router"
            r["routed_to"] = strategy_name
        results.extend(bucket_results)

    return results
