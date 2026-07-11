"""Shared checkpoint/tokenizer loading used by generate.py and inspect_model.py."""
import json

import numpy as np
from tokenizers import Tokenizer

from model import MoETransformer, ModelConfig


def load_model_and_tokenizer(checkpoint_path, config_path, tokenizer_path):
    """
    Load a trained MoETransformer + its tokenizer from disk.

    Returns (model, cfg, tokenizer).
    """
    with open(config_path, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig(**cfg_dict)

    model = MoETransformer(cfg)
    model.load_weights(checkpoint_path)
    model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, cfg, tokenizer


def compute_layer_utilization(captures, n_experts):
    """
    Given the `captures` list returned by model(..., capture=True), return
    top-1 expert utilization counts per layer, flattened over batch and
    sequence positions. Used for the training dashboard's expert-balance
    scrubber and could equally be reused anywhere else that needs a quick
    "which experts are firing" snapshot.
    """
    per_layer = []
    for cap in captures:
        topk_idx = np.array(cap["moe"]["topk_idx"])  # (B, T, top_k)
        top1 = topk_idx[..., 0].reshape(-1)
        counts = [int(np.sum(top1 == e)) for e in range(n_experts)]
        per_layer.append(counts)
    return per_layer