"""Shared checkpoint/tokenizer loading used by generate.py and inspect_model.py."""
import json

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