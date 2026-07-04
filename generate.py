"""
Generate text from a trained checkpoint
"""
import argparse

import mlx.core as mx

from common import load_model_and_tokenizer


def sample_top_p(logits, temperature=0.8, top_p=0.9):
    """Nucleus (top-p) sampling from a single row of logits."""
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    sorted_idx = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)

    # zero out tail beyond top_p
    mask = (cumulative - sorted_probs) < top_p
    sorted_probs = sorted_probs * mask.astype(sorted_probs.dtype)
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    next_idx_in_sorted = mx.random.categorical(mx.log(sorted_probs + 1e-10))
    next_token = mx.take_along_axis(sorted_idx, next_idx_in_sorted.reshape(-1, 1), axis=-1)
    return next_token.reshape(-1)


def main():
    """Parse args, load a checkpoint, and stream sampled text to stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model, _cfg, tok = load_model_and_tokenizer(args.checkpoint, args.config, args.tokenizer)

    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")

    ids = [bos_id] + tok.encode(args.prompt).ids
    tokens = mx.array([ids])

    print(f"Prompt: {args.prompt!r}\n---")
    generated = list(ids)

    cache = None
    for _ in range(args.max_tokens):
        logits, _, cache, _ = model(tokens, cache=cache)
        next_logits = logits[:, -1, :]
        next_token = sample_top_p(next_logits, args.temperature, args.top_p)
        token_id = next_token.item()
        if token_id == eos_id:
            break
        generated.append(token_id)
        tokens = next_token.reshape(1, 1)

    text = tok.decode(generated)
    print(text)


if __name__ == "__main__":
    main()