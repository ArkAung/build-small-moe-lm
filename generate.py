"""
Step 5: Generate text from a trained checkpoint, and optionally inspect
expert routing statistics (which experts fire on which tokens) -- this is
the fun part for actually seeing what your MoE learned.

Usage:
    python generate.py --checkpoint checkpoints/final.safetensors \
                        --config checkpoints/config.json \
                        --tokenizer data/tokenizer.json \
                        --prompt "Once upon a time" --max_tokens 200

For an instruction-tuned checkpoint (see finetune.py), pass --instruct so
the prompt is wrapped in the same template the model was fine-tuned on:

    python generate.py --checkpoint checkpoints_instruct/final.safetensors \
                        --config checkpoints_instruct/config.json \
                        --tokenizer data/tokenizer.json \
                        --instruct --max_tokens 100
"""
import argparse

import mlx.core as mx

from common import load_model_and_tokenizer
from prepare_instruct_data import build_prompt, DEFAULT_INSTRUCTION


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
    parser.add_argument("--instruct", action="store_true",
                         help="Wrap the prompt in the instruction/response template "
                              "used by finetune.py, for instruction-tuned checkpoints.")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION,
                         help="Instruction text used when --instruct is set. Note: the "
                              "fine-tune only ever saw the default instruction text, so "
                              "overriding this is experimental and off-distribution.")
    args = parser.parse_args()

    model, _cfg, tok = load_model_and_tokenizer(args.checkpoint, args.config, args.tokenizer)

    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")

    prompt_text = build_prompt(args.instruction) if args.instruct else args.prompt
    ids = [bos_id] + tok.encode(prompt_text).ids
    tokens = mx.array([ids])

    print(f"Prompt: {prompt_text!r}\n---")
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