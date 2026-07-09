"""
Prepare instruction-tuning data from a joke dataset (or any short-text
dataset) using an already-trained tokenizer.

Each example is wrapped in a fixed template:

    ### Instruction:
    Tell me a joke.

    ### Response:
    <joke text><eos>

and tokenized as a single sequence, alongside a parallel loss mask that is
0 over the instruction/template tokens (and any padding) and 1 over the
response tokens -- so fine-tuning only trains the model to predict the
response, not to reproduce the instruction template.

This intentionally uses one fixed generic instruction rather than
per-example topics, since ysharma/short_jokes has no topic labels. Extending
to topic-conditioned instructions (e.g. via keyword extraction per joke) is
a reasonable next step but out of scope here.

Usage:
    python prepare_instruct_data.py \
        --tokenizer data/tokenizer.json \
        --out_dir data_instruct \
        --hf_dataset ysharma/short_jokes --text_column Joke \
        --max_len 128 --max_examples 50000
"""
import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
DEFAULT_INSTRUCTION = "Tell me a joke."


def build_prompt(instruction=DEFAULT_INSTRUCTION):
    """Format an instruction into the fixed template used for training/inference."""
    return PROMPT_TEMPLATE.format(instruction=instruction)


def get_text_iterator(hf_dataset=None, text_column="Joke", local_file=None, max_examples=None):
    """Yield one raw response text (e.g. one joke) at a time."""
    if local_file:
        with open(local_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    else:
        # Lazy import: `datasets` is a heavy optional dependency only needed
        # when downloading from the Hub, not when using --local_file.
        from datasets import load_dataset  # pylint: disable=import-outside-toplevel
        ds = load_dataset(hf_dataset, split="train")
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))
        for row in ds:
            text = str(row[text_column]).strip()
            if text:
                yield text


def build_example(tokenizer, prompt_ids, response_text, max_len, pad_id, eos_id):
    """
    Tokenize one (fixed instruction, response) pair into a padded id array
    plus an aligned loss mask (1 = response/eos token, 0 = instruction,
    template, or padding).
    """
    response_ids = tokenizer.encode(response_text).ids
    ids = prompt_ids + response_ids + [eos_id]
    mask = [0] * len(prompt_ids) + [1] * (len(response_ids) + 1)

    if len(ids) > max_len:
        return None  # skip examples that don't fit; simplest correct policy

    pad_amount = max_len - len(ids)
    ids = ids + [pad_id] * pad_amount
    mask = mask + [0] * pad_amount
    return ids, mask


def main():
    """Parse args, build instruction/response examples, and write train/val shards."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                         help="Path to the tokenizer.json trained in the pretraining stage.")
    parser.add_argument("--out_dir", type=str, default="data_instruct")
    parser.add_argument("--hf_dataset", type=str, default="ysharma/short_jokes")
    parser.add_argument("--text_column", type=str, default="Joke")
    parser.add_argument("--local_file", type=str, default=None,
                         help="Optional local .txt file (one response per line) instead of --hf_dataset.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=128,
                         help="Padded sequence length. Examples longer than this are skipped.")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION,
                         help="Fixed instruction text used for every example.")
    parser.add_argument("--val_fraction", type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    prompt_text = build_prompt(args.instruction)
    prompt_ids = [bos_id] + tokenizer.encode(prompt_text).ids

    all_ids, all_masks = [], []
    skipped = 0
    text_iter = get_text_iterator(args.hf_dataset, args.text_column, args.local_file, args.max_examples)
    for text in tqdm(text_iter, desc="Building instruction examples"):
        example = build_example(tokenizer, prompt_ids, text, args.max_len, pad_id, eos_id)
        if example is None:
            skipped += 1
            continue
        ids, mask = example
        all_ids.append(ids)
        all_masks.append(mask)

    print(f"Built {len(all_ids)} examples ({skipped} skipped for exceeding max_len={args.max_len})")

    ids_arr = np.array(all_ids, dtype=np.uint16)
    mask_arr = np.array(all_masks, dtype=np.uint8)

    n = len(ids_arr)
    n_val = max(1, int(n * args.val_fraction))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    for split_name, idx in [("train", train_idx), ("val", val_idx)]:
        ids_arr[idx].tofile(os.path.join(args.out_dir, f"instruct_{split_name}_ids.bin"))
        mask_arr[idx].tofile(os.path.join(args.out_dir, f"instruct_{split_name}_mask.bin"))
        print(f"{split_name}: {len(idx)} examples")

    meta = {
        "max_len": args.max_len,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "instruction_template": prompt_text,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta.json -> {args.out_dir}/meta.json")


if __name__ == "__main__":
    main()