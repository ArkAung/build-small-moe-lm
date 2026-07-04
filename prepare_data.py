"""
Step 2: Prepare data for MoE training.

1. Downloads TinyStories (or loads a local text file if --local_file is given).
2. Trains a BPE tokenizer from scratch (small vocab, appropriate for a small model).
3. Tokenizes the corpus and writes it out as a flat uint16 binary file for
   fast memory-mapped loading during training.

Usage:
    python prepare_data.py --vocab_size 8192 --out_dir data

If you want to use your own text instead of TinyStories:
    python prepare_data.py --local_file my_corpus.txt --vocab_size 8192
"""
import argparse
import os
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm


def get_text_iterator(local_file=None, max_examples=None):
    if local_file:
        with open(local_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    else:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train")
        if max_examples:
            ds = ds.select(range(max_examples))
        for row in ds:
            text = row["text"].strip()
            if text:
                yield text


def train_tokenizer(text_iter_fn, vocab_size, out_dir):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(text_iter_fn(), trainer=trainer)

    tok_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"Tokenizer saved to {tok_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def tokenize_corpus(tokenizer, text_iter_fn, out_dir, split_name="train", val_fraction=0.01):
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    all_ids = []
    print("Tokenizing corpus...")
    for text in tqdm(text_iter_fn()):
        ids = tokenizer.encode(text).ids
        all_ids.append(bos_id)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    arr = np.array(all_ids, dtype=np.uint16)
    n_val = int(len(arr) * val_fraction)

    train_arr = arr[:-n_val] if n_val > 0 else arr
    val_arr = arr[-n_val:] if n_val > 0 else arr[:0]

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"Train tokens: {len(train_arr):,} -> {train_path}")
    print(f"Val tokens:   {len(val_arr):,} -> {val_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--local_file", type=str, default=None,
                         help="Optional path to a local .txt file (one doc per line). "
                              "If omitted, downloads TinyStories from HF.")
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap number of TinyStories examples (for a quicker first run).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    text_iter_fn = lambda: get_text_iterator(args.local_file, args.max_examples)

    tokenizer = train_tokenizer(text_iter_fn, args.vocab_size, args.out_dir)
    tokenize_corpus(tokenizer, text_iter_fn, args.out_dir)


if __name__ == "__main__":
    main()