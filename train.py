"""
MoE transformer trainer
"""
import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model import MoETransformer, ModelConfig


class BinDataset:
    """Memory-maps a uint16 .bin token file and yields random contiguous chunks."""
    def __init__(self, path, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def get_batch(self, batch_size):
        max_start = len(self.data) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([self.data[s : s + self.seq_len] for s in starts]).astype(np.int32)
        y = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)


def loss_fn(model, x, y, aux_coef):
    logits, aux_loss, _, _ = model(x)
    ce = nn.losses.cross_entropy(logits, y, reduction="mean")
    total = ce + aux_coef * aux_loss
    return total, (ce, aux_loss)


def evaluate(model, val_ds, seq_len, batch_size, n_batches=20):
    losses = []
    for _ in range(n_batches):
        x, y = val_ds.get_batch(batch_size)
        logits, aux_loss, _, _ = model(x)
        ce = nn.losses.cross_entropy(logits, y, reduction="mean")
        losses.append(ce.item())
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # model size args
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # figure out vocab size from the tokenizer we trained in step 2
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_heads,
        max_seq_len=args.seq_len,
        n_experts=args.n_experts,
        top_k=args.top_k,
        aux_loss_coef=args.aux_loss_coef,
    )
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    model = MoETransformer(cfg)
    mx.eval(model.parameters())
    print(f"Model initialized: {model.num_params() / 1e6:.1f}M total params, vocab_size={vocab_size}")

    train_ds = BinDataset(os.path.join(args.data_dir, "train.bin"), args.seq_len)
    val_ds = BinDataset(os.path.join(args.data_dir, "val.bin"), args.seq_len)

    def lr_schedule(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        # simple cosine decay to 10% of peak lr
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        progress = min(progress, 1.0)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    start_time = time.time()
    running_ce, running_aux = [], []

    for step in range(1, args.steps + 1):
        optimizer.learning_rate = lr_schedule(step)

        x, y = train_ds.get_batch(args.batch_size)
        (loss, (ce, aux_loss)), grads = loss_and_grad_fn(model, x, y, args.aux_loss_coef)

        # gradient clipping by global norm
        grads, _ = optim.clip_grad_norm(grads, args.grad_clip)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        running_ce.append(ce.item())
        running_aux.append(aux_loss.item())

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (args.batch_size * args.seq_len * args.log_every) / elapsed if step > args.log_every else 0
            print(f"step {step:6d} | lr {optimizer.learning_rate.item():.2e} | "
                  f"ce {np.mean(running_ce):.4f} | ppl {np.exp(np.mean(running_ce)):.2f} | "
                  f"aux {np.mean(running_aux):.4f} | {tok_per_sec:.0f} tok/s")
            running_ce, running_aux = [], []
            start_time = time.time()

        if step % args.eval_every == 0:
            val_ce = evaluate(model, val_ds, args.seq_len, args.batch_size)
            print(f"  [eval] step {step} | val_ce {val_ce:.4f} | val_ppl {np.exp(val_ce):.2f}")

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"step_{step}.safetensors")
            model.save_weights(ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    model.save_weights(os.path.join(args.out_dir, "final.safetensors"))
    print("Training complete.")


if __name__ == "__main__":
    main()