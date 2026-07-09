"""
Instruction fine-tune a pretrained MoE checkpoint on (instruction, response)
examples produced by prepare_instruct_data.py.

Key differences from train.py (pretraining):
  - Warm-starts from an existing checkpoint instead of random init.
  - Each training example is one full (instruction, response) pair, padded
    to a fixed length -- not a random crop of a flat token stream.
  - The loss is masked: only response tokens (and the trailing eos)
    contribute, so the model isn't penalized for "failing" to predict the
    fixed instruction template.
  - Right-padding + causal attention means padded positions can never be
    attended to by real tokens (padding always comes after them in the
    sequence), so no separate attention/padding mask is needed -- masking
    the loss alone is sufficient.

Usage:
    python finetune.py \
        --init_checkpoint checkpoints/final.safetensors \
        --init_config checkpoints/config.json \
        --tokenizer data/tokenizer.json \
        --data_dir data_instruct \
        --out_dir checkpoints_instruct \
        --steps 1000 --lr 3e-5
"""
import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx
from mlx import nn
import mlx.optimizers as optim

from common import load_model_and_tokenizer


class InstructDataset:
    """Memory-maps padded (ids, loss_mask) example arrays for random-batch sampling."""
    def __init__(self, ids_path, mask_path, max_len):
        self.ids = np.memmap(ids_path, dtype=np.uint16, mode="r").reshape(-1, max_len)
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r").reshape(-1, max_len)
        self.max_len = max_len
        self.n = self.ids.shape[0]

    def get_batch(self, batch_size):
        """Sample `batch_size` full padded examples (not random crops)."""
        idx = np.random.randint(0, self.n, size=batch_size)
        ids = self.ids[idx].astype(np.int32)
        mask = self.mask[idx].astype(np.float32)
        x = ids[:, :-1]
        y = ids[:, 1:]
        y_mask = mask[:, 1:]  # mask[i] means "does ids[i] count", aligned to y = ids[1:]
        return mx.array(x), mx.array(y), mx.array(y_mask)


def masked_loss_fn(model, x, y, y_mask, aux_coef):
    """Cross-entropy averaged only over response tokens, plus the usual aux loss."""
    logits, aux_loss, _, _ = model(x)
    per_token_ce = nn.losses.cross_entropy(logits, y, reduction="none")
    denom = mx.maximum(mx.sum(y_mask), 1.0)  # avoid divide-by-zero on a pathological batch
    ce = mx.sum(per_token_ce * y_mask) / denom
    total = ce + aux_coef * aux_loss
    return total, (ce, aux_loss)


def evaluate(model, val_ds, batch_size, n_batches=20):
    """Average masked cross-entropy over a few random validation batches."""
    losses = []
    for _ in range(n_batches):
        x, y, y_mask = val_ds.get_batch(batch_size)
        logits, _, _, _ = model(x)
        per_token_ce = nn.losses.cross_entropy(logits, y, reduction="none")
        denom = mx.maximum(mx.sum(y_mask), 1.0)
        ce = mx.sum(per_token_ce * y_mask) / denom
        losses.append(ce.item())
    return float(np.mean(losses))


def main():
    """Parse args, warm-start from a pretrained checkpoint, and fine-tune on instruction data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--init_config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data_instruct")
    parser.add_argument("--out_dir", type=str, default="checkpoints_instruct")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    # Fine-tuning uses a much lower LR than pretraining: we're nudging an
    # already-competent model, not learning language from scratch, and a
    # pretraining-scale LR would risk wrecking what it already knows.
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--aux_loss_coef", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg, _tok = load_model_and_tokenizer(args.init_checkpoint, args.init_config, args.tokenizer)
    mx.eval(model.parameters())
    print(f"Warm-started from {args.init_checkpoint}: {model.num_params() / 1e6:.1f}M total params")

    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    max_len = meta["max_len"]
    print(f"Instruction template used at data-prep time: {meta['instruction_template']!r}")

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_ds = InstructDataset(
        os.path.join(args.data_dir, "instruct_train_ids.bin"),
        os.path.join(args.data_dir, "instruct_train_mask.bin"),
        max_len,
    )
    val_ds = InstructDataset(
        os.path.join(args.data_dir, "instruct_val_ids.bin"),
        os.path.join(args.data_dir, "instruct_val_mask.bin"),
        max_len,
    )
    print(f"Train examples: {train_ds.n}, val examples: {val_ds.n}")

    def lr_schedule(step):
        """Linear warmup then cosine decay to 10% of the peak learning rate."""
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        progress = min(progress, 1.0)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    loss_and_grad_fn = nn.value_and_grad(model, masked_loss_fn)

    start_time = time.time()
    running_ce, running_aux = [], []

    for step in range(1, args.steps + 1):
        optimizer.learning_rate = lr_schedule(step)

        x, y, y_mask = train_ds.get_batch(args.batch_size)
        (_, (ce, aux_loss)), grads = loss_and_grad_fn(model, x, y, y_mask, args.aux_loss_coef)

        grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        running_ce.append(ce.item())
        running_aux.append(aux_loss.item())

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            if step > args.log_every:
                steps_per_sec = args.log_every / elapsed
            else:
                steps_per_sec = 0
            print(f"step {step:6d} | lr {optimizer.learning_rate.item():.2e} | "
                  f"ce {np.mean(running_ce):.4f} | ppl {np.exp(np.mean(running_ce)):.2f} | "
                  f"aux {np.mean(running_aux):.4f} | {steps_per_sec:.2f} steps/s")
            running_ce, running_aux = [], []
            start_time = time.time()

        if step % args.eval_every == 0:
            val_ce = evaluate(model, val_ds, args.batch_size)
            print(f"  [eval] step {step} | val_ce {val_ce:.4f} | val_ppl {np.exp(val_ce):.2f}")

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"step_{step}.safetensors")
            model.save_weights(ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    model.save_weights(os.path.join(args.out_dir, "final.safetensors"))
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()