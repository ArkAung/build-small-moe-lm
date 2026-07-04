# Small MoE LLM from scratch (on MLX)

A minimal but complete pipeline to train a small Mixture-of-Experts transformer
from scratch using MLX. Built for learning the mechanics of MoE
(routing, load balancing, expert specialization).

## 0. Setup

```bash
uv sync
```

## 1. Prepare data + tokenizer

```bash
# quick smoke-test run (few minutes)
uv run python prepare_data.py --vocab_size 8192 --max_examples 50000

# full TinyStories corpus (~2M stories, run once you trust the pipeline)
uv run python prepare_data.py --vocab_size 8192
```

Produces `data/tokenizer.json`, `data/train.bin`, `data/val.bin`.

Want to use your own text instead? Put one document per line in a `.txt` file and:
```bash
uv run python prepare_data.py --local_file mycorpus.txt --vocab_size 8192
```

### Inspecting the tokenizer (for teaching / understanding)

Before touching the model at all, it's worth generating an interactive view
of what the tokenizer actually learned:

```bash
uv run python inspect_tokenizer.py --tokenizer data/tokenizer.json --out tokenizer_inspection.html
```

Open `tokenizer_inspection.html` in a browser. It shows:
- **Vocab stats** and a byte-length histogram — how many single-byte vs.
  multi-byte pieces made it into the vocab.
- **Merges in rank order**, browsable by range (earliest / early-mid / middle
  / late / final) — this is the clearest way to *see* BPE training progress
  from raw byte pairs toward whole common words.
- **A live tokenizer playground** — type anything and watch it segment in
  real time. This re-implements the exact learned BPE merges directly in
  the browser

## 2. Sanity-check training run

Before committing real time, run a tiny config for ~200 steps and confirm
loss decreases and nothing crashes:

```bash
uv run python train.py --steps 200 --d_model 256 --n_layers 4 --n_experts 4 \
    --batch_size 16 --seq_len 256 --eval_every 50 --save_every 200
```

You should see `ce` (cross-entropy) trending down and `ppl` (perplexity)
dropping from ~vocab_size initially toward something much smaller.

## 3. Real training run

```bash
uv run python train.py \
    --steps 5000 \
    --d_model 512 --n_layers 8 --n_heads 8 \
    --n_experts 8 --top_k 2 \
    --batch_size 32 --seq_len 512 \
    --lr 3e-4 --warmup_steps 200 \
    --eval_every 200 --save_every 500
```

## 4. Generate text

```bash
uv run python generate.py \
    --checkpoint checkpoints/final.safetensors \
    --config checkpoints/config.json \
    --tokenizer data/tokenizer.json \
    --prompt "Once upon a time" --max_tokens 200
```

## 5. Inspect internals (for teaching / understanding)

Once you have a checkpoint, generate an interactive inspection view for any
prompt:

```bash
uv run python inspect_model.py \
    --checkpoint checkpoints/final.safetensors \
    --config checkpoints/config.json \
    --tokenizer data/tokenizer.json \
    --prompt "Once upon a time there was a" \
    --out inspection.html
```

Open `inspection.html`. It shows,
per layer:
- **Attention heatmap**, selectable by head — which tokens attend to which.
- **Router decisions** — hover any token to see its top-k expert assignment
  and confidence; the chip color shows its top-1 expert.
- **Expert utilization** — bar chart of how many tokens (in this one prompt)
  went to each expert. Flat bars = balanced routing; one or two tall bars =
  router collapse.

**Note:** capturing internals forces an unfused (slower, non-`mx.fast`)
attention computation so the raw weights can be pulled out. This is fine
for a single inspection prompt but should never be used during training.

## What to watch during training

- **`ce` / `ppl`**: standard language modeling loss/perplexity. Should fall
  steadily; on TinyStories with this size model you should see coherent-ish
  short sentences emerge within a few thousand steps.
- **`aux`**: the load-balancing loss. If this stays high/flat, your router
  has collapsed onto a few experts. If it's near zero from step 1, the
  router may not be learning anything useful either — check it's moving.

## Known simplification: dense compute in the MoE layer

`model.py`'s `MoELayer` computes **every expert on every token** and masks
out the unselected contributions (see comments in the file). This is the
easiest way to get correct routing/gradients working in MLX without
wrestling with gather/scatter edge cases, but it means you aren't actually
getting the compute savings that make MoE attractive in production — you're
paying dense-model FLOPs for MoE-model quality/capacity.

**Good next exercise once this trains correctly:** convert the MoE layer to
real sparse dispatch — sort tokens by assigned expert, gather them into
per-expert batches, run each expert only on its assigned tokens, then
scatter results back. This is genuinely the most instructive part of
building an MoE and worth doing by hand once you've seen the dense version
work end to end.

## Other extensions worth trying

- **Expert capacity + token dropping**: cap how many tokens each expert can
  take per batch; drop (or reroute) overflow. Standard in production MoE,
  and a good way to see the load-balancing loss actually matter.
- **Shared expert**: add one always-active expert alongside the routed ones
  (DeepSeek-MoE style) and compare specialization patterns.
- **Routing visualization**: log which expert each token goes to and plot
  it against token identity/position — this is where you'll actually *see*
  specialization (e.g., punctuation vs. content words, or domain splits if
  you mix multiple text domains in step 1).
- **Scale the active/total ratio**: keep active params fixed, increase
  `n_experts` and total params, and see how far quality improves for free.