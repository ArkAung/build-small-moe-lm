"""
Step 3: A small Mixture-of-Experts decoder-only transformer in MLX.

Design choices (deliberately simple, for learning purposes):
- Pre-norm transformer blocks, RMSNorm, RoPE positional embeddings, SwiGLU experts.
- Top-k token routing (k=2 by default) over N experts.
- Switch-Transformer-style load-balancing auxiliary loss to prevent expert collapse.
- No expert capacity dropping (simplest correct version) -- fine at this scale.
  Add capacity-based dropping later if you want to study that failure mode.
"""
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8          # set < n_heads for grouped-query attention
    max_seq_len: int = 1024
    n_experts: int = 8
    top_k: int = 2
    expert_hidden_mult: float = 3.0   # expert FFN hidden dim = d_model * mult (SwiGLU trims this internally)
    aux_loss_coef: float = 0.01
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x, mask=None, cache=None, capture=None):
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k_cache, v_cache = cache
            q = self.rope(q, offset=k_cache.shape[2])
            k = self.rope(k, offset=k_cache.shape[2])
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        if self.n_kv_heads < self.n_heads:
            reps = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, reps, axis=1)
            v = mx.repeat(v, reps, axis=1)

        if capture is not None:
            # Unfused path so we can pull out the actual attention weights for
            # inspection. Only used by inspect.py -- training always uses the
            # fused kernel below, which never materializes these weights.
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            if mask is not None:
                scores = scores + mask
            weights = mx.softmax(scores.astype(mx.float32), axis=-1)
            out = (weights.astype(v.dtype)) @ v
            capture["attn_weights"] = weights  # (B, n_heads, T_q, T_k)
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out), (k, v)


class Expert(nn.Module):
    """A single SwiGLU FFN expert."""
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)   # gate
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)   # up
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)   # down

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    """
    Top-k routed mixture of experts.

    Returns (output, aux_loss) where aux_loss is the load-balancing loss
    that should be added to the main LM loss during training.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        hidden_dim = int(cfg.d_model * cfg.expert_hidden_mult)

        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = [Expert(cfg.d_model, hidden_dim) for _ in range(cfg.n_experts)]

    def __call__(self, x, capture=None):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)                       # (N, D), N = B*T
        N = x_flat.shape[0]

        router_logits = self.gate(x_flat)                # (N, n_experts)
        router_probs = mx.softmax(router_logits, axis=-1)

        # top-k selection (mx has no fused topk-with-values op, so do it via argsort)
        sorted_idx = mx.argsort(-router_probs, axis=-1)              # (N, n_experts)
        topk_idx = sorted_idx[:, : self.top_k]                       # (N, top_k)
        topk_probs = mx.take_along_axis(router_probs, topk_idx, axis=-1)  # (N, top_k)
        # renormalize the top-k probabilities so they sum to 1 (standard practice)
        topk_probs = topk_probs / mx.sum(topk_probs, axis=-1, keepdims=True)

        # dense-compute-and-mask approach: simplest correct implementation on MLX.
        # Not FLOP-optimal (computes every expert for every token) but avoids
        # gather/scatter edge cases while you're learning the mechanics.
        # Swap in a sparse-dispatch version once this is working end to end.
        out = mx.zeros_like(x_flat)
        for e in range(self.n_experts):
            expert_mask = (topk_idx == e).astype(x.dtype)            # (N, top_k), 1 where this token routes to expert e
            weight = mx.sum(topk_probs * expert_mask, axis=-1, keepdims=True)  # (N, 1)
            # skip compute is not free in MLX's graph mode, but weight=0 zeroes contribution correctly
            expert_out = self.experts[e](x_flat)                     # (N, D)
            out = out + expert_out * weight

        out = out.reshape(B, T, D)

        # --- load balancing auxiliary loss (Switch Transformer style) ---
        # fraction of tokens routed to each expert (top-1 proxy is standard)
        top1_idx = sorted_idx[:, 0]
        # one-hot via equality trick (no in-place scatter on mlx arrays):
        expert_range = mx.arange(self.n_experts).reshape(1, -1)
        one_hot_top1 = (top1_idx.reshape(-1, 1) == expert_range).astype(mx.float32)

        tokens_per_expert = mx.mean(one_hot_top1, axis=0)             # (n_experts,) fraction routed
        prob_per_expert = mx.mean(router_probs, axis=0)                # (n_experts,) average router prob
        aux_loss = self.n_experts * mx.sum(tokens_per_expert * prob_per_expert)

        if capture is not None:
            capture["router_probs"] = router_probs.reshape(B, T, self.n_experts)
            capture["topk_idx"] = topk_idx.reshape(B, T, self.top_k)
            capture["topk_probs"] = topk_probs.reshape(B, T, self.top_k)

        return out, aux_loss


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.moe_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = MoELayer(cfg)

    def __call__(self, x, mask=None, cache=None, capture=None):
        attn_capture = {} if capture is not None else None
        h, new_cache = self.attn(self.attn_norm(x), mask=mask, cache=cache, capture=attn_capture)
        x = x + h

        moe_capture = {} if capture is not None else None
        moe_out, aux_loss = self.moe(self.moe_norm(x), capture=moe_capture)
        x = x + moe_out

        if capture is not None:
            capture["attn"] = attn_capture
            capture["moe"] = moe_capture

        return x, aux_loss, new_cache


class MoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, tokens, cache=None, capture=False):
        """
        capture=True additionally returns a list (one dict per layer) containing
        raw attention weights and router decisions, for use by inspect.py.
        Training/generation should leave this False (default) -- capturing forces
        an unfused, slower attention path.
        """
        B, T = tokens.shape
        x = self.tok_emb(tokens)

        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)

        total_aux_loss = 0.0
        new_caches = []
        captures = [] if capture else None
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            block_capture = {} if capture else None
            x, aux_loss, block_new_cache = block(x, mask=mask, cache=block_cache, capture=block_capture)
            total_aux_loss = total_aux_loss + aux_loss
            new_caches.append(block_new_cache)
            if capture:
                captures.append(block_capture)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        avg_aux_loss = total_aux_loss / len(self.blocks)
        return logits, avg_aux_loss, new_caches, captures

    def num_params(self):
        leaves = self.parameters()
        def count(d):
            total = 0
            if isinstance(d, dict):
                for v in d.values():
                    total += count(v)
            elif isinstance(d, list):
                for v in d:
                    total += count(v)
            else:
                total += d.size
            return total
        return count(leaves)