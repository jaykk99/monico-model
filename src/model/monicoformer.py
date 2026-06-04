"""
MonicoFormer — Custom decoder-only transformer for Monico LLM
From scratch: no HuggingFace weights, custom architecture.

Architecture highlights:
- Grouped Query Attention (GQA)
- SwiGLU FFN
- RMSNorm
- Rotary Position Embeddings (RoPE) with extended theta
- Flash Attention 2 compatible
- Sliding window layers
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from einops import rearrange

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class MonicoConfig:
    # Model dimensions
    vocab_size: int = 128_000
    hidden_size: int = 4096
    intermediate_size: int = 14336      # ~3.5x hidden (SwiGLU convention)
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8        # GQA: 4 queries per KV head
    head_dim: int = 128
    max_position_embeddings: int = 131072  # 128K context

    # Architecture
    rms_norm_eps: float = 1e-5
    rope_theta: float = 5_000_000.0     # Extended for long context
    sliding_window: int = 4096          # Sliding window every N layers
    sliding_window_every_n: int = 4     # Apply sliding window every 4th layer

    # Training
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    initializer_range: float = 0.02

    # Inference
    use_cache: bool = True

    @property
    def num_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def monico_7b(cls):
        return cls(
            hidden_size=4096, intermediate_size=14336,
            num_hidden_layers=32, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128,
        )

    @classmethod
    def monico_13b(cls):
        return cls(
            hidden_size=5120, intermediate_size=17920,
            num_hidden_layers=40, num_attention_heads=40,
            num_key_value_heads=8, head_dim=128,
        )

    @classmethod
    def monico_70b(cls):
        return cls(
            hidden_size=8192, intermediate_size=28672,
            num_hidden_layers=80, num_attention_heads=64,
            num_key_value_heads=8, head_dim=128,
        )


# ─── RMSNorm ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ─── Rotary Embeddings ──────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 5_000_000.0,
                 max_seq_len: int = 131072):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q, k, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, cos, sin):
    return x * cos + rotate_half(x) * sin


# ─── Grouped Query Attention ─────────────────────────────────────────────────

class MonicoAttention(nn.Module):
    def __init__(self, config: MonicoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = config.num_groups
        self.head_dim = config.head_dim

        # Determine if this is a sliding window layer
        self.use_sliding_window = (layer_idx % config.sliding_window_every_n == 0)
        self.sliding_window = config.sliding_window

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, theta=config.rope_theta,
                                       max_seq_len=config.max_position_embeddings)

    def forward(self, hidden_states, attention_mask=None, past_kv=None, use_cache=False):
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q, k = self.rotary(q, k, seq_len=T)

        # KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Expand KV for GQA
        if self.num_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
                B, self.num_heads, k.shape[2], self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
                B, self.num_heads, v.shape[2], self.head_dim)

        # Attention
        if HAS_FLASH and self.config.use_flash_attention and hidden_states.is_cuda:
            # Flash Attention path
            q = q.transpose(1, 2)  # B T H D
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            window = (self.sliding_window, self.sliding_window) if self.use_sliding_window else (-1, -1)
            attn_out = flash_attn_func(q, k, v, causal=True, window_size=window)
            attn_out = attn_out.reshape(B, T, self.num_heads * self.head_dim)
        else:
            # Standard scaled dot-product attention
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            causal = torch.triu(torch.ones(T, k.shape[2], device=q.device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)

        return self.o_proj(attn_out), new_kv


# ─── SwiGLU FFN ─────────────────────────────────────────────────────────────

class MonicoMLP(nn.Module):
    def __init__(self, config: MonicoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─── Decoder Layer ───────────────────────────────────────────────────────────

class MonicoDecoderLayer(nn.Module):
    def __init__(self, config: MonicoConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MonicoAttention(config, layer_idx)
        self.mlp = MonicoMLP(config)
        self.input_layernorm  = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_kv=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states, attention_mask=attention_mask,
            past_kv=past_kv, use_cache=use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv


# ─── Full Model ───────────────────────────────────────────────────────────────

class MonicoModel(nn.Module):
    def __init__(self, config: MonicoConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MonicoDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, past_kvs=None, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        new_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs else None
            hidden_states, new_kv = layer(hidden_states, attention_mask, past_kv, use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        return self.norm(hidden_states), new_kvs


class MonicoForCausalLM(nn.Module):
    def __init__(self, config: MonicoConfig):
        super().__init__()
        self.config = config
        self.model = MonicoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids, attention_mask=None, labels=None,
                past_kvs=None, use_cache=False):
        hidden_states, new_kvs = self.model(input_ids, attention_mask, past_kvs, use_cache)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "past_key_values": new_kvs}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.7,
                 top_p=0.9, top_k=50, do_sample=True, repetition_penalty=1.1):
        """Simple generation loop with sampling."""
        past_kvs = None
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            inp = generated if past_kvs is None else generated[:, -1:]
            out = self.forward(inp, past_kvs=past_kvs, use_cache=True)
            past_kvs = out["past_key_values"]
            logits = out["logits"][:, -1, :]  # Last token

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            if do_sample:
                logits = logits / temperature
                # Top-k
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                # Top-p nucleus sampling
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = sorted_probs.cumsum(dim=-1)
                mask = cum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.gather(sorted_idx, -1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop at EOS (token 2 by default)
            if next_token.item() == 2:
                break

        return generated


# ─── Param count utility ──────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e9:.2f}B | Trainable: {trainable/1e9:.2f}B")
    return total, trainable


if __name__ == "__main__":
    cfg = MonicoConfig.monico_7b()
    print(f"MonicoFormer 7B config: {cfg}")
    model = MonicoForCausalLM(cfg)
    count_params(model)
    # Quick forward pass test
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(x, labels=x)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Logits shape: {out['logits'].shape}")
    print("✓ MonicoFormer forward pass OK")
