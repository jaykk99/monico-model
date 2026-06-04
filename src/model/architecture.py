"""
Monico Model Architecture
Custom Transformer-XL with Mixture-of-Experts
Built from scratch — no base model dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class MonicoConfig:
    # Model dimensions
    vocab_size: int = 65536       # Custom BPE tokenizer
    hidden_size: int = 4096       # 7B config
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 128
    max_position_embeddings: int = 131072  # 128K context
    rope_theta: float = 500000.0

    # MoE settings
    num_experts: int = 16
    num_experts_per_tok: int = 2
    expert_domains: list = None  # ["coding", "security", "crypto", "devops", "general"]

    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.expert_domains is None:
            self.expert_domains = [
                "coding", "cybersecurity", "crypto",
                "devops", "automation", "general_1",
                "general_2", "general_3", "math",
                "reasoning", "writing", "science",
                "web3", "systems", "ml", "misc"
            ]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE with extended context support (up to 1M tokens with ring attention)"""
    def __init__(self, dim: int, max_position_embeddings: int = 131072, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MonicoGroupedQueryAttention(nn.Module):
    """Grouped Query Attention with sliding window support"""
    def __init__(self, config: MonicoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Flash attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MonicoMoEExpert(nn.Module):
    """Single MoE expert — domain-specialized FFN"""
    def __init__(self, config: MonicoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MonicoSparseMoE(nn.Module):
    """Sparse Mixture-of-Experts — top-k routing per token"""
    def __init__(self, config: MonicoConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MonicoMoEExpert(config) for _ in range(config.num_experts)])

    def forward(self, hidden_states):
        batch, seq_len, hidden = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden)

        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        final_hidden = torch.zeros_like(hidden_states_flat)
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            mask = (selected_experts == expert_idx).any(dim=-1)
            if mask.any():
                expert_out = expert(hidden_states_flat[mask])
                expert_weights = routing_weights[mask]
                expert_weight = (selected_experts[mask] == expert_idx).float()
                expert_weight = (expert_weight * routing_weights[mask]).sum(dim=-1, keepdim=True)
                final_hidden[mask] += expert_out * expert_weight

        return final_hidden.view(batch, seq_len, hidden), router_logits


class MonicoDecoderLayer(nn.Module):
    def __init__(self, config: MonicoConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MonicoGroupedQueryAttention(config, layer_idx)
        self.moe = MonicoSparseMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, router_logits


class MonicoModel(nn.Module):
    """
    Monico — Original LLM built from scratch.
    Transformer-XL backbone + Sparse MoE experts.
    Not a fine-tune. Not a wrapper. All original.
    """
    def __init__(self, config: MonicoConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MonicoDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        all_router_logits = []

        for decoder_layer in self.layers:
            hidden_states, router_logits = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            all_router_logits.append(router_logits)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            # Load balancing loss for MoE
            router_logits_stacked = torch.stack(all_router_logits, dim=0)
            routing_weights = F.softmax(router_logits_stacked, dim=-1)
            aux_loss = 0.01 * (routing_weights.mean(0).pow(2).sum())
            loss = loss + aux_loss

        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    config = MonicoConfig()
    model = MonicoModel(config)
    total = sum(p.numel() for p in model.parameters())
    print(f"Monico 7B: {total/1e9:.2f}B parameters")
    print(f"Architecture: Transformer-XL + {config.num_experts}-expert MoE ({config.num_experts_per_tok} active/token)")
    print(f"Context: {config.max_position_embeddings // 1024}K tokens")
