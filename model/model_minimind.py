"""
MiniMind Model Architecture Implementation.

This module implements a compact Transformer decoder-only model with modern
optimizations including:
- RoPE (Rotary Position Embeddings)
- GQA (Grouped Query Attention)
- RMSNorm (pre-normalization)
- SwiGLU activation
- Optional MoE (Mixture of Experts)
- Flash Attention support

The model is designed for educational purposes and can run on consumer hardware
while demonstrating production-grade LLM architecture patterns.
"""

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class MiniMindConfig(PretrainedConfig):
    """
    Configuration class for MiniMind model.
    
    Compatible with HuggingFace transformers format. Contains all hyperparameters
    for model architecture, training, and inference.
    
    Key parameters are organized into sections:
    - Basic architecture: hidden_size, num_layers, vocab_size
    - Attention: num_attention_heads, num_key_value_heads (GQA)
    - Position embeddings: max_position_embeddings, rope_theta, rope_scaling
    - MoE: use_moe, num_experts_per_tok, n_routed_experts
    - Optimization: flash_attn, dropout, rms_norm_eps
    """
    
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        ####################################################
        # MoE-specific configurations
        # Only active when use_moe=True
        ####################################################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Basic model configuration
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        
        # YaRN (Yet another RoPE extensioN) configuration
        # Extends context length by scaling RoPE frequencies
        # Extrapolation length = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        
        # Flash Attention for efficient computation (PyTorch 2.0+)
        self.flash_attn = flash_attn
        
        ####################################################
        # MoE (Mixture of Experts) configuration
        # Only active when use_moe=True
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # Number of experts per token (top-k routing)
        self.n_routed_experts = n_routed_experts  # Total number of routed experts
        self.n_shared_experts = n_shared_experts  # Number of shared experts (always active)
        self.scoring_func = scoring_func  # Scoring function for routing (default: 'softmax')
        self.aux_loss_alpha = aux_loss_alpha  # Auxiliary loss coefficient for load balancing
        self.seq_aux = seq_aux  # Whether to compute auxiliary loss per sequence
        self.norm_topk_prob = norm_topk_prob  # Whether to normalize top-k probabilities


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union, cast
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Alternative to LayerNorm that normalizes by RMS instead of variance.
    More stable for large models and commonly used in modern LLMs (Llama, etc.).
    
    Formula: output = weight * (x / sqrt(mean(x²) + eps))
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (same as LayerNorm's weight)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with float32 computation for stability."""
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(
    dim: int, 
    end: int = 32768,  # 32K default context
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin values for Rotary Position Embedding (RoPE).
    
    Args:
        dim: Head dimension size
        end: Maximum sequence length to precompute
        rope_base: Base frequency for RoPE (default 1M from Llama)
        rope_scaling: Optional YaRN scaling configuration for longer contexts
    
    Returns:
        Tuple of (cos_values, sin_values) tensors of shape (end, dim)
    """
    
    # Compute base frequencies (inverse frequencies)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Apply YaRN scaling if enabled (for extrapolation beyond training length)
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0)
        )
        
        # Only apply scaling if current sequence exceeds original training length
        if end / orig_max > 1.0:
            # Find correlation dimension (where wavelength > orig_max)
            corr_dim = next(
                (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), 
                dim // 2
            )
            
            # Compute beta scaling factors (linear interpolation)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # YaRN formula: λ = (β·α - β + 1)/(β·α)
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor
            )
            freqs = freqs * scale

    # Compute position embeddings for all positions
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    
    # Duplicate cos/sin to match dimension (RoPE splits head_dim in half)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        cos: Precomputed cos values
        sin: Precomputed sin values
        position_ids: Optional position indices (if not using full sequence)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
    
    Returns:
        Rotated query and key tensors
    """
    
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the hidden dims of the input.
        RoPE mechanism: splits embedding into two halves and rotates them.
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # Apply rotation: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for Grouped Query Attention (GQA).
    
    In GQA, queries have more heads than keys/values. This function repeats
    the KV heads to match query heads for attention computation.
    
    Args:
        x: KV tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        n_rep: Number of times to repeat (query_heads // kv_heads)
    
    Returns:
        Repeated tensor of shape (batch, seq_len, num_kv_heads * n_rep, head_dim)
    """
    
    if n_rep == 1:
        return x
    
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]  # Add dimension for repetition
        .expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention with GQA (Grouped Query Attention) and RoPE.
    
    Supports optional Flash Attention for efficient computation on GPUs.
    Implements KV caching for generation tasks.
    """
    
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        assert config.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"
        
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Projection layers (no bias for efficiency)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
        # Flash Attention availability check (PyTorch 2.0+)
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn
        if not self.flash:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for attention layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            position_embeddings: Precomputed (cos, sin) for RoPE
            past_key_value: Cached KV from previous generation steps
            use_cache: Whether to return updated KV cache
            attention_mask: Attention mask (0 for padded positions)
        
        Returns:
            Tuple of (output_tensor, updated_cache)
        """
        
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # Reshape to separate heads
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        # Apply RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # KV caching for generation
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        present_kv = (xk, xv) if use_cache else None

        # Reshape for attention computation
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # Use Flash Attention if available and conditions met
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention handles causal masking automatically
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(batch_size, 1, 1, -1).expand(batch_size, self.n_local_heads, seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # Causal masking for autoregressive
            )
        else:
            # Manual attention implementation
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Causal mask (upper triangular -inf)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

            # Apply padding mask if provided
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # Softmax and apply to values
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present_kv


class FeedForward(nn.Module):
    """
    SwiGLU feed-forward network (FFN).
    
    Uses SwiGLU activation: down_proj(gate_proj(x) * up_proj(x))
    More efficient than standard FFN with ReLU.
    """
    
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        
        # Calculate intermediate size (8/3 * hidden_size, multiple of 64)
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # SwiGLU projections
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 'silu' for SwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU computation: down_proj(silu(gate_proj(x)) * up_proj(x))"""
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    MoE (Mixture of Experts) routing gate.
    
    Routes each token to top-k experts based on learned gating weights.
    Implements load balancing via auxiliary loss.
    """
    
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize gating weights with Kaiming uniform."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts and compute load balancing loss.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
        
        Returns:
            Tuple of:
            - topk_idx: Expert indices for each token
            - topk_weight: Routing weights for each expert
            - aux_loss: Load balancing auxiliary loss
        """
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Compute routing scores
        logits = F.linear(hidden_states, self.weight, None)
        
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'Unsupported MoE gating function: {self.scoring_func}')

        # Select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # Normalize top-k probabilities if enabled
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # Compute auxiliary load balancing loss during training
        aux_loss = 0.0
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(batch_size, -1)
            
            if self.seq_aux:
                # Sequence-level auxiliary loss
                scores_for_seq_aux = scores_for_aux.view(batch_size, seq_len, -1)
                ce = torch.zeros(batch_size, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, 
                    topk_idx_for_aux_loss,
                    torch.ones(batch_size, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token-level auxiliary loss
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
                
        return topk_idx, topk_weight, aux_loss # type: ignore 


class MOEFeedForward(nn.Module):
    """
    MoE (Mixture of Experts) Feed-Forward layer.
    
    Combines multiple expert FFNs with a routing mechanism.
    Supports shared experts (always active) and routed experts (top-k).
    """
    
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # Routed experts (selected per token)
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        
        # Gating network for routing
        self.gate = MoEGate(config)
        
        # Shared experts (always active, similar to DeepSeek MoE)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Routing logic:
        1. Compute routing weights and expert indices
        2. Dispatch tokens to selected experts
        3. Aggregate expert outputs weighted by routing scores
        4. Add shared expert outputs (if any)
        """
        
        identity = x
        orig_shape = x.shape
        batch_size, seq_len, _ = x.shape
        
        # Get routing decisions
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # Reshape for expert computation
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # Training: compute all expert outputs (simpler but less efficient)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # Each expert processes its assigned tokens
            for i, expert in enumerate(self.experts):
                expert_mask = (flat_topk_idx == i)
                if expert_mask.any():
                    y[expert_mask] = expert(x[expert_mask]).to(y.dtype)
            
            # Weighted sum of expert outputs
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # Inference: optimized routing with less computation
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # Add shared expert outputs (always active)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        # Store auxiliary loss for backward pass
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(
        self, 
        x: torch.Tensor, 
        flat_expert_indices: torch.Tensor, 
        flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficient MoE inference without computing all experts.
        
        Groups tokens by expert assignment for batched computation.
        Much faster than naive implementation for large batches.
        """
        
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # Process each expert's assigned tokens in batch
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # Skip if no tokens for this expert
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            
            # Apply routing weights and accumulate
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(
                0, 
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), 
                expert_out
            )

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    Single Transformer block with attention and feed-forward layers.
    
    Implements pre-normalization with RMSNorm (unlike original Transformers
    which use post-normalization). Architecture: Attention + MLP with skip connections.
    """
    
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        
        # Attention components
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)
        
        # Normalization layers (pre-norm architecture)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # FFN (dense or MoE)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for a single transformer block."""
        
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        hidden_states += residual
        
        # Feed-forward with residual connection
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind Transformer model (decoder-only).
    
    Embeds input tokens, applies multiple transformer blocks, and final normalization.
    Precomputes and caches RoPE embeddings for efficiency.
    """
    
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # Embedding and dropout
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            MiniMindBlock(layer_id, config) 
            for layer_id in range(self.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Precompute RoPE embeddings (cached in buffer)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass for the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Mask for padding tokens
            past_key_values: Cached KV from previous steps (generation)
            use_cache: Whether to return updated cache
        
        Returns:
            Tuple of (hidden_states, updated_cache, aux_loss)
        """
        
        batch_size, seq_length = input_ids.shape
        
        # Handle cache format
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers) #type: ignore 
        
        # Compute start position for RoPE (for cached generation)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0 #type: ignore 
        
        # Embed tokens
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # Get position embeddings for current positions (ensure buffers are treated as Tensors)
        freqs_cos = cast(torch.Tensor, self.freqs_cos)
        freqs_sin = cast(torch.Tensor, self.freqs_sin)
        position_embeddings = (
            freqs_cos[start_pos:start_pos + seq_length],
            freqs_sin[start_pos:start_pos + seq_length]
        )

        # Pass through transformer blocks
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)): #type: ignore 
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Sum auxiliary losses from MoE layers
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, cast(torch.Tensor, aux_loss)


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind model for causal language modeling.
    
    Wraps the base model with a language modeling head. Supports generation
    via HuggingFace's GenerationMixin. Ties embedding weights with lm_head.
    """
    
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # Model and language modeling head
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # Tie input and output embeddings (common practice, saves parameters)
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # Output container for consistent interface
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs
            attention_mask: Padding mask
            past_key_values: Cached KV
            use_cache: Whether to return cache
            logits_to_keep: Number of logits to compute (for efficiency)
        
        Returns:
            CausalLMOutputWithPast with logits, hidden_states, aux_loss, past_key_values
        """
        
        # Get hidden states from base model
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # Compute logits (only for requested positions if specified)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        
        # Package outputs
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT