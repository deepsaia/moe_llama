"""
Basic building blocks for the transformer architecture.

This module contains fundamental components used throughout the model:
- RotaryPositionalEmbeddings (RoPE): Relative positional encoding
- RMSNorm: Root Mean Square Layer Normalization for stable training
- TransformerBlock: Complete transformer layer with attention and MoE
"""

import torch
import torch.nn as nn
from loguru import logger



class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) for better positional understanding.

    RoPE applies a rotation to the query and key vectors based on their position,
    providing relative positional information without adding parameters.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864

    Args:
        dim: Dimension of the embeddings (typically head_dim)
        max_seq_len: Maximum sequence length to precompute embeddings for
    """

    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        logger.debug(f"Initializing RoPE with dim={dim}, max_seq_len={max_seq_len}")

        # Calculate inverse frequencies for rotary embeddings
        # These determine the rotation frequency for each dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute position embeddings for efficiency
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, seq_len):
        """
        Pre-compute the frequency tensors for positional embeddings.

        This creates cos and sin tensors that will be used to rotate
        the query and key vectors during attention.

        Args:
            seq_len: Sequence length to precompute for
        """
        # Position indices [0, 1, ..., seq_len-1]
        position = torch.arange(seq_len, dtype=torch.float32)

        # Outer product: position * inv_freq
        # Shape: [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", position, self.inv_freq)

        # Duplicate frequencies for both halves of the embedding
        # Shape: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Pre-compute cos and sin for efficiency
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def rotate_half(self, x):
        """
        Helper function to rotate vectors by 90 degrees.

        Splits the embedding in half and swaps them with a sign flip,
        effectively creating a 90-degree rotation in the complex plane.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Rotated tensor with same shape as input
        """
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, seq_len):
        """
        Apply rotary positional embeddings to query and key tensors.

        This rotates the query and key vectors by an angle proportional
        to their position in the sequence.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            seq_len: Current sequence length

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        # Resize cos and sin to match sequence length
        cos = self.cos[:seq_len, :].to(q.device)
        sin = self.sin[:seq_len, :].to(q.device)

        # Apply rotation using the formula:
        # x_rotated = x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

    def forward(self, q, k, seq_len):
        """
        Forward pass for rotary positional embeddings.

        Args:
            q: Query tensor
            k: Key tensor
            seq_len: Current sequence length

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        return self.apply_rotary_pos_emb(q, k, seq_len)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes using only the root mean square (no mean centering),
    which is simpler and often performs just as well as LayerNorm.

    Reference: "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467

    Args:
        dim: Dimension to normalize over
        eps: Small constant for numerical stability
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (initialized to 1)
        self.weight = nn.Parameter(torch.ones(dim))
        logger.debug(f"Initializing RMSNorm with dim={dim}, eps={eps}")

    def _norm(self, x):
        """
        Apply RMS normalization.

        Formula: x / sqrt(mean(x^2) + eps)

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms

    def forward(self, x):
        """
        Forward pass with learnable scale.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized and scaled tensor with same shape
        """
        # Normalize in float32 for numerical stability
        output = self._norm(x.float()).type_as(x)
        # Apply learnable scale
        return output * self.weight


class TransformerBlock(nn.Module):
    """
    Complete Transformer block combining attention and MoE.

    This is the core building block of the model, consisting of:
    1. Multi-head self-attention with RoPE
    2. Mixture of Experts layer (instead of traditional FFN)

    Both sublayers use pre-normalization and residual connections.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_experts: Number of experts in MoE layer
        top_k: Number of experts to activate per token
        mlp_ratio: Ratio of MLP hidden dim to model dim
        dropout: Dropout probability
        use_rope: Whether to use rotary positional embeddings
        max_seq_len: Maximum sequence length
        shared_expert: Whether to include a shared expert
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_experts,
        top_k,
        mlp_ratio=4.0,
        dropout=0.1,
        use_rope=True,
        max_seq_len=512,
        shared_expert=False
    ):
        super().__init__()

        logger.info(
            f"Initializing TransformerBlock: dim={dim}, num_heads={num_heads}, "
            f"num_experts={num_experts}, top_k={top_k}, mlp_ratio={mlp_ratio}, "
            f"dropout={dropout}, use_rope={use_rope}, shared_expert={shared_expert}"
        )

        # Import here to avoid circular imports
        from moellama.attention import MultiHeadAttention
        from moellama.moe import MoELayer

        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            dim, num_heads, dropout, use_rope, max_seq_len
        )
        self.attn_norm = RMSNorm(dim)

        # Mixture of Experts layer
        self.moe = MoELayer(
            dim,
            num_experts,
            top_k,
            hidden_dim=int(dim * mlp_ratio),
            shared_expert=shared_expert,
            dropout=dropout
        )
        self.moe_norm = RMSNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=True):
        """
        Forward pass through the transformer block.

        Architecture:
        1. x = x + Attention(RMSNorm(x))
        2. x = x + MoE(RMSNorm(x))

        Args:
            x: Input tensor [batch, seq_len, dim]
            training: Whether in training mode (affects MoE routing)

        Returns:
            Tuple of (output, router_logits):
                - output: Transformed tensor [batch, seq_len, dim]
                - router_logits: Router logits for load balancing loss
        """
        batch_size, seq_len, dim = x.shape

        # Attention block with pre-normalization and residual connection
        # Pre-norm: normalize before attention
        attn_output = self.attention(x, seq_len)
        x = x + self.dropout(attn_output)
        # Post-norm: normalize after adding residual
        x = self.attn_norm(x)

        # MoE block with pre-normalization and residual connection
        moe_output, router_logits = self.moe(x, training)
        x = x + self.dropout(moe_output)
        x = self.moe_norm(x)

        return x, router_logits
