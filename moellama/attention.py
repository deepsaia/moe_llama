"""
Multi-head self-attention mechanism with optional rotary embeddings.

This module implements the attention mechanism that allows the model to
focus on different parts of the input sequence when processing each token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from moellama.layers import RotaryPositionalEmbeddings



class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking and optional RoPE.

    Attention allows each position to attend to all previous positions in
    the sequence, enabling the model to build rich contextual representations.

    The attention mechanism computes:
    1. Query, Key, Value projections for each head
    2. Attention scores = softmax(Q @ K^T / sqrt(d_k))
    3. Output = Attention scores @ V

    With causal masking, each position can only attend to itself and
    previous positions, enabling autoregressive generation.

    Args:
        dim: Model dimension (must be divisible by num_heads)
        num_heads: Number of parallel attention heads
        dropout: Dropout probability for attention weights
        use_rope: Whether to use Rotary Positional Embeddings
        max_seq_len: Maximum sequence length (for RoPE precomputation)

    Note:
        - Each head has dimension head_dim = dim / num_heads
        - Using multiple heads allows the model to attend to different
          aspects of the input simultaneously
    """

    def __init__(self, dim, num_heads, dropout=0.1, use_rope=True, max_seq_len=512):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        logger.info(
            f"Initializing MultiHeadAttention: dim={dim}, num_heads={num_heads}, "
            f"head_dim={self.head_dim}, use_rope={use_rope}, max_seq_len={max_seq_len}"
        )

        # Linear projections for queries, keys, and values
        # These learn to extract different aspects of the input
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection to combine all heads
        self.out_proj = nn.Linear(dim, dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Rotary positional embeddings (optional)
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len)

        # Scaling factor for attention scores
        # Dividing by sqrt(head_dim) prevents softmax saturation
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def apply_causal_mask(self, attn_scores):
        """
        Apply causal mask to attention scores to prevent attending to future tokens.

        Creates an upper triangular mask where:
        - True (masked) positions are set to -inf
        - False (allowed) positions remain unchanged

        After softmax, -inf becomes 0, effectively blocking attention to future tokens.

        Args:
            attn_scores: Attention scores [batch, heads, seq_len, seq_len]

        Returns:
            Masked attention scores with same shape
        """
        seq_len = attn_scores.size(-1)

        # Create upper triangular mask (diagonal=1 means above diagonal is True)
        # Example for seq_len=4:
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(attn_scores.device)

        # Set masked positions to -inf (will become 0 after softmax)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        return attn_scores

    def forward(self, x, seq_len):
        """
        Forward pass through multi-head attention.

        Process:
        1. Project input to Q, K, V
        2. Split into multiple heads
        3. Apply RoPE if enabled
        4. Compute attention scores
        5. Apply causal mask
        6. Apply softmax and dropout
        7. Multiply by values
        8. Concatenate heads and project

        Args:
            x: Input tensor [batch, seq_len, dim]
            seq_len: Current sequence length

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to query, key, value
        # Shape: [batch, seq_len, dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # Split dim into (num_heads, head_dim)
        # Shape: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to make heads the batch dimension
        # Shape: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary positional embeddings if enabled
        # RoPE encodes position information into Q and K
        if self.use_rope:
            q, k = self.rope(q, k, seq_len)

        # Compute attention scores
        # Q @ K^T gives similarity between all pairs of positions
        # Shape: [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask (prevent attending to future tokens)
        attn_scores = self.apply_causal_mask(attn_scores)

        # Compute attention weights with softmax
        # Converts scores to probabilities summing to 1
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # Weight the values by attention weights
        # Shape: [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to original dimensions
        # Transpose: [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Merge heads: [batch, seq_len, dim]
        attn_output = attn_output.view(batch_size, seq_len, self.dim)

        # Final output projection
        # Allows heads to interact and combine their information
        output = self.out_proj(attn_output)

        return output
