import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
import os
import requests
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import copy
import time
import datasets
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llama4_moe.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLaMA4MoE")

class CharacterTokenizer:
    """Character-level tokenizer as mentioned in the knowledge base"""
    
    def __init__(self, text=None, vocab_file=None):
        if vocab_file:
            self.load_vocab(vocab_file)
            logger.info(f"Loaded vocabulary from {vocab_file}, size: {self.vocab_size}")
        elif text:
            self.build_vocab(text)
            logger.info(f"Built vocabulary from text, size: {self.vocab_size}")
        else:
            raise ValueError("Either text or vocab_file must be provided")
    
    def build_vocab(self, text):
        """Build vocabulary from text"""
        logger.debug("Building vocabulary from text")
        # Get all unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mapping from characters to integers and vice versa
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
        # Add special tokens to vocabulary
        special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token
                
        self.vocab_size = len(self.stoi)
        self.pad_token_id = self.stoi[self.pad_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.unk_token_id = self.stoi[self.unk_token]
    
    def encode(self, text):
        """Convert text to token IDs"""
        return [self.stoi.get(ch, self.unk_token_id) for ch in text]
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        return ''.join([self.itos.get(idx, self.unk_token) for idx in token_ids])
    
    def __len__(self):
        return self.vocab_size
    
    def save_vocab(self, file_path):
        """Save vocabulary to file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for char, idx in sorted(self.stoi.items(), key=lambda x: x[1]):
                # Escape special characters for reliable parsing
                escaped_char = char
                if char == '\t':
                    escaped_char = '\\t'
                elif char == '\n':
                    escaped_char = '\\n'
                elif char == '\r':
                    escaped_char = '\\r'
                elif char == '\\':
                    escaped_char = '\\\\'
                
                f.write(f"{escaped_char}\t{idx}\n")
        logger.info(f"Vocabulary saved to {file_path}")
    
    def load_vocab(self, file_path):
        """Load vocabulary from file"""
        logger.info(f"Loading vocabulary from {file_path}")
        self.stoi = {}
        self.itos = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                        
                    # Split on the first tab only (in case the character is a tab)
                    parts = line.split('\t', 1)
                    
                    if len(parts) != 2:
                        logger.warning(f"Skipping invalid line {line_number} in vocabulary file: '{line}'")
                        continue
                        
                    escaped_char, idx_str = parts
                    
                    # Unescape special characters
                    char = escaped_char
                    if char == '\\t':
                        char = '\t'
                    elif char == '\\n':
                        char = '\n'
                    elif char == '\\r':
                        char = '\r'
                    elif char == '\\\\':
                        char = '\\'
                    
                    try:
                        idx = int(idx_str)
                        self.stoi[char] = idx
                        self.itos[idx] = char
                    except ValueError:
                        logger.warning(f"Invalid index on line {line_number}: '{idx_str}'")
                        continue
            
            if len(self.stoi) == 0:
                raise ValueError("No valid vocabulary entries found in file")
                
            self.vocab_size = len(self.stoi)
            self.pad_token = '<pad>'
            self.eos_token = '<eos>'
            self.unk_token = '<unk>'
            
            # Set token IDs with defaults if tokens don't exist
            self.pad_token_id = self.stoi.get(self.pad_token, 0)
            self.eos_token_id = self.stoi.get(self.eos_token, 1)
            self.unk_token_id = self.stoi.get(self.unk_token, 2)
            
            logger.info(f"Successfully loaded vocabulary with {self.vocab_size} tokens")
            
        except FileNotFoundError:
            logger.error(f"Vocabulary file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading vocabulary from {file_path}: {str(e)}")
            raise


class RotaryPositionalEmbeddings(nn.Module):
    """Rotary Positional Embeddings (RoPE) as mentioned in the knowledge base"""
    
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        logger.debug(f"Initializing RotaryPositionalEmbeddings with dim={dim}, max_seq_len={max_seq_len}")
        
        # Calculate frequencies for rotary embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute position embeddings for efficiency
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len):
        """Pre-compute the frequency tensors for positional embeddings"""
        # Position indices [0, 1, ..., seq_len-1]
        position = torch.arange(seq_len, dtype=torch.float32)
        
        # Outer product: position * inv_freq
        freqs = torch.einsum("i,j->ij", position, self.inv_freq)
        
        # Convert to complex numbers (cos + i*sin)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
    
    def rotate_half(self, x):
        """Helper function to rotate vectors by 90 degrees"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, seq_len):
        """Apply rotary positional embeddings to query and key tensors"""
        # Resize cos and sin to match sequence length
        cos = self.cos[:seq_len, :].to(q.device)
        sin = self.sin[:seq_len, :].to(q.device)
        
        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward(self, q, k, seq_len):
        """Forward pass for rotary positional embeddings"""
        return self.apply_rotary_pos_emb(q, k, seq_len)


class RMSNorm(nn.Module):
    """RMS Normalization as mentioned in the knowledge base"""
    
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        logger.debug(f"Initializing RMSNorm with dim={dim}, eps={eps}")
    
    def _norm(self, x):
        """RMS normalization"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        """Forward pass"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Expert(nn.Module):
    """Expert network - a simple feed-forward network as mentioned in the knowledge base"""
    
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        logger.debug(f"Initializing Expert with dim={dim}, hidden_dim={hidden_dim}, dropout={dropout}")
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    """Router network that selects which experts to use for each token"""
    
    def __init__(self, dim, num_experts, top_k, capacity_factor=1.0, 
                 add_noise=True, noise_epsilon=0.1, shared_expert=False):
        """
        Args:
            dim: Input dimension
            num_experts: Total number of experts
            top_k: Number of experts to select per token
            capacity_factor: Factor to determine expert capacity
            add_noise: Whether to add noise to router scores (to prevent expert collapse)
            noise_epsilon: Noise magnitude
            shared_expert: Whether to include a shared expert that always processes every token
        """
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.shared_expert = shared_expert
        
        logger.info(f"Initializing Router with dim={dim}, num_experts={num_experts}, top_k={top_k}, "
                   f"capacity_factor={capacity_factor}, add_noise={add_noise}, "
                   f"noise_epsilon={noise_epsilon}, shared_expert={shared_expert}")
        
        # Router linear layer to produce scores for each expert
        self.router_lin = nn.Linear(dim, num_experts + 1 if shared_expert else num_experts)
        
        # For load balancing loss
        self.register_buffer("expert_counts", torch.zeros(num_experts))
    
    def forward(self, x, training=True):
        """Forward pass of the router"""
        # Get router logits
        router_logits = self.router_lin(x)
        
        # Add noise during training to prevent expert collapse
        if training and self.add_noise:
            noise = torch.randn_like(router_logits) * self.noise_epsilon
            router_logits = router_logits + noise
        
        # Get top-k experts
        if self.shared_expert:
            # Separate shared expert from regular experts
            shared_logits = router_logits[:, -1:]
            expert_logits = router_logits[:, :-1]
            
            # Get top-k experts from regular experts
            top_k_logits, top_k_indices = torch.topk(expert_logits, k=self.top_k, dim=-1)

            # Create a 2D tensor for shared expert indices with the same shape as shared_logits
            shared_indices = torch.full_like(shared_logits, self.num_experts)
            
            # Combine with shared expert (always included)
            top_k_logits = torch.cat([top_k_logits, shared_logits], dim=-1)
            top_k_indices = torch.cat([
                top_k_indices, 
                shared_indices.long()
            ], dim=-1)
        else:
            top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        
        # Convert logits to weights using softmax
        router_weights = F.softmax(top_k_logits, dim=-1)
        
        # Calculate capacity (max tokens per expert)
        seq_len = x.size(0)
        capacity = min(seq_len, int(self.capacity_factor * seq_len / self.num_experts * self.top_k))
        
        # Create a mask to indicate which tokens are routed to which experts
        expert_mask = torch.zeros(seq_len, self.num_experts + (1 if self.shared_expert else 0), 
                                 device=x.device)
        
        # For each token, mark the selected experts
        for i in range(seq_len):
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j]
                expert_mask[i, expert_idx] = 1
        
        # If using shared expert, make sure it's always active
        if self.shared_expert:
            expert_mask[:, self.num_experts] = 1
        
        return {
            "router_logits": router_logits,
            "top_k_indices": top_k_indices,
            "router_weights": router_weights,
            "expert_mask": expert_mask,
            "capacity": capacity
        }


class MoELayer(nn.Module):
    """Mixture of Experts layer as described in the knowledge base"""
    
    def __init__(self, dim, num_experts, top_k, hidden_dim=None, 
                 capacity_factor=1.0, add_noise=True, noise_epsilon=0.1,
                 shared_expert=False, dropout=0.1):
        """
        Args:
            dim: Input dimension
            num_experts: Total number of experts
            top_k: Number of experts to select per token
            hidden_dim: Hidden dimension for experts (default: 4 * dim)
            capacity_factor: Factor to determine expert capacity
            add_noise: Whether to add noise to router scores
            noise_epsilon: Noise magnitude
            shared_expert: Whether to include a shared expert
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert = shared_expert
        
        logger.info(f"Initializing MoELayer with dim={dim}, num_experts={num_experts}, top_k={top_k}, "
                   f"capacity_factor={capacity_factor}, add_noise={add_noise}, "
                   f"noise_epsilon={noise_epsilon}, shared_expert={shared_expert}")
        
        # Set hidden dimension for experts (typically larger than input dim)
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        # Create router
        self.router = Router(
            dim, 
            num_experts, 
            top_k, 
            capacity_factor,
            add_noise,
            noise_epsilon,
            shared_expert
        )
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim, dropout) for _ in range(num_experts)
        ])
        
        # Shared expert (if enabled)
        self.shared_expert_net = Expert(dim, hidden_dim, dropout) if shared_expert else None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, training=True):
        """Forward pass through the MoE layer"""
        batch_size, seq_len, dim = x.shape
        x = x.view(-1, dim)  # [batch_size * seq_len, dim]
        
        # Get routing information
        router_info = self.router(x, training)
        router_logits = router_info["router_logits"]
        top_k_indices = router_info["top_k_indices"]  # [batch_size * seq_len, top_k]
        router_weights = router_info["router_weights"]  # [batch_size * seq_len, top_k]
        expert_mask = router_info["expert_mask"]  # [batch_size * seq_len, num_experts + shared]
        
        # Initialize output tensor
        final_output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            token_mask = expert_mask[:, expert_idx].bool()
            if token_mask.sum() == 0:
                continue  # Skip if no tokens routed to this expert
            
            # Get the tokens for this expert
            expert_input = x[token_mask]
            
            # Process through the expert network
            expert_output = self.experts[expert_idx](expert_input)
            
            # Apply weights for tokens routed to this expert
            # For each token, find its weight for this expert
            weights = []
            for i, is_routed in enumerate(token_mask):
                if is_routed:
                    # Find the position of this expert in top_k_indices
                    pos = (top_k_indices[i] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        weights.append(router_weights[i, pos[0]])
            
            weights = torch.stack(weights)
            weighted_output = expert_output * weights.unsqueeze(-1)
            
            # Add to final output
            final_output[token_mask] += weighted_output
        
        # Process shared expert if enabled
        if self.shared_expert and self.shared_expert_net is not None:
            # The last position in mask corresponds to shared expert
            token_mask = expert_mask[:, -1].bool()
            if token_mask.sum() > 0:
                shared_input = x[token_mask]
                shared_output = self.shared_expert_net(shared_input)
                
                # Get weights for shared expert (always the last in router_weights)
                weights = router_weights[token_mask, -1]
                weighted_output = shared_output * weights.unsqueeze(-1)
                
                final_output[token_mask] += weighted_output
        
        # Reshape back to original dimensions
        final_output = final_output.view(batch_size, seq_len, dim)
        final_output = self.dropout(final_output)
        
        # For training, return router logits for load balancing loss
        return final_output, router_logits


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking as mentioned in the knowledge base"""
    
    def __init__(self, dim, num_heads, dropout=0.1, use_rope=True, max_seq_len=512):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        logger.info(f"Initializing MultiHeadAttention with dim={dim}, num_heads={num_heads}, "
                   f"head_dim={self.head_dim}, use_rope={use_rope}, max_seq_len={max_seq_len}")
        
        # Linear layers for query, key, value
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Rotary positional embeddings
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def apply_causal_mask(self, attn_scores):
        """Apply causal mask to attention scores"""
        seq_len = attn_scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(attn_scores.device)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        return attn_scores
    
    def forward(self, x, seq_len):
        """Forward pass with causal masking"""
        batch_size, seq_len, _ = x.shape
        
        # Project to query, key, value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embeddings if enabled
        if self.use_rope:
            q, k = self.rope(q, k, seq_len)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        attn_scores = self.apply_causal_mask(attn_scores)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block combining attention and MoE as in LLaMA 4"""
    
    def __init__(self, dim, num_heads, num_experts, top_k, 
                 mlp_ratio=4.0, dropout=0.1, use_rope=True, 
                 max_seq_len=512, shared_expert=False):
        super().__init__()
        
        logger.info(f"Initializing TransformerBlock with dim={dim}, num_heads={num_heads}, "
                   f"num_experts={num_experts}, top_k={top_k}, mlp_ratio={mlp_ratio}, "
                   f"dropout={dropout}, use_rope={use_rope}, max_seq_len={max_seq_len}, "
                   f"shared_expert={shared_expert}")
        
        # Attention part
        self.attention = MultiHeadAttention(
            dim, num_heads, dropout, use_rope, max_seq_len
        )
        self.attn_norm = RMSNorm(dim)
        
        # MoE part
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
        """Forward pass through the transformer block"""
        batch_size, seq_len, dim = x.shape
        
        # Attention block with residual connection
        attn_output = self.attention(x, seq_len)
        x = x + self.dropout(attn_output)
        x = self.attn_norm(x)
        
        # MoE block with residual connection
        moe_output, router_logits = self.moe(x, training)
        x = x + self.dropout(moe_output)
        x = self.moe_norm(x)
        
        return x, router_logits


class LLaMA4MoE(nn.Module):
    """Complete LLaMA 4 MoE model as described in the knowledge base"""
    
    def __init__(self, vocab_size, dim, num_layers, num_heads, 
                 num_experts, top_k, max_seq_len=512, dropout=0.1,
                 shared_expert=False, load_balancing_loss_coef=0.01):
        """
        Args:
            vocab_size: Size of vocabulary
            dim: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_experts: Number of experts in MoE layer
            top_k: Number of experts to select per token
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            shared_expert: Whether to include a shared expert
            load_balancing_loss_coef: Coefficient for load balancing loss
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.shared_expert = shared_expert
        self.num_experts = num_experts
        
        logger.info(f"Initializing LLaMA4MoE with vocab_size={vocab_size}, dim={dim}, "
                   f"num_layers={num_layers}, num_heads={num_heads}, num_experts={num_experts}, "
                   f"top_k={top_k}, max_seq_len={max_seq_len}, dropout={dropout}, "
                   f"shared_expert={shared_expert}, load_balancing_loss_coef={load_balancing_loss_coef}")
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, num_heads, num_experts, top_k, 
                max_seq_len=max_seq_len, shared_expert=shared_expert
            ) for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights between token embeddings and LM head
        self.token_embeddings.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None, training=True):
        """
        Forward pass through the model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Optional labels for computing loss
            training: Whether in training mode (affects MoE routing)
            
        Returns:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            loss: Optional loss value if labels are provided
            load_balancing_loss: Load balancing loss for MoE training
        """
        batch_size, seq_len = input_ids.shape
        logger.debug(f"Forward pass with input shape: {input_ids.shape}")
        
        # Check sequence length
        if seq_len > self.max_seq_len:
            error_msg = f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Initialize load balancing loss
        total_load_balancing_loss = 0
        
        # Pass through transformer layers
        for layer in self.layers:
            x, router_logits = layer(x, training)
            
            # Compute load balancing loss for this layer
            if training:
                # Get router weights and top-k indices
                router_weights = F.softmax(router_logits, dim=-1)
                top_k_weights, top_k_indices = torch.topk(router_weights, k=self.layers[0].moe.top_k, dim=-1)
                
                # Compute fraction of tokens routed to each expert
                mask = torch.zeros_like(router_weights).scatter_(1, top_k_indices, 1)
                if self.shared_expert:
                    # Exclude shared expert from load balancing
                    mask = mask[:, :-1]
                
                tokens_per_expert = mask.float().mean(dim=0)
                router_prob = router_weights.mean(dim=0)
                if self.shared_expert:
                    router_prob = router_prob[:-1]
                
                # Compute load balancing loss
                load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob)
                total_load_balancing_loss += load_balancing_loss
        
        # Final normalization
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size), 
                shift_labels.view(-1)
            )
            
            # Add load balancing loss
            if training:
                loss += self.load_balancing_loss_coef * total_load_balancing_loss
                logger.debug(f"Loss: {loss.item():.4f}, Load balancing loss: {total_load_balancing_loss.item():.4f}")
        
        return {
            "logits": logits,
            "loss": loss,
            "load_balancing_loss": total_load_balancing_loss if training else None
        }
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using the model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: Keep only top k tokens (set to None to disable)
            top_p: Keep tokens with cumulative probability >= p (set to None to disable)
            
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        logger.info(f"Starting text generation with max_new_tokens={max_new_tokens}, "
                   f"temperature={temperature}, top_k={top_k}, top_p={top_p}")
        self.eval()
        
        # Start with the input
        generated = input_ids.clone()
        
        # Generate new tokens
        for i in range(max_new_tokens):
            # Get the last token in the sequence
            idx_cond = generated[:, -self.max_seq_len:]
            
            # Get predictions
            outputs = self(idx_cond)
            logits = outputs["logits"]
            
            # Focus on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))  # Safety check
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1]
                logits = torch.where(logits < min_val, torch.full_like(logits, float('-inf')), logits)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift right to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create mask for indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                
                # Set tokens to remove to negative infinity
                logits[indices_to_remove] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Log progress
            if i % 10 == 0 or i == max_new_tokens - 1:
                logger.debug(f"Generated {i+1}/{max_new_tokens} tokens")
        
        logger.info("Text generation completed")
        return generated


class TextDataset(Dataset):
    """Dataset for training the language model"""
    
    def __init__(self, texts, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        logger.info(f"Initializing TextDataset with {len(texts)} texts, seq_len={seq_len}")
        
        # Tokenize all texts
        self.token_ids = []
        for text in texts:
            # Add EOS token at the end
            tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
            self.token_ids.extend(tokens)
        
        # Split into sequences
        self.sequences = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len):
            seq = self.token_ids[i:i + seq_len]
            self.sequences.append(seq)
        
        logger.info(f"Created {len(self.sequences)} sequences from tokenized text")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


class LLaMA4Trainer:
    """Trainer for the LLaMA 4 MoE model"""
    
    def __init__(self, model, tokenizer, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        logger.info(f"Initialized trainer with device: {device}")
        
        # Training history
        self.history = {
            'loss': [],
            'perplexity': [],
            'load_balancing_loss': []
        }
    
    def train(self, train_dataset, eval_dataset=None, batch_size=16, 
              epochs=3, eval_steps=100, save_steps=None, output_dir='./model'):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            batch_size: Batch size
            epochs: Number of epochs
            eval_steps: Evaluate every eval_steps steps
            save_steps: Save model every save_steps steps
            output_dir: Directory to save model
        """
        logger.info("Starting training process")
        logger.info(f"Training configuration: batch_size={batch_size}, epochs={epochs}, "
                   f"eval_steps={eval_steps}, save_steps={save_steps}, output_dir={output_dir}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        
        # Training loop
        global_step = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            epoch_load_balancing_losses = []
            
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch+1}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, labels=input_ids, training=True)
                loss = outputs["loss"]
                load_balancing_loss = outputs["load_balancing_loss"]
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update parameters
                self.optimizer.step()
                
                # Record losses
                epoch_losses.append(loss.item())
                epoch_load_balancing_losses.append(load_balancing_loss.item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lb_loss': f"{load_balancing_loss.item():.4f}"
                })
                
                global_step += 1
                
                # Evaluate periodically
                if eval_dataset and global_step % eval_steps == 0:
                    eval_loss, eval_ppl = self.evaluate(eval_dataset, batch_size)
                    logger.info(f"Step {global_step} | Eval Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")
                    
                    # Save history
                    self.history['loss'].append(loss.item())
                    self.history['perplexity'].append(eval_ppl)
                    self.history['load_balancing_loss'].append(load_balancing_loss.item())
                
                # Save model periodically
                if save_steps and global_step % save_steps == 0:
                    self.save_model(output_dir)
            
            # Save model at the end of each epoch
            self.save_model(output_dir)
            
            # Compute average epoch loss
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_lb_loss = sum(epoch_load_balancing_losses) / len(epoch_load_balancing_losses)
            logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f} | Avg LB Loss: {avg_lb_loss:.4f}")
    
    def evaluate(self, eval_dataset, batch_size=16):
        """Evaluate the model on a dataset"""
        logger.info("Starting evaluation")
        self.model.eval()
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, labels=input_ids, training=False)
                loss = outputs["loss"]
                
                # Count tokens (excluding padding)
                num_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        logger.info(f"Evaluation completed | Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss, perplexity
    
    def save_model(self, output_dir):
        """Save the model and tokenizer"""
        logger.info(f"Saving model to {output_dir}")
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), f"{output_dir}/model.pt")
        
        # Save tokenizer
        self.tokenizer.save_vocab(f"{output_dir}/vocab.txt")
        
        logger.info(f"Model saved to {output_dir}")
    
    def plot_training_history(self):
        """Plot training history"""
        logger.info("Plotting training history")
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        
        # Perplexity plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history['perplexity'])
        plt.title('Evaluation Perplexity')
        plt.xlabel('Steps')
        plt.ylabel('Perplexity')
        
        # Load balancing loss plot
        plt.subplot(1, 3, 3)
        plt.plot(self.history['load_balancing_loss'])
        plt.title('Load Balancing Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        logger.info("Training history plot saved as 'training_history.png'")


def download_tiny_shakespeare(data_dir="data"):
    """
    Download the tiny shakespeare dataset directly from the source
    Returns the text content
    """
    logger.info("Downloading tiny shakespeare dataset")
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / "tiny_shakespeare.txt"
    
    if file_path.exists():
        logger.info(f"Using cached tiny shakespeare dataset from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        text = response.text
        # Save for future use
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        logger.info(f"Downloaded tiny shakespeare dataset ({len(text)} characters) and saved to {file_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to download tiny shakespeare dataset: {str(e)}")
        raise


def prepare_dataset(dataset_name="tiny_shakespeare", tokenizer=None, seq_len=128, max_samples=None, data_dir="data"):
    """
    Prepare dataset for training
    
    Args:
        dataset_name: Name of the dataset from Hugging Face
        tokenizer: Tokenizer to use (if None, will create a new one)
        seq_len: Sequence length
        max_samples: Maximum number of samples to use
        data_dir: Directory to store downloaded datasets
        
    Returns:
        train_dataset, eval_dataset, tokenizer
    """
    logger.info(f"Preparing dataset: {dataset_name}")
    
    # Load dataset
    if dataset_name == "tiny_shakespeare":
        # Tiny Shakespeare dataset (download directly)
        text = download_tiny_shakespeare(data_dir)
        
        # Split into train and validation
        train_size = int(0.9 * len(text))
        train_text = text[:train_size]
        eval_text = text[train_size:]
        
        logger.info(f"Dataset split: train={len(train_text)} chars, eval={len(eval_text)} chars")
        
        # Create tokenizer if not provided
        if tokenizer is None:
            tokenizer = CharacterTokenizer(train_text)
        
        # Create datasets
        train_dataset = TextDataset([train_text], tokenizer, seq_len=seq_len)
        eval_dataset = TextDataset([eval_text], tokenizer, seq_len=seq_len)
        
        return train_dataset, eval_dataset, tokenizer
    
    else:
        # Load from Hugging Face datasets
        logger.info(f"Loading dataset '{dataset_name}' from Hugging Face")
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name)
            
            # For demonstration, use a small portion of the dataset
            if max_samples:
                dataset = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            else:
                dataset = dataset["train"]
            
            # Extract text (assuming 'text' column)
            texts = dataset["text"]
            
            logger.info(f"Loaded {len(texts)} examples from dataset")
            
            # Create tokenizer if not provided
            if tokenizer is None:
                all_text = " ".join(texts[:1000])  # Use a sample to build vocab
                tokenizer = CharacterTokenizer(all_text)
            
            # Split into train and validation
            train_size = int(0.9 * len(texts))
            train_texts = texts[:train_size]
            eval_texts = texts[train_size:]
            
            logger.info(f"Dataset split: train={len(train_texts)} examples, eval={len(eval_texts)} examples")
            
            # Create datasets
            train_dataset = TextDataset(train_texts, tokenizer, seq_len=seq_len)
            eval_dataset = TextDataset(eval_texts, tokenizer, seq_len=seq_len)
            
            return train_dataset, eval_dataset, tokenizer
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {str(e)}")
            raise


def main():
    """Main function to train and evaluate the model"""
    # Configuration
    config = {
        "vocab_size": 100,  # Will be updated after tokenizer is created
        "dim": 256,         # Model dimension
        "num_layers": 4,    # Number of transformer layers
        "num_heads": 8,     # Number of attention heads
        "num_experts": 8,   # Number of experts in MoE layer
        "top_k": 2,         # Number of experts to select per token
        "max_seq_len": 128, # Maximum sequence length
        "dropout": 0.1,     # Dropout rate
        "shared_expert": True,  # Include a shared expert
        "load_balancing_loss_coef": 0.01,  # Coefficient for load balancing loss
        "batch_size": 16,
        "learning_rate": 3e-4,
        "epochs": 3,
        "eval_steps": 100,
        "dataset": "tiny_shakespeare",  # Can be changed to other datasets
        "seq_len": 128,
        "data_dir": "data"  # Directory to store datasets
    }
    
    logger.info("Starting main function")
    logger.info(f"Configuration: {config}")
    
    try:
        logger.info("Preparing dataset...")
        train_dataset, eval_dataset, tokenizer = prepare_dataset(
            dataset_name=config["dataset"],
            seq_len=config["seq_len"],
            data_dir=config["data_dir"]
        )
        
        # Update vocab size
        config["vocab_size"] = len(tokenizer)
        logger.info(f"Vocabulary size: {config['vocab_size']}")
        
        # Create model
        logger.info("Creating model...")
        model = LLaMA4MoE(
            vocab_size=config["vocab_size"],
            dim=config["dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
            shared_expert=config["shared_expert"],
            load_balancing_loss_coef=config["load_balancing_loss_coef"]
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=0.01
        )
        
        # Create trainer
        trainer = LLaMA4Trainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            eval_steps=config["eval_steps"]
        )
        
        # Generate some text
        logger.info("\nGenerating sample text...")
        prompt = "To be or not to be"
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
        
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated_text[len(prompt):]}")
        
        # Plot training history
        trainer.plot_training_history()
        
        # Save final model
        trainer.save_model("./llama4_moe_model")
        
        logger.info("Training completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")
        raise


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run main function
    main()