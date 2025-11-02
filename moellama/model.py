"""
Complete LLaMA4-style Mixture of Experts language model.

This module contains the full model architecture that combines all components:
token embeddings, transformer blocks with MoE, and the language modeling head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from moellama.layers import TransformerBlock, RMSNorm



class LLaMA4MoE(nn.Module):
    """
    Complete LLaMA 4 MoE language model.

    Architecture:
        Input tokens
        ↓
        Token Embeddings
        ↓
        Transformer Blocks (N layers)
        │ - Multi-head Attention with RoPE
        │ - Mixture of Experts
        ↓
        Final RMSNorm
        ↓
        Language Modeling Head
        ↓
        Output logits

    Key features:
    - Weight tying between token embeddings and LM head
    - Load balancing loss to encourage even expert usage
    - Autoregressive generation with temperature, top-k, top-p sampling

    Args:
        vocab_size: Size of the vocabulary
        dim: Model dimension (embedding size)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        num_experts: Number of experts in each MoE layer
        top_k: Number of experts to activate per token
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        shared_expert: Whether to use a shared expert in MoE
        load_balancing_loss_coef: Coefficient for load balancing loss
    """

    def __init__(
        self,
        vocab_size,
        dim,
        num_layers,
        num_heads,
        num_experts,
        top_k,
        max_seq_len=512,
        dropout=0.1,
        shared_expert=False,
        load_balancing_loss_coef=0.01
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.shared_expert = shared_expert
        self.num_experts = num_experts

        logger.info(
            f"Initializing LLaMA4MoE: vocab_size={vocab_size}, dim={dim}, "
            f"num_layers={num_layers}, num_heads={num_heads}, num_experts={num_experts}, "
            f"top_k={top_k}, max_seq_len={max_seq_len}, dropout={dropout}, "
            f"shared_expert={shared_expert}, load_balancing_loss_coef={load_balancing_loss_coef}"
        )

        # Token embeddings: map token IDs to dense vectors
        self.token_embeddings = nn.Embedding(vocab_size, dim)

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads,
                num_experts,
                top_k,
                max_seq_len=max_seq_len,
                shared_expert=shared_expert,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Final normalization before output
        self.norm = RMSNorm(dim)

        # Language modeling head: map from model dim to vocabulary
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying: share weights between embeddings and LM head
        # This reduces parameters and often improves performance
        self.token_embeddings.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights.

        Uses normal distribution with small std for stable training:
        - Linear layers: N(0, 0.02)
        - Embeddings: N(0, 0.02)

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, training=True):
        """
        Forward pass through the model.

        Process:
        1. Embed input tokens
        2. Pass through transformer layers
        3. Apply final normalization
        4. Project to vocabulary (get logits)
        5. Optionally compute loss if labels provided

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target token IDs for loss computation [batch, seq_len]
            training: Whether in training mode (affects MoE routing and loss)

        Returns:
            Dictionary containing:
                - logits: Predicted logits [batch, seq_len, vocab_size]
                - loss: Cross-entropy + load balancing loss (if labels provided)
                - load_balancing_loss: Load balancing loss component
        """
        batch_size, seq_len = input_ids.shape
        logger.debug(f"Forward pass with input shape: {input_ids.shape}")

        # Check sequence length
        if seq_len > self.max_seq_len:
            error_msg = f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get token embeddings
        # Shape: [batch, seq_len, dim]
        x = self.token_embeddings(input_ids)

        # Initialize load balancing loss accumulator
        total_load_balancing_loss = 0

        # Pass through all transformer layers
        for layer in self.layers:
            x, router_logits = layer(x, training)

            # Compute load balancing loss for this layer
            if training:
                # Load balancing encourages even distribution of tokens across experts
                # Formula: num_experts * sum(fraction_tokens_per_expert * mean_router_prob_per_expert)

                # Get router weights (probabilities)
                router_weights = F.softmax(router_logits, dim=-1)

                # Get top-k selection
                top_k = self.layers[0].moe.top_k
                top_k_weights, top_k_indices = torch.topk(router_weights, k=top_k, dim=-1)

                # Create mask for selected experts
                mask = torch.zeros_like(router_weights).scatter_(1, top_k_indices, 1)

                # Exclude shared expert from load balancing if present
                if self.shared_expert:
                    mask = mask[:, :-1]

                # Compute fraction of tokens routed to each expert
                tokens_per_expert = mask.float().mean(dim=0)

                # Compute mean router probability for each expert
                router_prob = router_weights.mean(dim=0)
                if self.shared_expert:
                    router_prob = router_prob[:-1]

                # Load balancing loss: encourages uniform distribution
                load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob)
                total_load_balancing_loss += load_balancing_loss

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary to get logits
        # Shape: [batch, seq_len, vocab_size]
        logits = self.lm_head(x)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # For next-token prediction, shift logits and labels
            # Predict token t+1 from tokens up to t
            shift_logits = logits[..., :-1, :].contiguous()  # Remove last logit
            shift_labels = labels[..., 1:].contiguous()  # Remove first label

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )

            # Add load balancing loss
            if training:
                loss += self.load_balancing_loss_coef * total_load_balancing_loss
                logger.debug(
                    f"Loss: {loss.item():.4f}, "
                    f"Load balancing loss: {total_load_balancing_loss.item():.4f}"
                )

        return {
            "logits": logits,
            "loss": loss,
            "load_balancing_loss": total_load_balancing_loss if training else None
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text autoregressively using the model.

        Sampling strategies:
        - Temperature: Controls randomness (higher = more random)
        - Top-k: Only sample from top k tokens
        - Top-p (nucleus): Sample from smallest set with cumulative prob >= p

        Args:
            input_ids: Starting token IDs [batch, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0 = greedy, >1 = more random)
            top_k: Keep only top k tokens (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        logger.info(
            f"Generating: max_new_tokens={max_new_tokens}, "
            f"temperature={temperature}, top_k={top_k}, top_p={top_p}"
        )
        self.eval()

        # Start with the input
        generated = input_ids.clone()

        # Generate tokens one at a time
        for i in range(max_new_tokens):
            # Crop to max sequence length (keep most recent tokens)
            idx_cond = generated[:, -self.max_seq_len:]

            # Get predictions
            outputs = self(idx_cond, training=False)
            logits = outputs["logits"]

            # Focus on the last time step (next token prediction)
            # Shape: [batch, vocab_size]
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))  # Safety check
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1]
                # Set logits below top-k threshold to -inf
                logits = torch.where(
                    logits < min_val,
                    torch.full_like(logits, float('-inf')),
                    logits
                )

            # Apply top-p (nucleus) filtering if specified
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift right to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original order
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )

                # Set removed tokens to -inf
                logits[indices_to_remove] = float('-inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Log progress periodically
            if i % 10 == 0 or i == max_new_tokens - 1:
                logger.debug(f"Generated {i+1}/{max_new_tokens} tokens")

        logger.info("Generation completed")
        return generated
