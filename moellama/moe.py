"""
Mixture of Experts (MoE) implementation.

This module implements the core MoE components:
- Expert: Individual feed-forward networks that specialize
- Router: Selects which experts process each token
- MoELayer: Combines router and experts with load balancing

The MoE architecture allows the model to scale to more parameters without
proportionally increasing computation, as only a subset of experts are
activated for each token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger



class Expert(nn.Module):
    """
    Individual expert network - a simple feed-forward network.

    Each expert is a two-layer MLP with GELU activation:
    Input -> Linear -> GELU -> Linear -> Dropout -> Output

    During training, different experts learn to specialize in different
    patterns or types of tokens.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension (typically 4x dim)
        dropout: Dropout probability for regularization
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        logger.debug(f"Initializing Expert: dim={dim}, hidden_dim={hidden_dim}, dropout={dropout}")

        # Two-layer MLP with GELU activation
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Smooth, non-linear activation
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass through the expert.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Output tensor with same shape
        """
        return self.net(x)


class Router(nn.Module):
    """
    Router network that decides which experts to use for each token.

    The router computes a score for each expert and selects the top-k
    experts with the highest scores. This sparse selection is key to
    the efficiency of MoE models.

    Load balancing:
        To prevent all tokens from routing to the same expert (expert collapse),
        the router can add noise during training and we compute a load balancing
        loss that encourages even distribution of tokens across experts.

    Args:
        dim: Input dimension
        num_experts: Total number of experts
        top_k: Number of experts to select per token
        capacity_factor: Factor to determine expert capacity (for future extensions)
        add_noise: Whether to add noise to router scores (prevents expert collapse)
        noise_epsilon: Magnitude of noise to add
        shared_expert: Whether to include a shared expert that always processes every token
    """

    def __init__(
        self,
        dim,
        num_experts,
        top_k,
        capacity_factor=1.0,
        add_noise=True,
        noise_epsilon=0.1,
        shared_expert=False
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.shared_expert = shared_expert

        logger.info(
            f"Initializing Router: dim={dim}, num_experts={num_experts}, top_k={top_k}, "
            f"capacity_factor={capacity_factor}, add_noise={add_noise}, "
            f"noise_epsilon={noise_epsilon}, shared_expert={shared_expert}"
        )

        # Linear layer to produce routing scores
        # +1 if using shared expert (extra slot in output)
        num_outputs = num_experts + 1 if shared_expert else num_experts
        self.router_lin = nn.Linear(dim, num_outputs)

        # For tracking expert usage (for load balancing loss)
        self.register_buffer("expert_counts", torch.zeros(num_experts))

    def forward(self, x, training=True):
        """
        Forward pass of the router.

        Process:
        1. Compute routing logits for each expert
        2. Add noise during training (prevents expert collapse)
        3. Select top-k experts per token
        4. Compute routing weights using softmax
        5. Create expert assignment mask

        Args:
            x: Input tensor [seq_len, dim]
            training: Whether in training mode (affects noise)

        Returns:
            Dictionary containing:
                - router_logits: Raw logits for all experts [seq_len, num_experts]
                - top_k_indices: Indices of selected experts [seq_len, top_k]
                - router_weights: Weights for selected experts [seq_len, top_k]
                - expert_mask: Binary mask for expert assignment [seq_len, num_experts]
                - capacity: Maximum tokens per expert
        """
        # Compute routing logits
        # Shape: [seq_len, num_experts] or [seq_len, num_experts+1] if shared expert
        router_logits = self.router_lin(x)

        # Add noise during training to prevent expert collapse
        # This encourages exploration of different experts
        if training and self.add_noise:
            noise = torch.randn_like(router_logits) * self.noise_epsilon
            router_logits = router_logits + noise

        # Handle shared expert separately if enabled
        if self.shared_expert:
            # Separate shared expert logits from regular expert logits
            shared_logits = router_logits[:, -1:]  # Last column
            expert_logits = router_logits[:, :-1]  # All but last

            # Select top-k from regular experts
            top_k_logits, top_k_indices = torch.topk(expert_logits, k=self.top_k, dim=-1)

            # Create shared expert indices (all pointing to the shared expert)
            # Shape: [seq_len, 1]
            shared_indices = torch.full_like(shared_logits, self.num_experts)

            # Combine: top-k experts + shared expert
            # The shared expert is always included for every token
            top_k_logits = torch.cat([top_k_logits, shared_logits], dim=-1)
            top_k_indices = torch.cat([top_k_indices, shared_indices.long()], dim=-1)
        else:
            # Standard top-k selection without shared expert
            top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)

        # Convert logits to weights using softmax
        # Weights sum to 1 across the top-k experts for each token
        router_weights = F.softmax(top_k_logits, dim=-1)

        # Calculate capacity (max tokens per expert)
        # This can be used for more advanced batching strategies
        seq_len = x.size(0)
        capacity = min(seq_len, int(self.capacity_factor * seq_len / self.num_experts * self.top_k))

        # Create binary mask indicating which tokens go to which experts (VECTORIZED)
        # Shape: [seq_len, num_experts] or [seq_len, num_experts+1] if shared expert
        num_mask_cols = self.num_experts + (1 if self.shared_expert else 0)
        expert_mask = torch.zeros(seq_len, num_mask_cols, device=x.device)

        # Vectorized mask creation using scatter_
        # For each token, mark its selected experts (no Python loops!)
        expert_mask.scatter_(1, top_k_indices, 1.0)

        # Ensure shared expert is always active if enabled
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
    """
    Complete Mixture of Experts layer.

    This layer combines the router and all expert networks. For each token:
    1. Router selects top-k experts and computes routing weights
    2. Selected experts process the token
    3. Expert outputs are combined using routing weights

    The key benefit of MoE is that we can have many experts (large capacity)
    but only activate a few per token (low computational cost).

    Example:
        With 8 experts and top_k=2, each token is processed by only 2 experts,
        but different tokens may choose different experts based on their content.

    Args:
        dim: Input/output dimension
        num_experts: Total number of experts
        top_k: Number of experts to activate per token
        hidden_dim: Hidden dimension for experts (default: 4 * dim)
        capacity_factor: Factor for expert capacity calculation
        add_noise: Whether to add noise to routing
        noise_epsilon: Noise magnitude
        shared_expert: Whether to include a shared expert
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim,
        num_experts,
        top_k,
        hidden_dim=None,
        capacity_factor=1.0,
        add_noise=True,
        noise_epsilon=0.1,
        shared_expert=False,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert = shared_expert

        logger.info(
            f"Initializing MoELayer: dim={dim}, num_experts={num_experts}, top_k={top_k}, "
            f"hidden_dim={hidden_dim}, shared_expert={shared_expert}"
        )

        # Set hidden dimension (typically 4x model dimension)
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
        """
        Forward pass through the MoE layer.

        Process:
        1. Flatten batch and sequence dimensions
        2. Route tokens to experts
        3. Process each expert's assigned tokens
        4. Combine expert outputs using routing weights
        5. Reshape back to original dimensions

        Args:
            x: Input tensor [batch, seq_len, dim]
            training: Whether in training mode

        Returns:
            Tuple of (output, router_logits):
                - output: Processed tensor [batch, seq_len, dim]
                - router_logits: Router logits for load balancing loss
        """
        batch_size, seq_len, dim = x.shape

        # Flatten batch and sequence dimensions for easier processing
        # Shape: [batch * seq_len, dim]
        x = x.view(-1, dim)

        # Get routing information
        router_info = self.router(x, training)
        router_logits = router_info["router_logits"]
        top_k_indices = router_info["top_k_indices"]  # [batch*seq_len, top_k]
        router_weights = router_info["router_weights"]  # [batch*seq_len, top_k]
        expert_mask = router_info["expert_mask"]  # [batch*seq_len, num_experts]

        # Initialize output tensor
        final_output = torch.zeros_like(x)

        # Process each expert (vectorized)
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            token_mask = expert_mask[:, expert_idx].bool()

            if token_mask.sum() == 0:
                # Skip if no tokens were routed to this expert
                continue

            # Get the tokens assigned to this expert
            expert_input = x[token_mask]

            # Process through the expert network
            expert_output = self.experts[expert_idx](expert_input)

            # Find routing weights for this expert (VECTORIZED - much faster!)
            # Create a mask for where top_k_indices equals expert_idx
            # Shape: [num_tokens_routed, top_k]
            expert_positions = (top_k_indices[token_mask] == expert_idx)

            # Get the weights using the mask
            # For each routed token, find which position in top_k has this expert
            # Shape: [num_tokens_routed]
            weights = (router_weights[token_mask] * expert_positions).sum(dim=1)

            # Apply routing weights to expert outputs
            weighted_output = expert_output * weights.unsqueeze(-1)

            # Add to final output
            final_output[token_mask] += weighted_output

        # Process shared expert if enabled
        if self.shared_expert and self.shared_expert_net is not None:
            # Shared expert processes ALL tokens
            token_mask = expert_mask[:, -1].bool()

            if token_mask.sum() > 0:
                shared_input = x[token_mask]
                shared_output = self.shared_expert_net(shared_input)

                # Get weights for shared expert (last position in router_weights)
                weights = router_weights[token_mask, -1]
                weighted_output = shared_output * weights.unsqueeze(-1)

                final_output[token_mask] += weighted_output

        # Reshape back to original dimensions
        final_output = final_output.view(batch_size, seq_len, dim)
        final_output = self.dropout(final_output)

        # Return output and router logits (for load balancing loss)
        return final_output, router_logits
