import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from datasets import load_dataset
import tiktoken

# Configuration class
class ModelConfig:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2 tokenizer size
        self.d_model = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_ff = 2048
        self.max_seq_len = 1024
        self.dropout = 0.1
        self.num_experts = 8
        self.top_k = 2  # Number of experts to activate per token
        self.expert_capacity = 64  # Max tokens per expert
        self.aux_loss_coef = 0.01  # Load balancing loss coefficient

# Rotary Positional Embedding (RoPE)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
    def forward(self, x, seq_len=None):
        # x: [batch_size, seq_len, n_heads, head_dim]
        batch_size, seq_len, n_heads, head_dim = x.shape
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.repeat(batch_size, n_heads, 1, 1)
        emb = emb.reshape(batch_size, seq_len, n_heads, head_dim)
        
        # Apply rotary embeddings
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        x_rot = x * cos_emb + self.rotate_half(x) * sin_emb
        return x_rot
    
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

# RMSNorm implementation
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale

# Multi-head self-attention with RoPE
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply rotary positional embeddings
        q = self.rope(q)
        k = self.rope(k)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.o_proj(output)
        return output

# Expert network (small MLP)
class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x):
        return self.ffn(x)

# Router for MoE
class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        
    def forward(self, x):
        # Get router logits
        logits = self.gate(x)
        
        # Add noise during training for load balancing
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
            
        # Get top-k experts
        top_k_logits, top_k_indices = logits.topk(self.config.top_k, dim=-1)
        
        # Create mask for top-k experts
        top_k_mask = F.one_hot(top_k_indices, num_classes=self.config.num_experts).float()
        
        # Apply softmax to top-k logits
        router_probs = F.softmax(top_k_logits, dim=-1)
        
        return router_probs, top_k_indices, top_k_mask

# Mixture of Experts layer
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.router = Router(config)
        self.shared_expert = Expert(config)  # Shared expert as mentioned in the document
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        
        # Get router decisions
        router_probs, expert_indices, expert_mask = self.router(x_flat)
        
        # Initialize output
        expert_output = torch.zeros_like(x_flat)
        
        # Process through each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            expert_tokens = (expert_indices == expert_idx).any(dim=-1)
            
            if expert_tokens.any():
                # Process tokens through expert
                expert_input = x_flat[expert_tokens]
                expert_out = expert(expert_input)
                
                # Get router weights for this expert
                expert_weights = router_probs[expert_tokens]
                expert_weights = expert_weights[expert_indices[expert_tokens] == expert_idx]
                
                # Apply weights to expert output
                weighted_output = expert_out * expert_weights.unsqueeze(-1)
                
                # Add to final output
                expert_output[expert_tokens] += weighted_output
        
        # Process all tokens through shared expert
        shared_output = self.shared_expert(x_flat)
        
        # Combine expert outputs with shared expert output
        final_output = expert_output + shared_output
        
        # Reshape back to original shape
        final_output = final_output.reshape(batch_size, seq_len, d_model)
        
        return final_output, expert_mask

# Transformer decoder block with MoE
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = MoELayer(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attn(self.attn_norm(x), mask)
        x = x + self.dropout(attn_out)
        
        # MoE feedforward with residual connection
        moe_out, expert_mask = self.moe(self.moe_norm(x))
        x = x + self.dropout(moe_out)
        
        return x, expert_mask

# Complete LLaMA 4 model with MoE
class LLaMA4MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.blocks = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_layers)
        ])
        
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        # Create positional indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token and positional embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        # Forward through decoder blocks
        expert_masks = []
        for block in self.blocks:
            x, expert_mask = block(x, mask)
            expert_masks.append(expert_mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Add load balancing auxiliary loss
            aux_loss = self._load_balancing_loss(expert_masks)
            loss = loss + self.config.aux_loss_coef * aux_loss
            
        return logits, loss
    
    def _load_balancing_loss(self, expert_masks):
        """
        Calculate load balancing loss to ensure experts are used evenly
        """
        loss = 0
        for expert_mask in expert_masks:
            # Expert mask shape: [batch*seq_len, num_experts]
            # Calculate fraction of tokens routed to each expert
            expert_usage = expert_mask.float().mean(0)
            
            # Calculate probability of routing to each expert
            router_prob = expert_mask.float().mean(0)
            
            # Calculate load balancing loss
            loss += (expert_usage * router_prob).sum()
            
        return loss / len(expert_masks)
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the model
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# Dataset class for text data
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # Create input and target
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

# Training function
def train_model(model, train_loader, val_loader, optimizer, device, epochs=10):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}')
        
        # Validation
        val_loss = evaluate_model(model, val_loader, device)
        print(f'Validation Loss: {val_loss:.4f}')
    
    return model

# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    model.train()
    return avg_loss

# Main function to run training
def main():
    # Load configuration
    config = ModelConfig()
    
    # Initialize tokenizer (using GPT-2 tokenizer for simplicity)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]
    
    # Filter out empty texts
    train_texts = [text for text in train_texts if len(text.strip()) > 0]
    val_texts = [text for text in val_texts if len(text.strip()) > 0]
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLaMA4MoE(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train model
    model = train_model(model, train_loader, val_loader, optimizer, device, epochs=5)
    
    # Save model
    torch.save(model.state_dict(), "llama4_moe_model.pth")
    
    # Example generation
    prompt = "The future of artificial intelligence is"
    prompt_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    
    generated = model.generate(input_tensor, max_new_tokens=50, temperature=0.8)
    generated_text = tokenizer.decode(generated[0].cpu().numpy())
    
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()