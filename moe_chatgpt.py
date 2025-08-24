import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
import math

# =========================
# MoE Layer
# =========================
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        ) for _ in range(n_experts)])
        
        # Gating network
        self.gate = nn.Linear(input_dim, n_experts)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        gate_scores = self.gate(x)  # [batch_size, seq_len, n_experts]
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        topk_probs = F.softmax(topk_scores, dim=-1)  # normalize over top-k
        
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i]  # [batch_size, seq_len]
            prob = topk_probs[..., i].unsqueeze(-1)  # [batch_size, seq_len, 1]
            expert_out = torch.zeros_like(x)
            for e in range(self.n_experts):
                mask = (expert_idx == e).float().unsqueeze(-1)  # mask where expert e is selected
                if mask.sum() > 0:
                    expert_out += mask * self.experts[e](x)
            output += prob * expert_out
        return output

# =========================
# MoE-based LLM (small transformer block example)
# =========================
class MoELLMMini(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, ff_dim=256, n_layers=2, n_experts=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(embed_dim, n_heads, batch_first=True),
                'moe': MoELayer(embed_dim, ff_dim, n_experts),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim)
            }))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            # MoE feedforward
            x = layer['norm2'](x + layer['moe'](x))
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# =========================
# Dataset wrapper
# =========================
class TextDataset(Dataset):
    def __init__(self, tokenizer, dataset_name="tiny_shakespeare", split="train", block_size=128):
        raw_dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Prepare text
        self.examples = []
        for item in raw_dataset:
            text = item['text'] if 'text' in item else item['content']
            tokenized = tokenizer(text, truncation=True, max_length=block_size, padding='max_length', return_tensors="pt")
            self.examples.append(tokenized['input_ids'].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        y = x.clone()
        return x, y  # LM task: predict next token

# =========================
# Training class
# =========================
class Trainer:
    def __init__(self, model, dataset, lr=1e-3, batch_size=16, epochs=5, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in self.dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                # shift logits for LM prediction
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {total_loss/len(self.dataloader):.4f}")

# =========================
# Inference class
# =========================
class Inference:
    def __init__(self, model, tokenizer, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_length=50):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated = input_ids
        for _ in range(max_length):
            logits = self.model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # required for padding

    dataset = TextDataset(tokenizer, dataset_name="tiny_shakespeare", split="train", block_size=64)
    
    model = MoELLMMini(vocab_size=len(tokenizer), embed_dim=128, ff_dim=256, n_layers=2, n_experts=4)
    
    trainer = Trainer(model, dataset, lr=1e-3, batch_size=8, epochs=3)
    trainer.train()
    
    infer = Inference(model, tokenizer)
    prompt = "To be, or not to be"
    generated_text = infer.generate(prompt, max_length=50)
    print("Generated:", generated_text)
