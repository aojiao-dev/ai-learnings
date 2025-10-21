from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np, torch
import tiktoken


# -----------------------------
# Dataset Library
# -----------------------------
class TinyStoriesDataset(Dataset):
    def __init__(self, texts, encoder, context_len=256, min_len=16):
        tokens = []
        for text in texts:
            if text:
                ids = encoder.encode(text)
                if len(ids) >= min_len:
                    eot = (
                        encoder.eot_token
                        if hasattr(encoder, "eot_token")
                        else encoder.encode("<|endoftext|>")[0]
                    )
                    tokens.extend(ids + [eot])

        self.encoder = encoder
        self.context_len = context_len
        self.tokens = np.array(tokens, dtype=np.int32)
        self.n_blocks = (len(self.tokens) - 1) // self.context_len

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.context_len
        end = start + self.context_len
        assert end + 1 <= len(self.tokens)
        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1 : end + 1], dtype=torch.long)
        return x, y


# -----------------------------
# Multi-head Attention
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_len, context_len))
        self.register_buffer("causal_mask", mask.view(1, 1, context_len, context_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # reshape to heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_drop(self.proj(y))  # (B, T, C)
        return y


# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# -----------------------------
# Mini-GPT (One Block)
# -----------------------------
class TinyGPT(nn.Module):
    def __init__(
        self, vocab_size, context_len, d_model=256, n_heads=4, n_layers=1, dropout=0.1
    ):
        super().__init__()
        self.context_len = context_len
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, context_len, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (common in GPTs)
        self.lm_head.weight = self.tok_embed.weight

        # init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context_len, "Sequence length > context_len"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.tok_embed(idx) + self.pos_embed(pos)  # (B, T, C)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len :]  # crop to context
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# -----------------------------
# Tokenizer
# -----------------------------

# Run tokenizer
encoder = tiktoken.get_encoding("gpt2")
encoder_test = "A tiny dragon danced."
encoder_test_ids = encoder.encode(encoder_test)
print(encoder_test_ids, encoder.decode(encoder_test_ids))


# -----------------------------
# Dataset Loader
# -----------------------------
def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# Load simple text for pre-training
dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")
texts = [x["text"] for x in dataset]
print("text length: ", len(texts))
print("example text: ", texts[0])

train_dataset = TinyStoriesDataset(texts, encoder)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
)

# -----------------------------
# Training (few steps)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = encoder.n_vocab
context_len = train_dataset.context_len

model = TinyGPT(
    vocab_size, context_len, d_model=256, n_heads=4, n_layers=1, dropout=0.1
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

torch.manual_seed(0)


def loss_on_batch(xb, yb):
    logits = model(xb)
    # Flatten for token-level cross-entropy
    return F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))


# quick smoke test
xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)
print("batch:", xb.shape, yb.shape)
print("init loss:", float(loss_on_batch(xb, yb).item()))

# tiny training loop (a few iterations just to see loss drop)
model.train()
for step, (xb, yb) in enumerate(train_loader):
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad(set_to_none=True)
    loss = loss_on_batch(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 50 == 0:
        torch.save(model.state_dict(), "gpt2_mini.pth")
        print(f"step {step:4d}  loss {loss.item():.4f}")
    if step == 200:  # keep it short for demo; increase later
        break

# -----------------------------
# Inference
# -----------------------------
model = TinyGPT(
    vocab_size, context_len, d_model=256, n_heads=4, n_layers=1, dropout=0.1
).to(device)
model.load_state_dict(torch.load("gpt2_mini.pth", map_location=device))
model.eval()
with torch.no_grad():
    seed_ids = encoder.encode("A little girl once said ")
    idx = torch.tensor([seed_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=50)
    print(encoder.decode(out[0].tolist()))
