import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BITNET 1.58-BIT CORE ---
class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-8)
        w_quant = torch.sign(w) * (w.abs() > (0.5 * scale)).float()
        return F.linear(x, w + (w_quant - w).detach(), self.bias)

class MonaCoreV27(nn.Module):
    def __init__(self, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 512)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(512, 8, batch_first=True) for _ in range(8)])
        self.actor = BitLinear(512, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks: h = block(h)
        return self.actor(h[:, -1, :])