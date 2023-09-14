import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size)).to(x.device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class PositionalEmbedding(nn.Module):
    """Positional embeddings"""

    def __init__(self, size: int, emb_type: str, **kwargs):
        super().__init__()

        if emb_type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif emb_type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif emb_type == "learnable":
            self.layer = LearnableEmbedding(size)
        else:
            raise ValueError(f"Unknown positional embedding type: {emb_type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)
