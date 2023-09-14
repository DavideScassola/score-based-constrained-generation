import torch
from torch import nn

from src.models.score_based.nn.positional_embedding import PositionalEmbedding


class Block(nn.Module):
    """Residual block for the denoising model."""

    def __init__(self, size: int):
        super().__init__()

        self.layer = nn.Linear(size, size)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.activation(self.layer(x))


class MLP(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        """Denoising NN with positional embeddings.

        Args:
            hidden_size: Hidden size of blocks.
            hidden_layers: Number of hidden layers.
            emb_size: Size of positional embeddings.
            time_emb: Type of temporal embedding.
            input_emb: Type of input embedding.

        """
        super().__init__()
        self.n_features = n_features
        self.time_mlp = PositionalEmbedding(emb_size, emb_type=time_emb)
        self.input_mlps = [
            PositionalEmbedding(emb_size, emb_type=input_emb, scale=25.0)
            for _ in range(self.n_features)
        ]

        concat_size = len(self.time_mlp.layer)
        for mlp_layer in self.input_mlps:
            concat_size += len(mlp_layer.layer)
        layers = [
            nn.Linear(concat_size, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
        ]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Linear(hidden_size, self.n_features))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.squeeze())
        x = torch.cat(
            [mlp(x[:, idx]) for idx, mlp in enumerate(self.input_mlps)] + [t_emb],
            dim=-1,
        )
        x = self.sequential(x)
        return x
