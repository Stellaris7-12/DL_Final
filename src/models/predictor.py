from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CodecAutoregressivePredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        codebook_size: int,
        hidden_dim: int = 512,
        ff_dim: int = 2048,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        token_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        self.token_embedding = nn.Embedding(codebook_size, token_embedding_dim)
        self.token_proj = nn.Linear(token_embedding_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, context_embeddings: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(context_tokens.long()).mean(dim=2)
        token_hidden = self.token_proj(token_emb)
        embedding_hidden = self.embedding_proj(context_embeddings)
        hidden = self.positional_encoding(token_hidden + embedding_hidden)
        causal_mask = torch.triu(
            torch.ones(hidden.size(1), hidden.size(1), device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.norm(hidden[:, -1, :])
        return self.output_proj(hidden)
