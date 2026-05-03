from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class CodecFrameConfig:
    sample_rate: int
    frame_shift: float
    samples_per_frame: int
    num_codebooks: int
    codebook_size: int
    embedding_dim: int


@dataclass
class CodecBatch:
    pre_quant_embeddings: torch.Tensor
    tokens: torch.Tensor
    post_quant_embeddings: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)


class CodecAdapter(nn.Module):
    name: str

    def frame_config(self) -> CodecFrameConfig:
        raise NotImplementedError

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> CodecBatch:
        raise NotImplementedError

    def quantize_embeddings(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def decode_tokens(self, tokens: torch.Tensor, aux: dict[str, Any] | None = None) -> torch.Tensor:
        raise NotImplementedError
