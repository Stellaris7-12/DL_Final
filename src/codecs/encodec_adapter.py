from __future__ import annotations

from typing import Any

import torch

from src.codecs.base import CodecAdapter, CodecBatch, CodecFrameConfig
from src.utils.audio import resample_audio


class EncodecAdapter(CodecAdapter):
    def __init__(self, model_name: str = "24khz", target_bandwidth_kbps: float = 6.0) -> None:
        super().__init__()
        try:
            from encodec import EncodecModel
        except ImportError as exc:
            raise ImportError(
                "The `encodec` package is required for EncodecAdapter. "
                "Install dependencies from requirements.txt."
            ) from exc

        if model_name != "24khz":
            raise ValueError("Only the official 24kHz EnCodec model is currently supported.")

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(target_bandwidth_kbps)
        self.name = "encodec"
        self.target_bandwidth_kbps = target_bandwidth_kbps
        self.sample_rate = getattr(self.model, "sample_rate", 24_000)
        self.channels = getattr(self.model, "channels", 1)
        self.frame_rate = getattr(self.model, "frame_rate", 75)

    def frame_config(self) -> CodecFrameConfig:
        samples_per_frame = int(round(self.sample_rate / self.frame_rate))
        codebook_size = 1024
        num_codebooks = getattr(self.model.quantizer, "n_q", 8)
        embedding_dim = getattr(self.model.encoder, "dimension", getattr(self.model.quantizer, "dimension", 128))
        return CodecFrameConfig(
            sample_rate=self.sample_rate,
            frame_shift=samples_per_frame / self.sample_rate,
            samples_per_frame=samples_per_frame,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
        )

    def _prepare(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3:
            raise ValueError(f"Expected waveform shape [B, T] or [B, 1, T], got {tuple(waveform.shape)}")
        if waveform.size(1) != self.channels:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torch.stack(
                [resample_audio(example, sample_rate, self.sample_rate) for example in waveform],
                dim=0,
            )
        return waveform

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> CodecBatch:
        prepared = self._prepare(waveform, sample_rate)
        encoded = self.model.encoder(prepared)
        tokens = self.model.quantizer.encode(encoded)
        post = self.model.quantizer.decode(tokens)
        return CodecBatch(
            pre_quant_embeddings=encoded.transpose(1, 2),
            tokens=tokens.long(),
            post_quant_embeddings=post.transpose(1, 2),
            aux={},
        )

    def quantize_embeddings(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = embeddings.transpose(1, 2)
        tokens = self.model.quantizer.encode(embeddings)
        post = self.model.quantizer.decode(tokens)
        return tokens.long(), post.transpose(1, 2)

    def decode_tokens(self, tokens: torch.Tensor, aux: dict[str, Any] | None = None) -> torch.Tensor:
        quantized = self.model.quantizer.decode(tokens.long())
        decoded = self.model.decoder(quantized)
        return decoded.squeeze(1)
