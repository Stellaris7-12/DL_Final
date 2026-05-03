from __future__ import annotations

from typing import Any

import torch
from huggingface_hub import hf_hub_download

from src.codecs.base import CodecAdapter, CodecBatch, CodecFrameConfig
from src.utils.audio import resample_audio


class FACodecAdapter(CodecAdapter):
    def __init__(self, repo_id: str = "amphion/naturalspeech3_facodec") -> None:
        super().__init__()
        try:
            from ns3_codec import FACodecDecoder, FACodecEncoder
        except ImportError as exc:
            raise ImportError(
                "The `naturalspeech3_facodec` package is required for FACodecAdapter. "
                "Install dependencies from requirements.txt."
            ) from exc

        encoder_ckpt = hf_hub_download(repo_id=repo_id, filename="ns3_facodec_encoder.bin")
        decoder_ckpt = hf_hub_download(repo_id=repo_id, filename="ns3_facodec_decoder.bin")

        self.encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        self.decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )

        self.encoder.load_state_dict(torch.load(encoder_ckpt, map_location="cpu"))
        self.decoder.load_state_dict(torch.load(decoder_ckpt, map_location="cpu"))

        self.name = "facodec"
        self.sample_rate = 16_000
        self.embedding_dim = 256
        self.num_codebooks = 6
        self.codebook_size = 1024
        self.samples_per_frame = 200
        self.frame_shift_seconds = self.samples_per_frame / self.sample_rate
        self.repo_id = repo_id

    def frame_config(self) -> CodecFrameConfig:
        return CodecFrameConfig(
            sample_rate=self.sample_rate,
            frame_shift=self.frame_shift_seconds,
            samples_per_frame=self.samples_per_frame,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
        )

    def _prepare(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3:
            raise ValueError(f"Expected waveform shape [B, T] or [B, 1, T], got {tuple(waveform.shape)}")
        if waveform.size(1) != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torch.stack(
                [resample_audio(example, sample_rate, self.sample_rate) for example in waveform],
                dim=0,
            )
        return waveform

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> CodecBatch:
        prepared = self._prepare(waveform, sample_rate)
        encoded = self.encoder(prepared)
        vq_post_emb, vq_ids, _, _, speaker_embeddings = self.decoder(
            encoded,
            eval_vq=False,
            vq=True,
        )
        if vq_ids.size(1) == prepared.size(0):
            normalized_tokens = vq_ids.permute(1, 0, 2).contiguous()
        else:
            normalized_tokens = vq_ids
        return CodecBatch(
            pre_quant_embeddings=encoded.transpose(1, 2),
            tokens=normalized_tokens.long(),
            post_quant_embeddings=vq_post_emb.transpose(1, 2),
            aux={"speaker_embeddings": speaker_embeddings},
        )

    def quantize_embeddings(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = embeddings.transpose(1, 2)
        vq_post_emb, vq_ids, _, _, _ = self.decoder(
            embeddings,
            eval_vq=False,
            vq=True,
        )
        if vq_ids.size(1) == embeddings.size(0):
            normalized_tokens = vq_ids.permute(1, 0, 2).contiguous()
        else:
            normalized_tokens = vq_ids
        return normalized_tokens.long(), vq_post_emb.transpose(1, 2)

    def decode_tokens(self, tokens: torch.Tensor, aux: dict[str, Any] | None = None) -> torch.Tensor:
        if aux is None or "speaker_embeddings" not in aux:
            raise ValueError("FACodec decoding requires `speaker_embeddings` in aux.")
        speaker_embeddings = aux["speaker_embeddings"]
        if tokens.size(0) == speaker_embeddings.size(0):
            tokens = tokens.permute(1, 0, 2).contiguous()
        if hasattr(self.decoder, "vq2emb"):
            post_quant = self.decoder.vq2emb(tokens.long())
        else:
            raise RuntimeError("The installed FACodec package does not expose `vq2emb`.")
        decoded = self.decoder.inference(post_quant, speaker_embeddings)
        return decoded.squeeze(1)
