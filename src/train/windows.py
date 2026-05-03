from __future__ import annotations

import torch

from src.codecs.base import CodecBatch


def _normalize_token_layout(tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tokens.dim() != 3:
        raise ValueError(f"Expected 3D tokens, got shape {tuple(tokens.shape)}")
    if tokens.size(0) == batch_size:
        return tokens
    if tokens.size(1) == batch_size:
        return tokens.permute(1, 0, 2).contiguous()
    raise ValueError(f"Could not normalize token layout for shape {tuple(tokens.shape)}")


def sample_training_windows(codec_batch: CodecBatch, context_frames: int) -> dict[str, torch.Tensor]:
    pre = codec_batch.pre_quant_embeddings
    post = codec_batch.post_quant_embeddings
    tokens = _normalize_token_layout(codec_batch.tokens, pre.size(0))
    batch_size, total_frames, _ = pre.shape

    if total_frames <= context_frames:
        raise ValueError(
            f"Need more than {context_frames} frames but only got {total_frames}. "
            "Increase segment length or reduce context_frames."
        )

    starts = torch.randint(
        low=0,
        high=total_frames - context_frames,
        size=(batch_size,),
        device=pre.device,
    )

    context_embeddings = []
    context_tokens = []
    target_pre = []
    target_post = []
    for batch_index in range(batch_size):
        start = int(starts[batch_index].item())
        end = start + context_frames
        context_embeddings.append(pre[batch_index, start:end, :])
        context_tokens.append(tokens[batch_index, :, start:end].transpose(0, 1))
        target_pre.append(pre[batch_index, end, :])
        target_post.append(post[batch_index, end, :])

    return {
        "context_embeddings": torch.stack(context_embeddings, dim=0),
        "context_tokens": torch.stack(context_tokens, dim=0),
        "target_pre": torch.stack(target_pre, dim=0),
        "target_post": torch.stack(target_post, dim=0),
    }
