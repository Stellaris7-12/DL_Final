from __future__ import annotations

import torch
import torch.nn.functional as F


def codec_prediction_loss(
    predicted_pre_quant: torch.Tensor,
    target_pre_quant: torch.Tensor,
    predicted_post_quant: torch.Tensor,
    target_post_quant: torch.Tensor,
    post_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    pre_loss = F.l1_loss(predicted_pre_quant, target_pre_quant)
    post_loss = F.l1_loss(predicted_post_quant, target_post_quant)
    total = pre_loss + post_weight * post_loss
    return total, {
        "loss": float(total.detach().item()),
        "pre_loss": float(pre_loss.detach().item()),
        "post_loss": float(post_loss.detach().item()),
    }
