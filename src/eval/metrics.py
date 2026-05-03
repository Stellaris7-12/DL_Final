from __future__ import annotations

import torch


def build_metric_fns(sample_rate: int, pesq_mode: str = "wb", personalized: bool = False) -> dict[str, callable]:
    from torchmetrics.audio import (
        DeepNoiseSuppressionMeanOpinionScore,
        PerceptualEvaluationSpeechQuality,
        ShortTimeObjectiveIntelligibility,
    )

    stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=False)
    pesq = PerceptualEvaluationSpeechQuality(sample_rate, pesq_mode)
    dnsmos = DeepNoiseSuppressionMeanOpinionScore(sample_rate, personalized=personalized)

    def _batchify(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x

    def _stoi(pred: torch.Tensor, target: torch.Tensor) -> float:
        value = stoi(_batchify(pred.cpu()), _batchify(target.cpu()))
        stoi.reset()
        return float(value.item())

    def _pesq(pred: torch.Tensor, target: torch.Tensor) -> float:
        value = pesq(_batchify(pred.cpu()), _batchify(target.cpu()))
        pesq.reset()
        return float(value.item())

    def _dnsmos(pred: torch.Tensor, target: torch.Tensor | None = None) -> float:
        value = dnsmos(_batchify(pred.cpu()))
        dnsmos.reset()
        if value.ndim == 0:
            return float(value.item())
        return float(value[..., -1].mean().item())

    return {
        "stoi": _stoi,
        "pesq": _pesq,
        "dnsmos": _dnsmos,
    }


def align_waveforms(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(pred.numel(), target.numel())
    return pred[:length], target[:length]
