from __future__ import annotations

import math

import torch
import torchaudio


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max().clamp_min(1e-8)
    return waveform / peak


def resample_audio(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def prepare_waveform(
    waveform: torch.Tensor,
    source_sr: int,
    target_sr: int,
    mono: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    if mono:
        waveform = to_mono(waveform)
    waveform = resample_audio(waveform, source_sr, target_sr)
    if normalize:
        waveform = normalize_waveform(waveform)
    return waveform


def crop_or_pad_1d(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    current = waveform.size(-1)
    if current == target_samples:
        return waveform
    if current > target_samples:
        start = torch.randint(0, current - target_samples + 1, (1,)).item()
        return waveform[..., start : start + target_samples]

    pad = target_samples - current
    return torch.nn.functional.pad(waveform, (0, pad))


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate))
