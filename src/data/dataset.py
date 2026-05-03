from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio import crop_or_pad_1d, load_audio, prepare_waveform, seconds_to_samples


@dataclass
class AudioExample:
    utt_id: str
    speaker_id: str
    split: str
    waveform: torch.Tensor
    sample_rate: int
    path: str


class WSJ0SegmentDataset(Dataset[AudioExample]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        segment_seconds: float,
        sample_rate: int,
        normalize: bool = True,
        mono: bool = True,
        mode: str = "train",
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.segment_seconds = segment_seconds
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.mono = mono
        self.mode = mode
        self.segment_samples = seconds_to_samples(segment_seconds, sample_rate)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> AudioExample:
        row = self.manifest.iloc[index]
        waveform, source_sr = load_audio(row["path"])
        waveform = prepare_waveform(
            waveform,
            source_sr=source_sr,
            target_sr=self.sample_rate,
            mono=self.mono,
            normalize=self.normalize,
        )

        if self.mode == "train":
            waveform = crop_or_pad_1d(waveform, self.segment_samples)

        return AudioExample(
            utt_id=row["utt_id"],
            speaker_id=row["speaker_id"],
            split=row["split"],
            waveform=waveform.squeeze(0),
            sample_rate=self.sample_rate,
            path=row["path"],
        )


def collate_audio_batch(batch: list[AudioExample]) -> dict[str, Any]:
    lengths = torch.tensor([item.waveform.numel() for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    waveforms = []
    for item in batch:
        pad = max_len - item.waveform.numel()
        waveforms.append(torch.nn.functional.pad(item.waveform, (0, pad)))

    return {
        "waveforms": torch.stack(waveforms, dim=0),
        "lengths": lengths,
        "utt_ids": [item.utt_id for item in batch],
        "speaker_ids": [item.speaker_id for item in batch],
        "paths": [item.path for item in batch],
        "sample_rate": batch[0].sample_rate,
        "split": batch[0].split,
    }
