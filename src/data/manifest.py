from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class AudioRecord:
    utt_id: str
    split: str
    speaker_id: str
    path: str


def infer_speaker_id(wav_path: Path) -> str:
    for parent in wav_path.parents:
        if parent.name.startswith("si_"):
            break
        if parent.name:
            return parent.name
    return wav_path.parent.name


def iter_split_records(split_dir: str | Path, split_name: str) -> Iterable[AudioRecord]:
    split_dir = Path(split_dir)
    for wav_path in sorted(split_dir.rglob("*.wav")):
        yield AudioRecord(
            utt_id=wav_path.stem,
            split=split_name,
            speaker_id=infer_speaker_id(wav_path),
            path=str(wav_path.resolve()),
        )


def build_wsj0_manifest(dataset_root: str | Path, split: str) -> pd.DataFrame:
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    records = [asdict(record) for record in iter_split_records(split_dir, split)]
    if not records:
        raise RuntimeError(f"No wav files found under {split_dir}")
    return pd.DataFrame(records)


def save_manifest(manifest: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
