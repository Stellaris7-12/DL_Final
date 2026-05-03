from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str = "wsj0_codec_lm"
    seed: int = 42
    artifacts_dir: str = "artifacts"


@dataclass
class DatasetConfig:
    root: str = "wsj0"
    train_split: str = "si_tr_s"
    val_split: str = "si_dt_05"
    test_split: str = "si_et_05"
    sample_rate: int = 16000
    segment_seconds: float = 4.0
    normalize: bool = True
    mono: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 8
    max_epochs: int = 50
    learning_rate: float = 6e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 3.0
    context_frames: int = 80
    amp: bool = True
    device: str = "cuda"
    log_every_steps: int = 10
    log_every_epochs: int = 5
    checkpoint_metric: str = "val_loss"


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    ff_dim: int = 2048
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    token_embedding_dim: int = 512


@dataclass
class EvaluationConfig:
    max_eval_files: int | None = None
    dnsmos_personalized: bool = False
    save_audio_examples: int = 8
    pesq_mode: str = "wb"


@dataclass
class CodecsConfig:
    target_bandwidth_kbps: float = 6.0
    facodec_repo_id: str = "amphion/naturalspeech3_facodec"
    encodec_model: str = "24khz"
    hub_endpoint: str | None = None


@dataclass
class CodecSelection:
    name: str = "facodec"


@dataclass
class ExperimentConfig:
    project: ProjectConfig
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    codecs: CodecsConfig
    codec: CodecSelection

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.project.artifacts_dir)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(*config_paths: str | Path) -> ExperimentConfig:
    merged: dict[str, Any] = {}
    for config_path in config_paths:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged = _merge(merged, loaded)

    return ExperimentConfig(
        project=ProjectConfig(**merged.get("project", {})),
        dataset=DatasetConfig(**merged.get("dataset", {})),
        training=TrainingConfig(**merged.get("training", {})),
        model=ModelConfig(**merged.get("model", {})),
        evaluation=EvaluationConfig(**merged.get("evaluation", {})),
        codecs=CodecsConfig(**merged.get("codecs", {})),
        codec=CodecSelection(**merged.get("codec", {})),
    )
