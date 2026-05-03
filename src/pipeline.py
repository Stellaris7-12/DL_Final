from __future__ import annotations

from src.codecs.factory import build_codec_adapter
from src.config import ExperimentConfig
from src.eval.rollout import EvaluationArtifacts, evaluate_codec_lm
from src.train.trainer import TrainArtifacts, train_codec_lm


def run_training(config: ExperimentConfig) -> TrainArtifacts:
    codec = build_codec_adapter(config)
    return train_codec_lm(config, codec)


def run_evaluation(config: ExperimentConfig, checkpoint_path: str) -> EvaluationArtifacts:
    codec = build_codec_adapter(config)
    return evaluate_codec_lm(config, codec, checkpoint_path)
