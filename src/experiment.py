from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig, load_experiment_config
from src.eval.rollout import EvaluationArtifacts
from src.pipeline import run_evaluation, run_training
from src.reporting import prepare_report_materials, summarize_codecs
from src.train.trainer import TrainArtifacts


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_experiment_config(codec_name: str, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    config = load_experiment_config("configs/default.yaml", f"configs/{codec_name}.yaml")
    if not overrides:
        return config

    data = {
        "project": config.project.__dict__,
        "dataset": config.dataset.__dict__,
        "training": config.training.__dict__,
        "model": config.model.__dict__,
        "evaluation": config.evaluation.__dict__,
        "codecs": config.codecs.__dict__,
        "codec": config.codec.__dict__,
    }
    merged = _merge_dict(data, overrides)
    return ExperimentConfig(
        project=replace(config.project, **merged["project"]),
        dataset=replace(config.dataset, **merged["dataset"]),
        training=replace(config.training, **merged["training"]),
        model=replace(config.model, **merged["model"]),
        evaluation=replace(config.evaluation, **merged["evaluation"]),
        codecs=replace(config.codecs, **merged["codecs"]),
        codec=replace(config.codec, **merged["codec"]),
    )


def run_codec_experiment(
    codec_name: str,
    overrides: dict[str, Any] | None = None,
) -> tuple[ExperimentConfig, TrainArtifacts, EvaluationArtifacts]:
    config = build_experiment_config(codec_name, overrides=overrides)
    train_artifacts = run_training(config)
    evaluation_artifacts = run_evaluation(config, train_artifacts.checkpoint_path)
    return config, train_artifacts, evaluation_artifacts


def compare_codecs(summary_paths: dict[str, str | Path]):
    return summarize_codecs(summary_paths)


def build_report_assets(
    history_paths: dict[str, str | Path],
    summary_paths: dict[str, str | Path],
    output_dir: str | Path = "artifacts/report",
):
    return prepare_report_materials(
        history_paths=history_paths,
        summary_paths=summary_paths,
        output_dir=output_dir,
    )
