from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.codecs.base import CodecAdapter
from src.config import ExperimentConfig
from src.data.dataset import WSJ0SegmentDataset, collate_audio_batch
from src.data.manifest import build_wsj0_manifest, save_manifest
from src.models.predictor import CodecAutoregressivePredictor
from src.train.losses import codec_prediction_loss
from src.train.windows import sample_training_windows
from src.utils.runtime import choose_device, ensure_dir, save_json, set_seed


@dataclass
class TrainArtifacts:
    run_dir: str
    checkpoint_path: str
    train_manifest_path: str
    val_manifest_path: str
    history_path: str


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, str, str]:
    manifests_dir = ensure_dir(config.artifacts_dir / "manifests")
    train_manifest = build_wsj0_manifest(config.dataset.root, config.dataset.train_split)
    val_manifest = build_wsj0_manifest(config.dataset.root, config.dataset.val_split)

    train_manifest_path = manifests_dir / f"{config.codec.name}_train.csv"
    val_manifest_path = manifests_dir / f"{config.codec.name}_val.csv"
    save_manifest(train_manifest, train_manifest_path)
    save_manifest(val_manifest, val_manifest_path)

    train_ds = WSJ0SegmentDataset(
        manifest=train_manifest,
        segment_seconds=config.dataset.segment_seconds,
        sample_rate=config.dataset.sample_rate,
        normalize=config.dataset.normalize,
        mono=config.dataset.mono,
        mode="train",
    )
    val_ds = WSJ0SegmentDataset(
        manifest=val_manifest,
        segment_seconds=config.dataset.segment_seconds,
        sample_rate=config.dataset.sample_rate,
        normalize=config.dataset.normalize,
        mono=config.dataset.mono,
        mode="eval",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, str(train_manifest_path), str(val_manifest_path)


def build_predictor(config: ExperimentConfig, codec: CodecAdapter) -> CodecAutoregressivePredictor:
    frame_cfg = codec.frame_config()
    return CodecAutoregressivePredictor(
        embedding_dim=frame_cfg.embedding_dim,
        codebook_size=frame_cfg.codebook_size,
        hidden_dim=config.model.hidden_dim,
        ff_dim=config.model.ff_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        token_embedding_dim=config.model.token_embedding_dim,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["waveforms"] = batch["waveforms"].to(device)
    moved["lengths"] = batch["lengths"].to(device)
    return moved


@torch.no_grad()
def _validation_epoch(
    model: CodecAutoregressivePredictor,
    codec: CodecAdapter,
    val_loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.eval()
    codec.eval()

    running = {"loss": 0.0, "pre_loss": 0.0, "post_loss": 0.0, "batches": 0}
    for batch in tqdm(val_loader, desc="validation", leave=False):
        batch = _move_batch(batch, device)
        codec_batch = codec.encode(batch["waveforms"], batch["sample_rate"])
        windows = sample_training_windows(codec_batch, config.training.context_frames)
        pred_pre = model(windows["context_embeddings"], windows["context_tokens"])
        _, pred_post = codec.quantize_embeddings(pred_pre.unsqueeze(1))
        pred_post = pred_post.squeeze(1)
        loss, metrics = codec_prediction_loss(
            predicted_pre_quant=pred_pre,
            target_pre_quant=windows["target_pre"],
            predicted_post_quant=pred_post,
            target_post_quant=windows["target_post"],
        )
        running["loss"] += float(loss.item())
        running["pre_loss"] += metrics["pre_loss"]
        running["post_loss"] += metrics["post_loss"]
        running["batches"] += 1

    denom = max(running["batches"], 1)
    return {
        "val_loss": running["loss"] / denom,
        "val_pre_loss": running["pre_loss"] / denom,
        "val_post_loss": running["post_loss"] / denom,
    }


def train_codec_lm(config: ExperimentConfig, codec: CodecAdapter) -> TrainArtifacts:
    warnings.filterwarnings("ignore")
    set_seed(config.project.seed)
    device = choose_device(config.training.device)
    codec = codec.to(device)
    for param in codec.parameters():
        param.requires_grad = False
    codec.eval()

    run_dir = ensure_dir(config.artifacts_dir / config.codec.name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    history_path = run_dir / "history.json"

    train_loader, val_loader, train_manifest_path, val_manifest_path = build_dataloaders(config)
    model = build_predictor(config, codec).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.amp and device.type == "cuda")
    best_val = math.inf
    best_ckpt = checkpoints_dir / "best.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, config.training.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_pre = 0.0
        epoch_post = 0.0
        steps = 0

        progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch in progress:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                codec_batch = codec.encode(batch["waveforms"], batch["sample_rate"])
                windows = sample_training_windows(codec_batch, config.training.context_frames)

            with torch.cuda.amp.autocast(enabled=config.training.amp and device.type == "cuda"):
                pred_pre = model(windows["context_embeddings"], windows["context_tokens"])
                pred_tokens, pred_post = codec.quantize_embeddings(pred_pre.unsqueeze(1))
                del pred_tokens
                pred_post = pred_post.squeeze(1)
                loss, metrics = codec_prediction_loss(
                    predicted_pre_quant=pred_pre,
                    target_pre_quant=windows["target_pre"],
                    predicted_post_quant=pred_post,
                    target_post_quant=windows["target_post"],
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += metrics["loss"]
            epoch_pre += metrics["pre_loss"]
            epoch_post += metrics["post_loss"]
            steps += 1
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps, 1),
            "train_pre_loss": epoch_pre / max(steps, 1),
            "train_post_loss": epoch_post / max(steps, 1),
        }
        val_metrics = _validation_epoch(model, codec, val_loader, device, config)
        merged = {**train_metrics, **val_metrics}
        history.append(merged)

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "metrics": merged,
                },
                best_ckpt,
            )

        save_json(history, history_path)
        should_log_epoch = (
            epoch == 1
            or epoch == config.training.max_epochs
            or epoch % config.training.log_every_epochs == 0
        )
        if should_log_epoch:
            tqdm.write(
                "[{codec}] epoch {epoch:03d}/{max_epoch:03d} "
                "train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                "train_pre={train_pre:.4f} train_post={train_post:.4f} "
                "val_pre={val_pre:.4f} val_post={val_post:.4f} best_val={best_val:.4f}".format(
                    codec=config.codec.name,
                    epoch=epoch,
                    max_epoch=config.training.max_epochs,
                    train_loss=merged["train_loss"],
                    val_loss=merged["val_loss"],
                    train_pre=merged["train_pre_loss"],
                    train_post=merged["train_post_loss"],
                    val_pre=merged["val_pre_loss"],
                    val_post=merged["val_post_loss"],
                    best_val=best_val,
                )
            )

    return TrainArtifacts(
        run_dir=str(run_dir),
        checkpoint_path=str(best_ckpt),
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        history_path=str(history_path),
    )
