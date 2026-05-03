from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

from src.codecs.base import CodecAdapter
from src.config import ExperimentConfig
from src.data.manifest import build_wsj0_manifest, save_manifest
from src.eval.metrics import align_waveforms, build_metric_fns
from src.models.predictor import CodecAutoregressivePredictor
from src.train.trainer import build_predictor
from src.train.windows import _normalize_token_layout
from src.utils.audio import prepare_waveform
from src.utils.runtime import choose_device, ensure_dir


@dataclass
class EvaluationArtifacts:
    manifest_path: str
    per_utterance_path: str
    summary_path: str
    samples_dir: str


def _load_checkpoint_model(
    checkpoint_path: str | Path,
    config: ExperimentConfig,
    codec: CodecAdapter,
    device: torch.device,
) -> CodecAutoregressivePredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_predictor(config, codec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _rollout_single(
    model: CodecAutoregressivePredictor,
    codec: CodecAdapter,
    waveform: torch.Tensor,
    sample_rate: int,
    context_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    codec_batch = codec.encode(waveform.unsqueeze(0), sample_rate)
    pre = codec_batch.pre_quant_embeddings
    tokens = _normalize_token_layout(codec_batch.tokens, batch_size=1)
    aux = codec_batch.aux

    total_frames = pre.size(1)
    if total_frames <= context_frames:
        raise ValueError(f"Utterance too short for context_frames={context_frames}")

    predicted_pre = pre[:, :context_frames, :].clone()
    predicted_tokens = tokens[:, :, :context_frames].clone()

    for frame_index in range(context_frames, total_frames):
        context_emb = predicted_pre[:, -context_frames:, :]
        context_tokens = predicted_tokens[:, :, -context_frames:].transpose(1, 2)
        next_pre = model(context_emb, context_tokens)
        next_tokens, _ = codec.quantize_embeddings(next_pre.unsqueeze(1))
        next_tokens = _normalize_token_layout(next_tokens, batch_size=1)
        predicted_pre = torch.cat([predicted_pre, next_pre.unsqueeze(1)], dim=1)
        predicted_tokens = torch.cat([predicted_tokens, next_tokens], dim=2)

    future_pred_tokens = predicted_tokens[:, :, context_frames:]
    recon_future = codec.decode_tokens(future_pred_tokens, aux=aux).squeeze(0)

    frame_cfg = codec.frame_config()
    future_start = context_frames * frame_cfg.samples_per_frame
    prepared = waveform
    if waveform.dim() == 2:
        prepared = waveform.squeeze(0)
    target_future = prepared[future_start:]
    return recon_future, target_future, aux


def evaluate_codec_lm(
    config: ExperimentConfig,
    codec: CodecAdapter,
    checkpoint_path: str | Path,
) -> EvaluationArtifacts:
    warnings.filterwarnings("ignore")
    device = choose_device(config.training.device)
    codec = codec.to(device)
    codec.eval()
    test_manifest = build_wsj0_manifest(config.dataset.root, config.dataset.test_split)
    if config.evaluation.max_eval_files is not None:
        test_manifest = test_manifest.iloc[: config.evaluation.max_eval_files].copy()

    run_dir = ensure_dir(config.artifacts_dir / config.codec.name)
    eval_dir = ensure_dir(run_dir / "evaluation")
    samples_dir = ensure_dir(eval_dir / "samples")
    manifest_path = eval_dir / "test_manifest.csv"
    save_manifest(test_manifest, manifest_path)

    model = _load_checkpoint_model(checkpoint_path, config, codec, device)
    metric_fns = build_metric_fns(
        sample_rate=config.dataset.sample_rate,
        pesq_mode=config.evaluation.pesq_mode,
        personalized=config.evaluation.dnsmos_personalized,
    )

    rows: list[dict[str, Any]] = []
    for sample_index, row in enumerate(tqdm(test_manifest.to_dict("records"), desc=f"evaluate {config.codec.name}")):
        waveform, source_sr = sf.read(row["path"], dtype="float32")
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
        waveform = prepare_waveform(
            waveform,
            source_sr=source_sr,
            target_sr=config.dataset.sample_rate,
            mono=config.dataset.mono,
            normalize=config.dataset.normalize,
        ).squeeze(0).to(device)

        pred_future, target_future, _ = _rollout_single(
            model=model,
            codec=codec,
            waveform=waveform,
            sample_rate=config.dataset.sample_rate,
            context_frames=config.training.context_frames,
        )
        pred_future = pred_future.detach().cpu()
        target_future = target_future.detach().cpu()
        pred_future, target_future = align_waveforms(pred_future, target_future)

        row_metrics = {
            "utt_id": row["utt_id"],
            "speaker_id": row["speaker_id"],
            "stoi": metric_fns["stoi"](pred_future, target_future),
            "pesq": metric_fns["pesq"](pred_future, target_future),
            "dnsmos": metric_fns["dnsmos"](pred_future),
        }
        rows.append(row_metrics)

        if sample_index < config.evaluation.save_audio_examples:
            sf.write(samples_dir / f"{row['utt_id']}_pred.wav", pred_future.numpy(), config.dataset.sample_rate)
            sf.write(samples_dir / f"{row['utt_id']}_target.wav", target_future.numpy(), config.dataset.sample_rate)

    results = pd.DataFrame(rows)
    per_utterance_path = eval_dir / "per_utterance_metrics.csv"
    summary_path = eval_dir / "summary_metrics.json"
    results.to_csv(per_utterance_path, index=False)
    summary = {
        "codec": config.codec.name,
        "num_files": int(len(results)),
        "stoi_mean": float(results["stoi"].mean()),
        "stoi_std": float(results["stoi"].std(ddof=0)),
        "pesq_mean": float(results["pesq"].mean()),
        "pesq_std": float(results["pesq"].std(ddof=0)),
        "dnsmos_mean": float(results["dnsmos"].mean()),
        "dnsmos_std": float(results["dnsmos"].std(ddof=0)),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    return EvaluationArtifacts(
        manifest_path=str(manifest_path),
        per_utterance_path=str(per_utterance_path),
        summary_path=str(summary_path),
        samples_dir=str(samples_dir),
    )
