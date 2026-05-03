from __future__ import annotations

from src.codecs.base import CodecAdapter
from src.codecs.encodec_adapter import EncodecAdapter
from src.codecs.facodec_adapter import FACodecAdapter
from src.config import ExperimentConfig


def build_codec_adapter(config: ExperimentConfig) -> CodecAdapter:
    if config.codec.name == "facodec":
        return FACodecAdapter(
            repo_id=config.codecs.facodec_repo_id,
            hub_endpoint=config.codecs.hub_endpoint,
        )
    if config.codec.name == "encodec":
        return EncodecAdapter(
            model_name=config.codecs.encodec_model,
            target_bandwidth_kbps=config.codecs.target_bandwidth_kbps,
        )
    raise ValueError(f"Unsupported codec: {config.codec.name}")
