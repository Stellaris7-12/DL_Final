from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_summary(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_codecs(summary_paths: dict[str, str | Path]) -> pd.DataFrame:
    rows = []
    for codec_name, path in summary_paths.items():
        row = load_summary(path)
        row["codec"] = codec_name
        rows.append(row)
    ordered = pd.DataFrame(rows)
    if "codec" in ordered.columns:
        first = ["codec"]
        rest = [column for column in ordered.columns if column not in first]
        ordered = ordered[first + rest]
    return ordered.sort_values("codec").reset_index(drop=True)
