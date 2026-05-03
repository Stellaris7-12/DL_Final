from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


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


def load_history(path: str | Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    return pd.DataFrame(history)


def export_summary_tables(summary_df: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "codec_summary.csv"
    md_path = output_dir / "codec_summary.md"
    tex_path = output_dir / "codec_summary.tex"

    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    tex_path.write_text(summary_df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
    }


def plot_training_curves(
    history_paths: dict[str, str | Path],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for codec_name, history_path in history_paths.items():
        history = load_history(history_path)
        if history.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["epoch"], history["train_loss"], label="train_loss", linewidth=2)
        axes[0].plot(history["epoch"], history["val_loss"], label="val_loss", linewidth=2)
        axes[0].set_title(f"{codec_name} Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(history["epoch"], history["train_pre_loss"], label="train_pre", linewidth=2)
        axes[1].plot(history["epoch"], history["train_post_loss"], label="train_post", linewidth=2)
        axes[1].plot(history["epoch"], history["val_pre_loss"], label="val_pre", linewidth=2, linestyle="--")
        axes[1].plot(history["epoch"], history["val_post_loss"], label="val_post", linewidth=2, linestyle="--")
        axes[1].set_title(f"{codec_name} Loss Components")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        fig.tight_layout()
        output_path = output_dir / f"{codec_name}_training_curves.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs[codec_name] = str(output_path)
    return outputs


def plot_metric_comparison(summary_df: pd.DataFrame, output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["stoi_mean", "pesq_mean", "dnsmos_mean"]
    pretty_names = {
        "stoi_mean": "STOI",
        "pesq_mean": "PESQ",
        "dnsmos_mean": "DNSMOS",
    }

    melted = summary_df[["codec", *metrics]].melt(id_vars="codec", var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map(pretty_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="codec", ax=ax)
    ax.set_title("Codec Comparison on WSJCAM0")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    fig.tight_layout()

    output_path = output_dir / "codec_metric_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def prepare_report_materials(
    history_paths: dict[str, str | Path],
    summary_paths: dict[str, str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = summarize_codecs(summary_paths)
    table_paths = export_summary_tables(summary_df, output_dir)
    curve_paths = plot_training_curves(history_paths, output_dir)
    comparison_plot = plot_metric_comparison(summary_df, output_dir)

    return {
        "summary_dataframe": summary_df,
        "table_paths": table_paths,
        "curve_paths": curve_paths,
        "comparison_plot": comparison_plot,
        "output_dir": str(output_dir),
    }
