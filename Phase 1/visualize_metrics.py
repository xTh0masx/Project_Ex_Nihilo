"""Utilities to create presentation-friendly visuals from training metrics.

Run directly to render a dashboard PNG for each ``metrics.json`` in ``models``::

    python -m src.visualize_metrics

Or point to a specific metrics file::

    python -m src.visualize_metrics --metrics models/btc_usd/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "matplotlib wird benötigt, um die Dashboards zu rendern. "
        "Bitte installieren mit `pip install matplotlib`."
    ) from exc


def _plot_training_history(ax, history: Dict[str, List[float]], title: str) -> None:
    """Plot training and validation loss curves."""
    train = history.get("train_loss") or []
    val = history.get("val_loss") or []
    epochs = range(1, max(len(train), len(val)) + 1)

    if train:
        ax.plot(epochs[: len(train)], train, label="Train Loss", marker="o")
    if val:
        ax.plot(epochs[: len(val)], val, label="Validation Loss", marker="o")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)


def _plot_metric_bars(ax, metrics: Dict[str, float], title: str, color: str) -> None:
    labels = list(metrics.keys())
    values = list(metrics.values())
    ax.bar(labels, values, color=color, alpha=0.8)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


def _render_feature_list(ax, features: Iterable[str], target: str) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    text_lines = ["Features"]
    text_lines.extend(f"• {f}" for f in features)
    text_lines.append("")
    text_lines.append(f"Target: {target}")

    ax.text(
        0,
        1,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=11,
        transform=ax.transAxes,
        wrap=True,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )


def _render_config(ax, model_config: Dict, training_config: Dict, lookback: Optional[int]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    lines = ["Modell & Training"]
    lines.append("Model:")
    lines.extend(f"  - {k}: {v}" for k, v in model_config.items())
    lines.append("")
    lines.append("Training:")
    lines.extend(f"  - {k}: {v}" for k, v in training_config.items())
    if lookback is not None:
        lines.append("")
        lines.append(f"Lookback-Window: {lookback}")

    ax.text(
        0,
        1,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        transform=ax.transAxes,
        wrap=True,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )


def create_dashboard(metrics_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Create a PNG dashboard from a metrics.json file."""
    with metrics_path.open() as f:
        metrics = json.load(f)

    model_name = metrics_path.parent.name
    output_dir = output_dir or metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_dashboard.png"

    history = metrics.get("history", {})
    val_metrics = metrics.get("val_metrics", {})
    test_metrics = metrics.get("test_metrics", {})
    feature_columns = metrics.get("feature_columns") or []
    target = metrics.get("target_column") or metrics.get("data_config", {}).get("target_column", "")
    model_config = metrics.get("model_config", {})
    training_config = metrics.get("training_config", {})
    lookback = metrics.get("data_config", {}).get("lookback")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Modellevaluierung: {model_name}", fontsize=14)

    _plot_training_history(axes[0, 0], history, "Train vs. Validation Loss")
    _plot_metric_bars(axes[0, 1], val_metrics, "Validierungsmetriken", "#4c72b0")
    _plot_metric_bars(axes[1, 0], test_metrics, "Testmetriken", "#55a868")
    _render_feature_list(axes[1, 1], feature_columns, target)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add a small inset with model/training config for quick reference.
    inset_ax = fig.add_axes([0.68, 0.05, 0.28, 0.35])
    _render_config(inset_ax, model_config, training_config, lookback)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _find_metrics_files(metrics: Optional[Path], models_dir: Path) -> List[Path]:
    if metrics:
        return [metrics]
    return sorted(models_dir.glob("**/metrics.json"))


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Visualize training evaluations for presentation ready slides.")
    parser.add_argument("--metrics", type=Path, help="Path to a specific metrics.json file.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Root directory to scan for metrics.json files when --metrics is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where generated PNGs should be written. Defaults to the metrics file parent.",
    )

    args = parser.parse_args()

    metrics_path = args.metrics.resolve() if args.metrics else None
    models_dir = (args.models_dir or repo_root / "models").resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else None

    metrics_files = _find_metrics_files(metrics_path, models_dir)
    if not metrics_files:
        raise SystemExit("Keine metrics.json gefunden. Bitte Pfad oder models-Verzeichnis prüfen.")

    for metrics_file in metrics_files:
        output_path = create_dashboard(metrics_file, output_dir)
        print(f"Gespeichert: {output_path}")


if __name__ == "__main__":
    main()