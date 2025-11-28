"""Utilities for visualising the processed OHLCV datasets.

The feature pipeline stores chronological train/validation/test splits under
``data/processed`` (or whatever custom directory you configured).  This module
loads those parquet/CSV files and renders a couple of quick diagnostic plots:

* closing price time series per split
* forward-return label distribution histograms
* example technical indicators (RSI and EMA ratio)

Run the script directly once the pipeline has produced the splits::

    python Data_Access/visualize_processed_ohlcv.py \
        --output-dir data/processed --format parquet --save-prefix figures/ohlcv

If ``--save-prefix`` is omitted, the plots will be shown in an interactive
window instead of being written to disk.  When the pipeline was configured to
fall back to CSV files you can pass ``--format csv`` or let the script detect
the file extension automatically.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SPLIT_ORDER: Tuple[str, ...] = ("train", "validation", "test")


def _maybe_import_pipeline_output_dir() -> List[Path]:
    """Return candidate paths derived from the feature pipeline configuration."""

    candidates: List[Path] = []
    try:  # Import lazily so running the visualiser has no heavy dependencies.
        from Data_Access import ohlcv_feature_pipeline as pipeline
    except Exception as exc:  # pragma: no cover - best effort helper
        logging.debug("Unable to import pipeline config: %s", exc)
        return candidates

    configured = Path(pipeline.PIPELINE_SETTINGS.output_dir)
    pipeline_dir = Path(pipeline.__file__).resolve().parent

    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.append(Path.cwd() / configured)
        candidates.append(pipeline_dir / configured)

    return candidates


def discover_processed_directory() -> Tuple[Optional[Path], List[Path]]:
    """Attempt to locate the processed data directory automatically."""

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates: List[Path] = [
        script_dir / "data" / "processed",
        project_root / "data" / "processed",
        Path.cwd() / "data" / "processed",
    ]

    candidates.extend(_maybe_import_pipeline_output_dir())

    # Deduplicate while preserving order.
    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate, unique_candidates

    return None, unique_candidates


def infer_file_format(output_dir: Path, requested: str | None) -> str:
    """Determine whether the processed data is stored as parquet or CSV."""

    if requested:
        return requested

    for candidate in ("parquet", "csv"):
        candidate_path = output_dir / f"train.{candidate}"
        if candidate_path.exists():
            logging.debug("Detected %s files in %%s", candidate, output_dir)
            return candidate

    raise FileNotFoundError(
        "Unable to find train.{parquet,csv} in the processed data directory. "
        "Run the feature pipeline first or provide --output-dir/--format."
    )


def _load_split(path: Path, file_format: str) -> pd.DataFrame:
    """Read a single dataset split and normalise its timestamp index."""

    if file_format == "parquet":
        frame = pd.read_parquet(path)
    elif file_format == "csv":
        frame = pd.read_csv(path)
    else:  # pragma: no cover - defensive branch, format validated earlier
        raise ValueError(f"Unsupported format '{file_format}'")

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame.set_index("timestamp", inplace=True)
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)

    frame.sort_index(inplace=True)
    return frame


def load_splits(
    output_dir: Path, file_format: str, split_order: Iterable[str]
) -> Dict[str, pd.DataFrame]:
    """Load all requested dataset splits into memory."""

    splits: Dict[str, pd.DataFrame] = {}
    for split_name in split_order:
        path = output_dir / f"{split_name}.{file_format}"
        if not path.exists() and file_format == "parquet":
            # When parquet support is unavailable the pipeline produces CSV
            alt_path = path.with_suffix(".csv")
            if alt_path.exists():
                logging.info(
                    "Expected %s but found %s – reading CSV fallback instead.",
                    path.name,
                    alt_path.name,
                )
                path = alt_path
                file_format = "csv"
            else:
                raise FileNotFoundError(f"Could not locate processed split at {path}")
        elif not path.exists():
            raise FileNotFoundError(f"Could not locate processed split at {path}")

        splits[split_name] = _load_split(path, file_format)
    return splits


def plot_close_series(splits: Dict[str, pd.DataFrame]) -> plt.Figure:
    """Generate a figure showing the close price for each dataset split."""

    n_axes = len(splits)
    fig, axes = plt.subplots(n_axes, 1, figsize=(12, 3 * n_axes), sharex=False)
    if n_axes == 1:
        axes = (axes,)

    for ax, (split_name, frame) in zip(axes, splits.items()):
        ax.plot(frame.index, frame["close"], label=f"{split_name} close")
        ax.set_title(f"{split_name.title()} close price")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    return fig


def plot_target_histograms(splits: Dict[str, pd.DataFrame]) -> plt.Figure | None:
    """Render histograms for the forward-return labels."""

    target_columns = None
    for frame in splits.values():
        target_columns = [c for c in frame.columns if c.startswith("target_return_")]
        if target_columns:
            break

    if not target_columns:
        logging.warning("No target_return_* columns found; skipping histogram plot.")
        return None

    target_column = target_columns[0]
    n_axes = len(splits)
    fig, axes = plt.subplots(n_axes, 1, figsize=(10, 3 * n_axes), sharex=True)
    if n_axes == 1:
        axes = (axes,)

    for ax, (split_name, frame) in zip(axes, splits.items()):
        series = frame[target_column].dropna()
        ax.hist(series, bins=50, alpha=0.8, color="#1f77b4")
        ax.set_title(f"{split_name.title()} distribution of {target_column}")
        ax.set_xlabel("Forward return")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_indicator_examples(splits: Dict[str, pd.DataFrame]) -> plt.Figure | None:
    """Plot a couple of the engineered indicator series for reference."""

    indicators = [
        ("rsi_14", "RSI"),
        ("ema_ratio", "EMA ratio"),
        ("volume_zscore_30", "Volume z-score"),
    ]

    available = [col for col, _ in indicators if any(col in df for df in splits.values())]
    if not available:
        logging.warning("No indicator columns present; skipping indicator plot.")
        return None

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3 * len(available)))
    if len(available) == 1:
        axes = (axes,)

    for ax, (column, label) in zip(axes, [(c, l) for c, l in indicators if c in available]):
        for split_name, frame in splits.items():
            if column not in frame:
                continue
            ax.plot(frame.index, frame[column], label=split_name)
        ax.set_title(f"{label} across splits")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    return fig


def parse_args(default_output: Optional[Path]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=(
            "Directory containing train/validation/test parquet or CSV files. "
            "Defaults to the location discovered from the feature pipeline output."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default=None,
        help="Override the processed file format. Auto-detected by default.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=None,
        help="If provided, save the generated figures using this prefix instead of showing them.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Force displaying the plots even when --save-prefix is supplied.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (e.g. INFO, DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    detected_output, tried_candidates = discover_processed_directory()
    args = parse_args(detected_output)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_dir = args.output_dir or detected_output
    if output_dir is None:
        searched = "\n".join(f"  - {path}" for path in tried_candidates)
        raise FileNotFoundError(
            "Unable to locate the processed data directory automatically. "
            "Pass --output-dir explicitly. Searched paths:\n" + searched
        )

    if not output_dir.exists():
        searched = "\n".join(f"  - {path}" for path in tried_candidates)
        raise FileNotFoundError(
            f"Processed data directory not found: {output_dir}.\n"
            f"Previously attempted locations include:\n{searched}"
        )

    if detected_output and output_dir == detected_output:
        logging.info("Using processed data directory at %s", output_dir)

    file_format = infer_file_format(output_dir, args.format)
    splits = load_splits(output_dir, file_format, DEFAULT_SPLIT_ORDER)

    figures = {
        "close_series": plot_close_series(splits),
        "target_histogram": plot_target_histograms(splits),
        "indicator_examples": plot_indicator_examples(splits),
    }

    figures = {name: fig for name, fig in figures.items() if fig is not None}

    if not figures:
        logging.warning("No figures were generated; nothing to display or save.")
        return

    if args.save_prefix:
        output_dir = args.save_prefix.parent
        if output_dir and not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)

        for name, fig in figures.items():
            filename = args.save_prefix.with_name(f"{args.save_prefix.name}_{name}.png")
            fig.savefig(filename, bbox_inches="tight")
            logging.info("Saved %s", filename)
            if not args.show:
                plt.close(fig)

    if args.show or not args.save_prefix:
        plt.show()


if __name__ == "__main__":
    main()
