"""Render a compact SVG of training curves from a ``metrics.json`` file.

Example:
    python Logic/plot_training_metrics.py \
        --metrics models/btc_usd/metrics.json \
        --output reports/btc_usd_training_history.svg

The script depends only on the Python standard library, so it works in
restricted environments where extra plotting packages cannot be installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_losses(history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    if not history:
        raise ValueError("No training history available in metrics file")

    train = history.get("train_loss")
    val = history.get("val_loss")
    if not train or not val:
        raise ValueError("History is missing 'train_loss' or 'val_loss'")
    if len(train) != len(val):
        raise ValueError("Train and validation loss lists must be the same length")
    return {"train": train, "val": val}


def _format_metrics(block: Dict[str, float]) -> Iterable[str]:
    for key, value in block.items():
        yield f"{key}: {value:.6f}"


def _scale_points(values: List[float], width: int, height: int, margins: Tuple[int, int, int, int]):
    left, right, top, bottom = margins
    usable_width = width - left - right
    usable_height = height - top - bottom

    max_val = max(values)
    min_val = min(values)
    span = max_val - min_val if max_val != min_val else 1.0

    scaled = []
    for i, val in enumerate(values):
        x = left + usable_width * (i / max(1, len(values) - 1))
        y = top + usable_height * (1 - (val - min_val) / span)
        scaled.append((x, y))
    return scaled, min_val, max_val


def _polyline(points: List[Tuple[float, float]], color: str) -> str:
    coord_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coord_str}" />'


def _legend(x: int, y: int) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="160" height="60" rx="6" '
        'fill="#f5f5f5" stroke="#cccccc" />'
        f'<line x1="{x + 10}" y1="{y + 20}" x2="{x + 40}" y2="{y + 20}" stroke="#1f77b4" stroke-width="2" />'
        f'<text x="{x + 50}" y="{y + 24}" font-size="12" fill="#222">Train loss</text>'
        f'<line x1="{x + 10}" y1="{y + 40}" x2="{x + 40}" y2="{y + 40}" stroke="#ff7f0e" stroke-width="2" />'
        f'<text x="{x + 50}" y="{y + 44}" font-size="12" fill="#222">Validation loss</text>'
    )


def _metrics_box(x: int, y: int, lines: List[str]) -> str:
    if not lines:
        return ""
    line_height = 16
    height = line_height * len(lines) + 16
    contents = [f'<rect x="{x}" y="{y}" width="220" height="{height}" rx="6" fill="#f5f5f5" stroke="#cccccc" />']
    for i, line in enumerate(lines):
        contents.append(
            f'<text x="{x + 10}" y="{y + 22 + i * line_height}" font-size="12" fill="#222">{line}</text>'
        )
    return "".join(contents)


def plot_training_history(metrics_path: Path, output_path: Path) -> Path:
    metrics = _load_metrics(metrics_path)
    losses = _extract_losses(metrics.get("history", {}))

    width, height = 900, 520
    margins = (70, 50, 60, 60)
    train_points, min_val, max_val = _scale_points(losses["train"], width, height, margins)
    val_points, _, _ = _scale_points(losses["val"], width, height, margins)

    y_ticks = 5
    y_step = (max_val - min_val) / max(1, y_ticks - 1)
    y_labels = [min_val + i * y_step for i in range(y_ticks)]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family: Arial, sans-serif;}</style>',
        f'<text x="{width/2}" y="30" font-size="18" text-anchor="middle" fill="#222">Model training history</text>',
        # Axes
        f'<line x1="{margins[0]}" y1="{height - margins[3]}" x2="{width - margins[1]}" y2="{height - margins[3]}" stroke="#000" />',
        f'<line x1="{margins[0]}" y1="{height - margins[3]}" x2="{margins[0]}" y2="{margins[2]}" stroke="#000" />',
    ]

    # Y-axis ticks and labels
    for label in y_labels:
        y = margins[2] + (height - margins[2] - margins[3]) * (1 - (label - min_val) / (max_val - min_val or 1))
        svg_parts.append(f'<line x1="{margins[0]-5}" y1="{y:.2f}" x2="{margins[0]}" y2="{y:.2f}" stroke="#000" />')
        svg_parts.append(f'<text x="{margins[0]-10}" y="{y+4:.2f}" font-size="12" text-anchor="end" fill="#333">{label:.5f}</text>')

    # X-axis labels (epochs)
    for epoch, (x, y) in enumerate(train_points, start=1):
        svg_parts.append(f'<text x="{x:.2f}" y="{height - margins[3] + 20}" font-size="12" text-anchor="middle" fill="#333">{epoch}</text>')

    svg_parts.append(_polyline(train_points, "#1f77b4"))
    svg_parts.append(_polyline(val_points, "#ff7f0e"))
    svg_parts.append(_legend(width - margins[1] - 170, margins[2] + 10))

    summary_lines: List[str] = []
    val_metrics = metrics.get("val_metrics") or {}
    test_metrics = metrics.get("test_metrics") or {}
    if val_metrics:
        summary_lines.append("Validation metrics:")
        summary_lines.extend(_format_metrics(val_metrics))
    if test_metrics:
        if summary_lines:
            summary_lines.append("")
        summary_lines.append("Test metrics:")
        summary_lines.extend(_format_metrics(test_metrics))

    if summary_lines:
        svg_parts.append(_metrics_box(width - margins[1] - 230, margins[2] + 90, summary_lines))

    svg_parts.append('</svg>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/validation loss from metrics.json")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics.json produced by training")
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Where to save the SVG plot (default: alongside metrics.json)",
    )
    args = parser.parse_args()

    metrics_path = args.metrics.resolve()
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    if args.output:
        output_path = args.output
    else:
        output_path = metrics_path.with_name(f"{metrics_path.stem}_history.svg")

    saved_to = plot_training_history(metrics_path, output_path)
    print(f"Saved training history plot to {saved_to}")


if __name__ == "__main__":
    main()