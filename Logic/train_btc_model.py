"""Convenience CLI for training a BTC-USD forecasting model.

The script wires together the general-purpose utilities from
``Logic.nn_model`` with the processed datasets produced by
``Data_Access/ohlcv_feature_pipeline.py``.  It exposes a couple of
useful command line flags but intentionally keeps sensible defaults so
that ``python -m Logic.train_btc_model`` runs end-to-end without any
extra arguments.

Example usage::

    $ pip install -r requirements-training.txt
    $ python -m Logic.train_btc_model --epochs 25 --resume

The script prints validation/test metrics to stdout and persists the
model, scalers, and metadata in ``models/btc_usd`` by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from Logic.nn_model import DataConfig, ModelConfig, TrainingConfig, run_training_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "Data_Access" / "data" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "btc_usd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing train/val/test files")
    parser.add_argument("--target-column", default="target_return_5", help="Name of the supervised learning target column")
    parser.add_argument("--lookback", type=int, default=60, help="Number of past bars fed into the neural network")
    parser.add_argument("--architecture", choices=["lstm", "gru", "cnn"], default="lstm", help="Neural network backbone")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for recurrent layers")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of stacked recurrent layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied between layers")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Epochs without improvement before early stopping")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Hardware accelerator preference")
    parser.add_argument(
        "--scale-target",
        action="store_true",
        help="Standardise the target column in addition to the features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store checkpoints, scalers, and metrics",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Warm start training from an existing checkpoint in the output directory",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Using processed data from %s", parsed.data_dir)

    data_cfg = DataConfig(
        data_dir=parsed.data_dir,
        lookback=parsed.lookback,
        target_column=parsed.target_column,
        feature_columns=None,
        scale_target=parsed.scale_target,
    )

    model_cfg = ModelConfig(
        architecture=parsed.architecture,
        hidden_size=parsed.hidden_size,
        num_layers=parsed.num_layers,
        dropout=parsed.dropout,
    )

    training_cfg = TrainingConfig(
        batch_size=parsed.batch_size,
        max_epochs=parsed.epochs,
        patience=parsed.patience,
        learning_rate=parsed.learning_rate,
        device=parsed.device,
    )

    warm_start_path = None
    if parsed.resume:
        warm_start_path = parsed.output_dir / "best_model.pt"

    results = run_training_pipeline(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        output_dir=parsed.output_dir,
        warm_start_checkpoint=warm_start_path,
    )

    print("\nValidation metrics:")
    for key, value in results["val_metrics"].items():
        print(f"  {key}: {value:.6f}")

    print("\nTest metrics:")
    for key, value in results["test_metrics"].items():
        print(f"  {key}: {value:.6f}")

    history = results["history"]
    best_epoch = int(min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)) + 1
    print(f"\nBest epoch: {best_epoch}")
    print(f"Model saved to: {results['model_path']}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
