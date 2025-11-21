"""Lightweight demo that trains the neural net on synthetic price data."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from Logic.nn_model import DataConfig, ModelConfig, TrainingConfig, run_training_pipeline


def _generate_price_series(points: int = 720, seed: int = 1234) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time_index = pd.date_range("2023-01-01", periods=points, freq="H")

    base_trend = 0.0004 * np.sin(np.linspace(0, 6 * np.pi, points))
    noise = rng.normal(0, 0.0015, points)
    returns = base_trend + noise
    close = 100 * np.exp(np.cumsum(returns))

    frame = pd.DataFrame({"close": close}, index=time_index)
    frame["return_1"] = frame["close"].pct_change().fillna(0.0)
    frame["return_3"] = frame["close"].pct_change(periods=3).fillna(0.0)
    frame["target_return_1"] = frame["return_1"].shift(-1)

    return frame.dropna()


def _split_data(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(frame) * 0.7)
    val_end = int(len(frame) * 0.85)
    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:val_end]
    test = frame.iloc[val_end:]
    return train, validation, test


def _write_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> DataConfig:
    data_dir = Path("data/processed/demo")
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(data_dir / "train.parquet")
    validation.to_parquet(data_dir / "validation.parquet")
    test.to_parquet(data_dir / "test.parquet")

    return DataConfig(
        data_dir=data_dir,
        target_column="target_return_1",
        feature_columns=["close", "return_1", "return_3"],
        lookback=30,
    )


def run_demo_training() -> None:
    dataset = _generate_price_series()
    train, validation, test = _split_data(dataset)
    data_cfg = _write_splits(train, validation, test)

    model_cfg = ModelConfig(architecture="lstm", hidden_size=32, num_layers=1, dropout=0.05)
    training_cfg = TrainingConfig(max_epochs=12, batch_size=64, patience=3, learning_rate=1e-3)

    results = run_training_pipeline(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        output_dir=Path("models/demo_predictor"),
    )

    print("Validation metrics:", results["val_metrics"])
    print("Test metrics:", results["test_metrics"])
    print("Artifacts saved under", results["metrics_path"].parent)


if __name__ == "__main__":
    run_demo_training()