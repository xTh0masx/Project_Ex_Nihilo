"""Inference helpers that load trained sequence models for live signals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import importlib
import pickle
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from Logic.nn_model import DataConfig, ModelConfig, SequenceModel, _resolve_device

joblib_spec = importlib.util.find_spec("joblib")
joblib = importlib.import_module("joblib") if joblib_spec is not None else None


class IdentityScaler:
    """Minimal scaler that leaves values unchanged.

    This allows demo artefacts to be generated without requiring scikit-learn
    while still satisfying the ``transform`` interface expected by
    ``NeuralPricePredictor``.
    """

    def fit(self, _: np.ndarray) -> "IdentityScaler":  # pragma: no cover - trivial
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        return values

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        return values


class NeuralPricePredictor:
    """Generate next-step return forecasts from a saved neural network model."""

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.metrics_path = self.model_dir / "metrics.json"
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found in {self.model_dir}")

        with self.metrics_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.feature_columns: List[str] = metadata.get("feature_columns", [])
        data_cfg = DataConfig(**metadata["data_config"])
        model_cfg = ModelConfig(**metadata["model_config"])

        self.lookback = data_cfg.lookback
        self.target_scaled = data_cfg.scale_target

        self.device = _resolve_device(device)
        self.model = SequenceModel(input_size=len(self.feature_columns), cfg=model_cfg)
        state_dict = torch.load(self.model_dir / "best_model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        feature_scaler_path = self.model_dir / "feature_scaler.pkl"
        self.feature_scaler = self._load_scaler(feature_scaler_path)

        target_scaler_path = self.model_dir / "target_scaler.pkl"
        self.target_scaler = (
            self._load_scaler(target_scaler_path) if target_scaler_path.exists() else None
        )

    @staticmethod
    def _load_scaler(path: Path):
        if joblib is not None:
            return joblib.load(path)

        import pickle

        with path.open("rb") as handle:
            return pickle.load(handle)

    def _ensure_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(
                f"Feature columns {missing} missing from provided dataframe; expected {self.feature_columns}"
            )
        sanitized = frame[self.feature_columns].copy()
        sanitized.replace([np.inf, -np.inf], np.nan, inplace=True)
        sanitized.fillna(0.0, inplace=True)
        return sanitized

    def prepare_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure all configured features are present, deriving returns when needed."""

        working = frame.copy()
        if "close" not in working.columns:
            raise ValueError("Input frame must include a 'close' column for feature construction")

        def ensure_return_feature(feature: str, *, log: bool = False) -> None:
            period = self._extract_period(feature, prefix=("log_return_" if log else "return_"))
            series = np.log(working["close"]) if log else working["close"]
            transform = series.diff if log else series.pct_change
            working[feature] = transform(periods=period)

        def ensure_lag_feature(feature: str) -> None:
            period = self._extract_period(feature, prefix="lag_")
            working[feature] = working["close"].shift(periods=period)

        def ensure_ema_feature(feature: str) -> None:
            period = self._extract_period(feature, prefix="ema_")
            working[feature] = working["close"].ewm(span=period, adjust=False).mean()

        def ensure_rsi_feature(feature: str) -> None:
            window = self._extract_period(feature, prefix="rsi_")
            delta = working["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
            avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            working[feature] = 100 - (100 / (1 + rs))

        def ensure_volatility_feature(feature: str) -> None:
            window = self._extract_period(feature, prefix="volatility_")
            if "log_return_1" not in working.columns:
                ensure_return_feature("log_return_1", log=True)
            working[feature] = working["log_return_1"].rolling(window=window).std()

        def ensure_atr_feature(feature: str) -> None:
            window = self._extract_period(feature, prefix="atr_")
            for base_col in ("high", "low"):
                if base_col not in working.columns:
                    raise ValueError(
                        f"Feature column '{feature}' requires '{base_col}' values in the input frame"
                    )
            prev_close = working["close"].shift(1)
            ranges = pd.concat(
                [
                    working["high"] - working["low"],
                    (working["high"] - prev_close).abs(),
                    (working["low"] - prev_close).abs(),
                ],
                axis=1,
            )
            true_range = ranges.max(axis=1)
            working[feature] = true_range.rolling(window=window).mean()

        def ensure_volume_zscore_feature(feature: str) -> None:
            window = self._extract_period(feature, prefix="volume_zscore_")
            if "volume" not in working.columns:
                raise ValueError(
                    f"Feature column '{feature}' requires a 'volume' column in the input frame"
                )
            rolling_mean = working["volume"].rolling(window).mean()
            rolling_std = working["volume"].rolling(window).std()
            working[feature] = (working["volume"] - rolling_mean) / rolling_std

        for column in self.feature_columns:
            if column in working.columns:
                continue

            if column.startswith("return_"):
                ensure_return_feature(column)
                continue

            if column.startswith("log_return_"):
                ensure_return_feature(column, log=True)
                continue

            if column.startswith("lag_"):
                ensure_lag_feature(column)
                continue

            if column == "ema_ratio":
                if "ema_12" not in working.columns:
                    ensure_ema_feature("ema_12")
                if "ema_26" not in working.columns:
                    ensure_ema_feature("ema_26")
                working[column] = working["ema_12"] / working["ema_26"] - 1
                continue

            if column.startswith("ema_"):
                ensure_ema_feature(column)
                continue

            if column.startswith("rsi_"):
                ensure_rsi_feature(column)
                continue

            if column.startswith("volatility_"):
                ensure_volatility_feature(column)
                continue

            if column.startswith("atr_"):
                ensure_atr_feature(column)
                continue

            if column.startswith("volume_zscore_"):
                ensure_volume_zscore_feature(column)
                continue

            raise ValueError(
                f"Feature column '{column}' missing from provided dataframe; cannot infer its values"
            )

        return self._ensure_feature_frame(working)

    @staticmethod
    def _extract_period(feature_name: str, *, prefix: str) -> int:
        try:
            return int(feature_name[len(prefix) :])
        except ValueError as exc:
            raise ValueError(
                f"Return feature '{feature_name}' must follow the pattern '{prefix}<period>'"
            ) from exc

    def _features_from_prices(self, prices: Iterable[float]) -> pd.DataFrame:
        series = pd.Series(list(prices), dtype=float)
        frame = pd.DataFrame({"close": series})

        for column in self.feature_columns:
            if column.startswith("return_"):
                period = self._extract_period(column, prefix="return_")
                frame[column] = series.pct_change(periods=period)
            elif column.startswith("log_return_"):
                period = self._extract_period(column, prefix="log_return_")
                frame[column] = np.log(series).diff(periods=period)
            elif column.startswith("lag_"):
                period = self._extract_period(column, prefix="lag_")
                frame[column] = series.shift(periods=period)

        frame.index = pd.RangeIndex(len(frame))
        return self.prepare_feature_frame(frame)

    def _predict_tensor(self, features: np.ndarray) -> float:
        tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output: Tensor = self.model(tensor)  # type: ignore[assignment]
        raw_pred = output.cpu().numpy().squeeze().item()

        if self.target_scaler is not None:
            raw_pred = (
                self.target_scaler.inverse_transform(np.array([[raw_pred]])).squeeze().item()
            )

        return float(raw_pred)

    def predict_next_return(self, frame: pd.DataFrame) -> float:
        """Predict the next-step return using an already constructed feature frame."""

        ordered_features = self._ensure_feature_frame(frame)
        if len(ordered_features) < self.lookback:
            raise ValueError(
                f"At least {self.lookback} rows are required to build a prediction window"
            )

        window = ordered_features.tail(self.lookback).to_numpy()
        scaled = self.feature_scaler.transform(window)
        return self._predict_tensor(scaled)

    def predict_from_prices(self, prices: Iterable[float]) -> Optional[float]:
        """Convenience wrapper to derive features from close prices."""

        frame = self._features_from_prices(prices)
        if len(frame) < self.lookback:
            return None
        return self.predict_next_return(frame)


def create_minimal_demo_model(model_dir: Path) -> Path:
    """Materialize a lightweight demo artefact directory for examples/tests.

    The helper writes a tiny LSTM checkpoint together with feature/target
    scalers and accompanying metadata so that ``NeuralPricePredictor`` can load
    without running the full training pipeline. The model weights are random
    (no training), but sufficient for deterministic inference in example flows.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = ["close", "return_1"]
    data_cfg = DataConfig(
        lookback=15,
        feature_columns=feature_columns,
        scale_target=False,
    )
    model_cfg = ModelConfig(architecture="lstm", hidden_size=8, num_layers=1, dropout=0.0)

    # Persist the metadata the predictor expects.
    data_cfg_dict = data_cfg.__dict__.copy()
    data_cfg_dict["data_dir"] = str(data_cfg_dict["data_dir"])

    metadata = {
        "feature_columns": feature_columns,
        "data_config": data_cfg_dict,
        "model_config": model_cfg.__dict__,
        "val_mae": None,
        "test_mae": None,
    }
    metrics_path = model_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # Save a freshly initialised model checkpoint.
    model = SequenceModel(input_size=len(feature_columns), cfg=model_cfg)
    torch.save(model.state_dict(), model_dir / "best_model.pt")

    # Write simple no-op scalers for features/targets.
    feature_scaler = IdentityScaler()
    target_scaler = IdentityScaler()
    if joblib is not None:
        joblib.dump(feature_scaler, model_dir / "feature_scaler.pkl")
        joblib.dump(target_scaler, model_dir / "target_scaler.pkl")
    else:  # pragma: no cover - pickle fallback
        with (model_dir / "feature_scaler.pkl").open("wb") as handle:
            pickle.dump(feature_scaler, handle)
        with (model_dir / "target_scaler.pkl").open("wb") as handle:
            pickle.dump(target_scaler, handle)

    return model_dir