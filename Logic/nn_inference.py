"""Inference helpers that load trained sequence models for live signals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import importlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from Logic.nn_model import DataConfig, ModelConfig, SequenceModel, _resolve_device

joblib_spec = importlib.util.find_spec("joblib")
joblib = importlib.import_module("joblib") if joblib_spec is not None else None


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
        return frame[self.feature_columns]

    def _features_from_prices(self, prices: Iterable[float]) -> pd.DataFrame:
        series = pd.Series(list(prices), dtype=float)
        data = {}

        if "close" in self.feature_columns:
            data["close"] = series

        for column in self.feature_columns:
            if column.startswith("return_"):
                try:
                    period = int(column.split("_")[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Return feature '{column}' must follow the pattern 'return_<period>'"
                    ) from exc
                data[column] = series.pct_change(periods=period).fillna(0.0)

        frame = pd.DataFrame(data)
        frame.index = pd.RangeIndex(len(frame))
        return self._ensure_feature_frame(frame)

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