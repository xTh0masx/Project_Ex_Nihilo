"""Neural network training utilities for OHLCV-based forecasting.

This module provides a complete modeling pipeline consisting of:

* Sliding-window dataset preparation using engineered OHLCV features.
* Configurable neural network architectures (LSTM, GRU, or 1D CNN).
* Training and validation loops with early stopping and metric tracking.
* Persistence of the best model checkpoint together with scaling artefacts.

The implementation intentionally keeps the configuration surface small so it
can be adapted quickly to new assets or horizons. The default configuration
assumes that the feature pipeline from ``Data_Access/ohlcv_feature_pipeline``
was executed and produced ``train`` / ``validation`` / ``test`` parquet files
in ``data/processed``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

try:  # Optional dependency that ships with scikit-learn
    import joblib
except Exception:  # pragma: no cover - fallback when joblib is unavailable
    joblib = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - bubble up a clear error message
    raise ImportError(
        "scikit-learn is required for scaling features. Install it via 'pip install scikit-learn'."
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Configuration for locating processed datasets and preparing windows."""

    data_dir: Path = Path("data/processed")
    train_file: str = "train.parquet"
    val_file: str = "validation.parquet"
    test_file: str = "test.parquet"
    lookback: int = 60
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    scale_target: bool = False

    def resolve_paths(self) -> Tuple[Path, Path, Path]:
        """Return the full paths for the train/validation/test splits."""

        train_path = (self.data_dir / self.train_file).resolve()
        val_path = (self.data_dir / self.val_file).resolve()
        test_path = (self.data_dir / self.test_file).resolve()
        return train_path, val_path, test_path


@dataclass
class ModelConfig:
    """Hyper-parameters controlling the neural network architecture."""

    architecture: Literal["lstm", "gru", "cnn"] = "lstm"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    cnn_channels: int = 32
    cnn_kernel_size: int = 3
    output_size: int = 1

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        if self.architecture == "cnn":
            if self.cnn_kernel_size <= 0:
                raise ValueError("cnn_kernel_size must be positive")
            if self.cnn_channels <= 0:
                raise ValueError("cnn_channels must be positive")


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 5
    min_delta: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: Literal["mse", "bce"] = "mse"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    num_workers: int = 0
    gradient_clip: Optional[float] = None

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.loss not in {"mse", "bce"}:
            raise ValueError("loss must be either 'mse' or 'bce'")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Sliding-window dataset for sequence-to-one forecasting."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int) -> None:
        if features.ndim != 2:
            raise ValueError("features must be a 2D array")
        if targets.ndim != 2:
            raise ValueError("targets must be a 2D array (n_samples, target_dims)")
        if len(features) != len(targets):
            raise ValueError("features and targets must contain the same number of rows")
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        if len(features) < lookback:
            raise ValueError(
                "Not enough rows to build a sequence: "
                f"received {len(features)} rows but lookback requires {lookback}"
            )

        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.lookback = lookback
        self.sequence_length = len(self.targets) - lookback + 1

    def __len__(self) -> int:
        return self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        start = idx
        end = idx + self.lookback
        feature_window = self.features[start:end]
        target_value = self.targets[end - 1]
        return torch.from_numpy(feature_window), torch.from_numpy(target_value)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class SequenceModel(nn.Module):
    """Wrapper that exposes LSTM/GRU/CNN models with a unified interface."""

    def __init__(self, input_size: int, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()

        self.cfg = cfg
        self.architecture = cfg.architecture
        self.output_size = cfg.output_size

        if self.architecture == "lstm":
            self.network = nn.LSTM(
                input_size=input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.architecture == "gru":
            self.network = nn.GRU(
                input_size=input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.architecture == "cnn":
            self.network = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=cfg.cnn_channels,
                    kernel_size=cfg.cnn_kernel_size,
                    padding=cfg.cnn_kernel_size // 2,
                ),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Conv1d(
                    in_channels=cfg.cnn_channels,
                    out_channels=cfg.cnn_channels,
                    kernel_size=cfg.cnn_kernel_size,
                    padding=cfg.cnn_kernel_size // 2,
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:  # pragma: no cover - configuration validation prevents reaching this
            raise ValueError(f"Unsupported architecture '{self.architecture}'")

        self.regressor = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size if self.architecture != "cnn" else cfg.cnn_channels, cfg.output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.architecture == "cnn":
            # Expect input as (batch, seq_len, features) -> transpose for Conv1d
            x = x.transpose(1, 2)  # (batch, features, seq_len)
            features = self.network(x)
            features = features.squeeze(-1)
        else:
            features, _ = self.network(x)
            features = features[:, -1, :]
        return self.regressor(features)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _read_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported file extension '{path.suffix}'. Use parquet or csv.")

    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index)
        except (TypeError, ValueError):  # pragma: no cover - best-effort conversion
            pass

    frame = frame.sort_index()
    return frame


def _infer_target_column(frame: pd.DataFrame) -> str:
    if frame.empty:
        raise ValueError("Cannot infer target column from an empty dataframe")

    candidates = [
        column
        for column in frame.columns
        if "target" in column.lower() or column.lower().startswith("label")
    ]
    if not candidates:
        raise ValueError(
            "Unable to infer a target column. Specify DataConfig.target_column explicitly."
        )
    return candidates[0]


def _prepare_arrays(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: DataConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, List[str], StandardScaler, Optional[StandardScaler]]:
    target_column = cfg.target_column or _infer_target_column(train)

    feature_columns = cfg.feature_columns or [col for col in train.columns if col != target_column]
    missing_features = [col for col in feature_columns if col not in train.columns]
    if missing_features:
        raise ValueError(f"Requested feature columns not present in training data: {missing_features}")

    for frame_name, frame in {"validation": val, "test": test}.items():
        missing = [col for col in feature_columns + [target_column] if col not in frame.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} missing from {frame_name} dataframe. Ensure all splits share the same schema."
            )

    train_features = train[feature_columns].to_numpy()
    val_features = val[feature_columns].to_numpy()
    test_features = test[feature_columns].to_numpy()

    train_targets = train[[target_column]].to_numpy()
    val_targets = val[[target_column]].to_numpy()
    test_targets = test[[target_column]].to_numpy()

    feature_scaler = StandardScaler().fit(train_features)
    train_features_scaled = feature_scaler.transform(train_features)
    val_features_scaled = feature_scaler.transform(val_features)
    test_features_scaled = feature_scaler.transform(test_features)

    target_scaler: Optional[StandardScaler] = None
    if cfg.scale_target:
        target_scaler = StandardScaler().fit(train_targets)
        train_targets_scaled = target_scaler.transform(train_targets)
        val_targets_scaled = target_scaler.transform(val_targets)
        test_targets_scaled = target_scaler.transform(test_targets)
    else:
        train_targets_scaled = train_targets
        val_targets_scaled = val_targets
        test_targets_scaled = test_targets

    return (
        train_features_scaled,
        val_features_scaled,
        test_features_scaled,
        train_targets_scaled,
        val_targets_scaled,
        test_targets_scaled,
        target_column,
        feature_columns,
        feature_scaler,
        target_scaler,
    )


# ---------------------------------------------------------------------------
# Training and evaluation routines
# ---------------------------------------------------------------------------


def _resolve_device(device_preference: Literal["cpu", "cuda", "auto"]) -> torch.device:
    if device_preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_preference)


def _create_dataloaders(
    train_arrays: Tuple[np.ndarray, np.ndarray],
    val_arrays: Tuple[np.ndarray, np.ndarray],
    test_arrays: Tuple[np.ndarray, np.ndarray],
    cfg: DataConfig,
    training_cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SequenceDataset(train_arrays[0], train_arrays[1], cfg.lookback)
    val_dataset = SequenceDataset(val_arrays[0], val_arrays[1], cfg.lookback)
    test_dataset = SequenceDataset(test_arrays[0], test_arrays[1], cfg.lookback)

    def make_loader(dataset: SequenceDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=training_cfg.batch_size,
            shuffle=shuffle,
            num_workers=training_cfg.num_workers,
            drop_last=False,
        )

    return make_loader(train_dataset, shuffle=True), make_loader(val_dataset, shuffle=False), make_loader(test_dataset, shuffle=False)


def _create_loss(loss_name: Literal["mse", "bce"]) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss '{loss_name}'")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: Optional[float],
) -> float:
    model.train()
    epoch_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = loss_fn(outputs, targets)
        loss.backward()

        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        batch_size = features.shape[0]
        epoch_loss += loss.item() * batch_size
        total_samples += batch_size

    return epoch_loss / max(total_samples, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    loss_name: Literal["mse", "bce"],
    target_scaler: Optional[StandardScaler] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    outputs_list: List[Tensor] = []
    targets_list: List[Tensor] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = loss_fn(logits, targets)

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            outputs_list.append(logits.cpu())
            targets_list.append(targets.cpu())

    if total_samples == 0:
        raise ValueError("Validation/Test loader does not contain any samples")

    outputs_tensor = torch.cat(outputs_list, dim=0)
    targets_tensor = torch.cat(targets_list, dim=0)

    metrics: Dict[str, float] = {"loss": total_loss / total_samples}

    if loss_name == "bce":
        probabilities = torch.sigmoid(outputs_tensor)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == targets_tensor).float().mean().item()
        positive_rate = targets_tensor.mean().item()
        metrics.update(
            {
                "accuracy": accuracy,
                "positive_rate": positive_rate,
            }
        )
    else:
        preds = outputs_tensor.numpy()
        targets_np = targets_tensor.numpy()
        if target_scaler is not None:
            preds = target_scaler.inverse_transform(preds)
            targets_np = target_scaler.inverse_transform(targets_np)
        mae = float(np.mean(np.abs(preds - targets_np)))
        rmse = float(math.sqrt(np.mean((preds - targets_np) ** 2)))
        metrics.update({"mae": mae, "rmse": rmse})

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_cfg: TrainingConfig,
    target_scaler: Optional[StandardScaler] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float], nn.Module]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.weight_decay
    )
    loss_fn = _create_loss(training_cfg.loss)
    device = _resolve_device(training_cfg.device)
    model.to(device)
    loss_fn = loss_fn.to(device)

    logger.info("Training on device: %s", device)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Tensor]] = None
    epochs_no_improve = 0

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, training_cfg.max_epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            training_cfg.gradient_clip,
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            training_cfg.loss,
            target_scaler=target_scaler,
        )
        val_loss = val_metrics["loss"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch %d/%d - train_loss: %.6f - val_loss: %.6f",
            epoch,
            training_cfg.max_epochs,
            train_loss,
            val_loss,
        )

        if val_loss + training_cfg.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            logger.info("New best model found at epoch %d (val_loss=%.6f)", epoch, val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve > training_cfg.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    epochs_no_improve,
                )
                break

    if best_state is None:  # Should not happen but keep a defensive guard
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    best_metrics = _evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        training_cfg.loss,
        target_scaler=target_scaler,
    )
    return history, best_metrics, model


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _dump_pickle(obj: object, path: Path) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
    else:  # pragma: no cover - fallback for environments without joblib
        import pickle

        with path.open("wb") as handle:
            pickle.dump(obj, handle)

def _json_safe(value):
    """Convert common non-serialisable objects into JSON-friendly types."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_training_pipeline(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    output_dir: Path = Path("models"),
    warm_start_checkpoint: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute the end-to-end modeling pipeline.

    The function returns a dictionary containing the training history, the best
    validation metrics, and the evaluation metrics on the held-out test split.
    Artefacts (model checkpoint, scalers, metrics, and configuration) are saved
    in ``output_dir``. When ''warm_start_checkpoint'' is provided the model
    weights are initialised from the checkpoint which enables continued
    training when new data becomes available.
    """

    logger.info("Starting training pipeline")
    training_cfg.validate()
    model_cfg.validate()

    train_path, val_path, test_path = data_cfg.resolve_paths()
    logger.info("Loading processed datasets from %s", data_cfg.data_dir)
    train_df = _read_dataframe(train_path)
    val_df = _read_dataframe(val_path)
    test_df = _read_dataframe(test_path)

    (
        train_features,
        val_features,
        test_features,
        train_targets,
        val_targets,
        test_targets,
        target_column,
        feature_columns,
        feature_scaler,
        target_scaler,
    ) = _prepare_arrays(train_df, val_df, test_df, data_cfg)

    train_loader, val_loader, test_loader = _create_dataloaders(
        (train_features, train_targets),
        (val_features, val_targets),
        (test_features, test_targets),
        data_cfg,
        training_cfg,
    )

    model = SequenceModel(input_size=len(feature_columns), cfg=model_cfg)

    if warm_start_checkpoint is not None:
        candidate = warm_start_checkpoint.resolve()
        if candidate.exists():
            state_dict = torch.load(candidate, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("Loaded warm start weights from %s", candidate)
        else:
            logger.warning("Warm start checkpoint %s not found; training from scratch", candidate)

    history, val_metrics, trained_model = train_model(
        model,
        train_loader,
        val_loader,
        training_cfg,
        target_scaler=target_scaler,
    )

    loss_fn = _create_loss(training_cfg.loss)
    device = _resolve_device(training_cfg.device)
    trained_model.to(device)
    test_metrics = _evaluate(
        trained_model,
        test_loader,
        loss_fn.to(device),
        device,
        training_cfg.loss,
        target_scaler=target_scaler,
    )
    trained_model.cpu()

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.pt"
    torch.save(trained_model.state_dict(), model_path)
    logger.info("Saved best model checkpoint to %s", model_path)

    feature_scaler_path = output_dir / "feature_scaler.pkl"
    _dump_pickle(feature_scaler, feature_scaler_path)
    logger.info("Saved feature scaler to %s", feature_scaler_path)

    target_scaler_path: Optional[Path] = None
    if target_scaler is not None:
        target_scaler_path = output_dir / "target_scaler.pkl"
        _dump_pickle(target_scaler, target_scaler_path)
        logger.info("Saved target scaler to %s", target_scaler_path)

    metadata = _json_safe(
        {
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "training_config": asdict(training_cfg),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        }
    )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved training metadata to %s", metrics_path)

    return {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "feature_scaler_path": feature_scaler_path,
        "target_scaler_path": target_scaler_path,
        "metrics_path": metrics_path,
    }


if __name__ == "__main__":  # pragma: no cover - convenience script usage
    logging.basicConfig(level=logging.INFO)
    data_config = DataConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    run_training_pipeline(data_config, model_config, training_config)