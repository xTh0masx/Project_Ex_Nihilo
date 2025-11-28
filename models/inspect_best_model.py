#!/usr/bin/env python3
"""Utility script to inspect PyTorch checkpoints (best_model.pt by default).

Usage examples (choose whichever you prefer):
    python inspect_best_model.py                       # uses default models/btc_usd/best_model.pt
    python inspect_best_model.py path/to/model.pt      # optional positional override
    python inspect_best_model.py --path C:/foo/model.pt # explicit --path flag
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch


def _format_shape(tensor: torch.Tensor) -> str:
    shape = tuple(tensor.shape)
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def describe_state_dict(state_dict: Mapping[str, torch.Tensor]) -> None:
    print(f"State dict has {len(state_dict)} tensors:")
    for name, tensor in state_dict.items():
        shape = _format_shape(tensor)
        print(f"- {name}: shape={shape}, dtype={tensor.dtype}")
        flat = tensor.flatten()
        preview = flat[:5].tolist()
        print(f"    first values: {preview}")


def describe_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    print("Top-level keys in checkpoint:", list(checkpoint.keys()))
    for scalar_key in ("epoch", "loss", "best_metric", "global_step"):
        if scalar_key in checkpoint:
            print(f"{scalar_key}: {checkpoint[scalar_key]}")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(state_dict, Mapping):
        describe_state_dict(state_dict)
    else:
        print("Checkpoint does not contain a mapping of tensors.")


def inspect_checkpoint(path: Path, map_location: str = "cpu") -> None:
    print(f"Loading checkpoint from {path} ...")
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, torch.jit.ScriptModule):
        print("Loaded TorchScript module:")
        print(checkpoint)
    elif isinstance(checkpoint, Mapping):
        describe_checkpoint(checkpoint)
    else:
        print(f"Unknown checkpoint type: {type(checkpoint)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint file.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Optional positional path to a checkpoint. Equivalent to --path.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to the checkpoint (.pt) file. Defaults to models/btc_usd/best_model.pt next to this script.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to map tensors to when loading (e.g., 'cpu', 'cuda').",
    )
    return parser


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    script_dir = Path(__file__).resolve().parent
    default_checkpoint = script_dir / "models" / "btc_usd" / "best_model.pt"
    raw_path = args.checkpoint or args.path or default_checkpoint
    checkpoint_path = raw_path.expanduser()

    if checkpoint_path.is_file():
        return checkpoint_path

    script_name = Path(__file__).name
    help_lines = [
        f"Checkpoint not found at {checkpoint_path}.",
        "To inspect a different file, rerun with one of:",
        f"  python {script_name} /full/path/to/your_model.pt",
        f"  python {script_name} --path /full/path/to/your_model.pt",
    ]
    if raw_path == default_checkpoint:
        help_lines.extend(
            [
                "The default models/btc_usd/best_model.pt is missing.",
                "Ensure the weights exist locally (e.g., run 'git lfs pull' if applicable).",
            ]
        )

    raise FileNotFoundError("\n".join(help_lines))


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint_path(args)
    inspect_checkpoint(checkpoint_path, map_location=args.device)