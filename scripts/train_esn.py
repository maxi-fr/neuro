from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

from neuro.config import load_esn_config, resolve_artifact_dir, resolve_data_files
from neuro.predictor.train import train

if TYPE_CHECKING:
    from neuro.predictor.ridge import RidgeTrainingResult


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ESN training."""
    parser = argparse.ArgumentParser(description="Train an Echo State Network (ESN) predictor.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nn_predictor/esn.yaml"),
        help="Path to ESN predictor config file.",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Optional data directory override.")
    return parser.parse_args()


def main() -> None:
    """Train the ESN, save its numpy checkpoint and the training stats."""
    args = parse_args()
    cfg = load_esn_config(args.config)
    data_files = resolve_data_files(cfg, args.data_path)

    artifact_dir = resolve_artifact_dir(cfg.artifact, "esn")
    shutil.copy2(args.config, artifact_dir / args.config.name)

    result = cast("RidgeTrainingResult", train(cfg, data_files))
    result.save(artifact_dir)
    print(f"Validation rollout NMSE at horizon {cfg.model.horizon}: {result.rollout.pooled:.4f}", flush=True)
    print(f"Saved ESN predictor checkpoint -> {artifact_dir / 'model.npz'}", flush=True)


if __name__ == "__main__":
    main()
