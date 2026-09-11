"""Command-line entry: run the Black-Scholes / Monte Carlo / neural-net study."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, StudyConfig
from .pipeline import run_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Black-Scholes, Monte Carlo, and a small neural net "
            "on synthetic European call options."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and PNG outputs",
    )
    parser.add_argument("--seed", type=int, default=42, help="NumPy / PyTorch seed")
    parser.add_argument(
        "--mc-paths",
        type=int,
        default=50_000,
        help="Monte Carlo paths for the strike-grid comparison",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Neural-net training epochs",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8_000,
        help="Number of random contracts used to train/test the network",
    )
    parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="Disable antithetic variates in the Monte Carlo pricer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        mc_paths=args.mc_paths,
        epochs=args.epochs,
        n_samples=args.samples,
        mc_antithetic=not args.no_antithetic,
    )
    run_study(config)


if __name__ == "__main__":
    main()
