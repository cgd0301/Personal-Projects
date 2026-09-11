"""Default study settings: one GBM world, three pricing methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class StudyConfig:
    seed: int = 42
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # Benchmark contract used for the strike curve and MC convergence plot.
    spot: float = 100.0
    rate: float = 0.05
    sigma: float = 0.20
    maturity: float = 1.0

    strike_min: float = 70.0
    strike_max: float = 130.0
    n_strikes: int = 25

    # Monte Carlo: 50k paths for the main comparison; a sweep shows 1/sqrt(N).
    mc_paths: int = 50_000
    mc_antithetic: bool = True
    mc_path_grid: tuple[int, ...] = (1_000, 5_000, 20_000, 50_000, 100_000)
    mc_convergence_reps: int = 8

    # Random contracts used to train and test the network.
    n_samples: int = 8_000
    train_frac: float = 0.70
    val_frac: float = 0.15
    spot_min: float = 80.0
    spot_max: float = 120.0
    strike_sample_min: float = 70.0
    strike_sample_max: float = 130.0
    t_min: float = 0.10
    t_max: float = 2.0
    r_min: float = 0.01
    r_max: float = 0.08
    sigma_min: float = 0.10
    sigma_max: float = 0.40

    # Tiny MLP: 4 -> 32 -> 32 -> 1, trained with Adam + MSE.
    hidden_size: int = 32
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3

    # Out-of-range volatility used only as a simple extrapolation check.
    extra_sigma_min: float = 0.45
    extra_sigma_max: float = 0.60
    n_extra: int = 400

    mc_test_paths: int = 20_000

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.mc_paths = max(100, int(self.mc_paths))
        self.epochs = max(1, int(self.epochs))
        self.n_samples = max(100, int(self.n_samples))
        self.seed = int(self.seed)
