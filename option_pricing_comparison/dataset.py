"""Random European-call contracts and a train / validation / test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.random import Generator

from .black_scholes import call_price
from .config import StudyConfig


@dataclass
class FeatureScaler:
    """Standardize the four network inputs using training-set mean and std."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> FeatureScaler:
        self.mean = features.mean(axis=0)
        self.std = np.clip(features.std(axis=0), 1e-8, None)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler.fit must be called first.")
        return (features - self.mean) / self.std


def contract_features(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Dimensionless inputs: S/K, T, r, σ. Scaling by K makes prices homogeneous."""
    return np.column_stack(
        [
            np.asarray(spot, dtype=np.float64) / np.asarray(strike, dtype=np.float64),
            np.asarray(maturity, dtype=np.float64),
            np.asarray(rate, dtype=np.float64),
            np.asarray(sigma, dtype=np.float64),
        ]
    )


def sample_contracts(
    n: int,
    rng: Generator,
    config: StudyConfig,
    *,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
) -> pd.DataFrame:
    """Draw contract parameters uniformly from the configured box."""
    n = int(n)
    sigma_lo = config.sigma_min if sigma_min is None else sigma_min
    sigma_hi = config.sigma_max if sigma_max is None else sigma_max
    frame = pd.DataFrame(
        {
            "spot": rng.uniform(config.spot_min, config.spot_max, n),
            "strike": rng.uniform(config.strike_sample_min, config.strike_sample_max, n),
            "maturity": rng.uniform(config.t_min, config.t_max, n),
            "rate": rng.uniform(config.r_min, config.r_max, n),
            "sigma": rng.uniform(sigma_lo, sigma_hi, n),
        }
    )
    bs = call_price(
        frame["spot"].to_numpy(),
        frame["strike"].to_numpy(),
        frame["maturity"].to_numpy(),
        frame["rate"].to_numpy(),
        frame["sigma"].to_numpy(),
    )
    frame["bs_price"] = np.asarray(bs, dtype=np.float64)
    frame["target"] = frame["bs_price"] / frame["strike"]
    return frame


def features_from_frame(frame: pd.DataFrame) -> np.ndarray:
    return contract_features(
        frame["spot"].to_numpy(),
        frame["strike"].to_numpy(),
        frame["maturity"].to_numpy(),
        frame["rate"].to_numpy(),
        frame["sigma"].to_numpy(),
    )


def split_frame(
    frame: pd.DataFrame,
    rng: Generator,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle once, then cut 70 / 15 / 15 (or the fractions in config)."""
    n = len(frame)
    order = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = frame.iloc[order[:n_train]].reset_index(drop=True)
    val = frame.iloc[order[n_train : n_train + n_val]].reset_index(drop=True)
    test = frame.iloc[order[n_train + n_val :]].reset_index(drop=True)
    return train, val, test
