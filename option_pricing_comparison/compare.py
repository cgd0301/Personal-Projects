"""Error metrics versus the Black-Scholes benchmark."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_1d(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def regression_metrics(
    predicted: ArrayLike,
    benchmark: ArrayLike,
    rel_floor: float = 0.05,
) -> dict[str, float]:
    """MAE / RMSE / max absolute error, plus relative error on non-tiny prices.

    Deep OTM calls have Black-Scholes values near zero, so a 0.01 dollar gap
    would dominate a raw mean relative error. Relative stats are therefore
    averaged only where |C_BS| >= rel_floor.
    """
    pred = _as_1d(predicted)
    truth = _as_1d(benchmark)
    abs_err = np.abs(pred - truth)
    usable = np.abs(truth) >= rel_floor
    if np.any(usable):
        rel = abs_err[usable] / np.abs(truth[usable])
        mean_rel = float(np.mean(rel))
        median_rel = float(np.median(rel))
        n_rel = int(np.count_nonzero(usable))
    else:
        mean_rel = float("nan")
        median_rel = float("nan")
        n_rel = 0
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(abs_err**2))),
        "max_abs_error": float(np.max(abs_err)),
        "mean_rel_error": mean_rel,
        "median_rel_error": median_rel,
        "n_rel": n_rel,
        "n": int(pred.size),
    }
