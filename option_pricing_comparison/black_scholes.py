"""Black-Scholes closed-form prices for European calls and puts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

_NEAR_ZERO_T = 1e-12


def _as_arrays(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    return np.broadcast_arrays(
        np.asarray(spot, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(maturity, dtype=np.float64),
        np.asarray(rate, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )


def _maybe_scalar(values: NDArray[np.float64], like: ArrayLike) -> NDArray[np.float64] | float:
    if np.ndim(like) == 0 and values.size == 1:
        return float(values)
    return values


def d1_d2(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return d1 and d2 from the Black-Scholes formula.

    d1 = [ln(S/K) + (r + σ²/2) T] / (σ √T)
    d2 = d1 - σ √T
    """
    spot, strike, maturity, rate, sigma = _as_arrays(spot, strike, maturity, rate, sigma)
    vol_sqrt_t = sigma * np.sqrt(np.maximum(maturity, _NEAR_ZERO_T))
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def call_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64] | float:
    """European call: C = S N(d1) - K e^{-rT} N(d2)."""
    spot_arr, strike_arr, maturity_arr, rate_arr, sigma_arr = _as_arrays(
        spot, strike, maturity, rate, sigma
    )
    d1, d2 = d1_d2(spot_arr, strike_arr, maturity_arr, rate_arr, sigma_arr)
    price = spot_arr * norm.cdf(d1) - strike_arr * np.exp(-rate_arr * maturity_arr) * norm.cdf(d2)
    # At expiry the formula collapses to the intrinsic value max(S - K, 0).
    expired = maturity_arr < _NEAR_ZERO_T
    if np.any(expired):
        price = np.where(expired, np.maximum(spot_arr - strike_arr, 0.0), price)
    return _maybe_scalar(price, spot)


def put_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64] | float:
    """European put: P = K e^{-rT} N(-d2) - S N(-d1). Used for a parity check."""
    spot_arr, strike_arr, maturity_arr, rate_arr, sigma_arr = _as_arrays(
        spot, strike, maturity, rate, sigma
    )
    d1, d2 = d1_d2(spot_arr, strike_arr, maturity_arr, rate_arr, sigma_arr)
    price = strike_arr * np.exp(-rate_arr * maturity_arr) * norm.cdf(-d2) - spot_arr * norm.cdf(-d1)
    expired = maturity_arr < _NEAR_ZERO_T
    if np.any(expired):
        price = np.where(expired, np.maximum(strike_arr - spot_arr, 0.0), price)
    return _maybe_scalar(price, spot)


def put_call_parity_gap(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64] | float:
    """C - P should equal the forward value of the stock minus the strike: S - K e^{-rT}."""
    call = np.asarray(call_price(spot, strike, maturity, rate, sigma), dtype=np.float64)
    put = np.asarray(put_price(spot, strike, maturity, rate, sigma), dtype=np.float64)
    spot_arr, strike_arr, maturity_arr, rate_arr, _ = _as_arrays(
        spot, strike, maturity, rate, sigma
    )
    gap = call - put - (spot_arr - strike_arr * np.exp(-rate_arr * maturity_arr))
    return _maybe_scalar(gap, spot)
