"""Monte Carlo European-call prices under geometric Brownian motion."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def simulate_terminal_spot(
    spot: float,
    maturity: float,
    rate: float,
    sigma: float,
    n_paths: int,
    rng: Generator,
    antithetic: bool = True,
) -> NDArray[np.float64]:
    """Exact GBM solution at T: S_T = S exp((r - σ²/2)T + σ√T Z)."""
    n_paths = int(n_paths)
    if antithetic:
        n_half = n_paths // 2
        z = rng.standard_normal(n_half)
        z = np.concatenate([z, -z])
        if n_paths % 2 == 1:
            z = np.concatenate([z, rng.standard_normal(1)])
    else:
        z = rng.standard_normal(n_paths)

    drift = (rate - 0.5 * sigma**2) * maturity
    shock = sigma * np.sqrt(maturity) * z
    return np.asarray(spot * np.exp(drift + shock), dtype=np.float64)


def mc_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    n_paths: int,
    rng: Generator,
    antithetic: bool = True,
) -> tuple[float, float]:
    """Discounted average payoff, plus the Monte Carlo standard error.

    C_MC = e^{-rT} * mean(max(S_T - K, 0))
    With independent paths, SE = std(discounted) / sqrt(N).
    With antithetic pairs (Z, -Z), SE uses the pair averages, not all N paths.
    """
    terminal = simulate_terminal_spot(
        spot, maturity, rate, sigma, n_paths, rng, antithetic=antithetic
    )
    payoff = np.maximum(terminal - strike, 0.0)
    discounted = np.exp(-rate * maturity) * payoff
    price = float(np.mean(discounted))
    n_half = discounted.size // 2
    if antithetic and n_half >= 2:
        # Z and -Z are paired; the independent observations are the pair means.
        paired = 0.5 * (discounted[:n_half] + discounted[n_half : 2 * n_half])
        stderr = float(np.std(paired, ddof=1) / np.sqrt(paired.size))
    else:
        stderr = float(np.std(discounted, ddof=1) / np.sqrt(discounted.size))
    return price, stderr


def mc_call_prices(
    spots: NDArray[np.float64],
    strikes: NDArray[np.float64],
    maturities: NDArray[np.float64],
    rates: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    n_paths: int,
    rng: Generator,
    antithetic: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Price many contracts one at a time (params differ, so paths are not shared)."""
    n = len(strikes)
    prices = np.empty(n, dtype=np.float64)
    stderrs = np.empty(n, dtype=np.float64)
    for i in range(n):
        prices[i], stderrs[i] = mc_call_price(
            float(spots[i]),
            float(strikes[i]),
            float(maturities[i]),
            float(rates[i]),
            float(sigmas[i]),
            n_paths,
            rng,
            antithetic=antithetic,
        )
    return prices, stderrs
