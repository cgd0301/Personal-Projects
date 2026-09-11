"""Unit checks: formula, GBM measure, Monte Carlo SE, edges, homogeneity."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import norm

from option_pricing_comparison.black_scholes import (
    call_price,
    d1_d2,
    put_call_parity_gap,
)
from option_pricing_comparison.compare import regression_metrics
from option_pricing_comparison.dataset import contract_features
from option_pricing_comparison.monte_carlo import mc_call_price, simulate_terminal_spot


def _textbook_call(spot: float, strike: float, maturity: float, rate: float, sigma: float) -> float:
    """Independent textbook evaluation of the same BS formula (not the package)."""
    vol_sqrt_t = sigma * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2))


def _pair_stderr(discounted: np.ndarray) -> float:
    n_half = discounted.size // 2
    paired = 0.5 * (discounted[:n_half] + discounted[n_half : 2 * n_half])
    return float(np.std(paired, ddof=1) / np.sqrt(paired.size))


class TestBlackScholesAssumptions(unittest.TestCase):
    def test_atm_matches_textbook_and_known_value(self) -> None:
        spot, strike, maturity, rate, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        got = float(call_price(spot, strike, maturity, rate, sigma))
        textbook = _textbook_call(spot, strike, maturity, rate, sigma)
        d1, d2 = d1_d2(spot, strike, maturity, rate, sigma)
        self.assertAlmostEqual(float(d1), 0.35, places=12)
        self.assertAlmostEqual(float(d2), 0.15, places=12)
        self.assertAlmostEqual(got, textbook, places=12)
        self.assertAlmostEqual(got, 10.450583572185565, places=9)

    def test_put_call_parity_grid(self) -> None:
        spots = np.array([80.0, 100.0, 120.0])
        strikes = np.array([70.0, 100.0, 130.0])
        gap = np.asarray(
            put_call_parity_gap(spots, strikes, 1.0, 0.05, 0.2), dtype=np.float64
        )
        self.assertLess(float(np.max(np.abs(gap))), 1e-12)

    def test_call_is_homogeneous_of_degree_one(self) -> None:
        base = float(call_price(100.0, 90.0, 1.25, 0.03, 0.25))
        scaled = float(call_price(200.0, 180.0, 1.25, 0.03, 0.25))
        features = contract_features(
            np.array([100.0, 200.0]),
            np.array([90.0, 180.0]),
            np.array([1.25, 1.25]),
            np.array([0.03, 0.03]),
            np.array([0.25, 0.25]),
        )
        self.assertAlmostEqual(scaled / base, 2.0, places=10)
        self.assertLess(float(np.max(np.abs(features[0] - features[1]))), 1e-12)

    def test_sigma_zero_and_expiry(self) -> None:
        atm_spot = float(call_price(100.0, 100.0, 1.0, 0.05, 0.0))
        forward_intrinsic = max(100.0 - 100.0 * np.exp(-0.05 * 1.0), 0.0)
        k_atm_fwd = 100.0 * np.exp(0.05 * 1.0)
        atm_fwd = float(call_price(100.0, k_atm_fwd, 1.0, 0.05, 0.0))
        expired_itm = float(call_price(120.0, 100.0, 0.0, 0.05, 0.2))
        expired_otm = float(call_price(80.0, 100.0, 0.0, 0.05, 0.2))
        self.assertTrue(np.isfinite(atm_spot))
        self.assertAlmostEqual(atm_spot, forward_intrinsic, places=8)
        self.assertAlmostEqual(expired_itm, 20.0, places=12)
        self.assertAlmostEqual(expired_otm, 0.0, places=12)
        self.assertTrue(np.isfinite(atm_fwd))
        self.assertAlmostEqual(atm_fwd, 0.0, places=8)


class TestMonteCarloAssumptions(unittest.TestCase):
    def test_risk_neutral_terminal_mean(self) -> None:
        rng = np.random.default_rng(42)
        spot, maturity, rate, sigma = 100.0, 1.0, 0.05, 0.20
        terminal = simulate_terminal_spot(
            spot, maturity, rate, sigma, 80_000, rng, antithetic=False
        )
        forward = spot * np.exp(rate * maturity)
        rel = abs(float(np.mean(terminal)) - forward) / forward
        self.assertLess(rel, 0.01)

    def test_mc_price_is_unbiased_for_bs(self) -> None:
        spot, strike, maturity, rate, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bs = float(call_price(spot, strike, maturity, rate, sigma))
        rng = np.random.default_rng(7)
        n_rep, n_paths = 60, 8_000
        prices = np.empty(n_rep)
        for i in range(n_rep):
            prices[i], _ = mc_call_price(
                spot, strike, maturity, rate, sigma, n_paths, rng, antithetic=True
            )
        emp_se = float(np.std(prices, ddof=1) / np.sqrt(n_rep))
        self.assertLess(abs(float(np.mean(prices)) - bs), 4.0 * emp_se)

    def test_antithetic_stderr_uses_pair_averages(self) -> None:
        """Reported SE must use antithetic pair averages, not IID /sqrt(N)."""
        spot, strike, maturity, rate, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        rng_mc = np.random.default_rng(11)
        rng_sim = np.random.default_rng(11)
        n_rep, n_paths = 80, 4_000
        prices = np.empty(n_rep)
        pair_ses = np.empty(n_rep)
        reported_ses = np.empty(n_rep)
        for i in range(n_rep):
            prices[i], reported_ses[i] = mc_call_price(
                spot, strike, maturity, rate, sigma, n_paths, rng_mc, antithetic=True
            )
            terminal = simulate_terminal_spot(
                spot, maturity, rate, sigma, n_paths, rng_sim, antithetic=True
            )
            discounted = np.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
            pair_ses[i] = _pair_stderr(discounted)
        empirical_se = float(np.std(prices, ddof=1))
        mean_reported = float(np.mean(reported_ses))
        self.assertLess(abs(mean_reported - float(np.mean(pair_ses))), 1e-12)
        self.assertLess(abs(mean_reported - empirical_se) / empirical_se, 0.25)

    def test_call_never_below_discounted_intrinsic(self) -> None:
        spots = np.linspace(70.0, 140.0, 15)
        strike, maturity, rate, sigma = 100.0, 0.75, 0.04, 0.3
        calls = np.asarray(call_price(spots, strike, maturity, rate, sigma), dtype=np.float64)
        floor = np.maximum(spots - strike * np.exp(-rate * maturity), 0.0)
        self.assertTrue(np.all(calls >= floor - 1e-10))


class TestMetricsAssumptions(unittest.TestCase):
    def test_relative_error_skips_tiny_benchmark_prices(self) -> None:
        pred = np.array([0.02, 10.0])
        truth = np.array([1e-12, 10.0])
        stats = regression_metrics(pred, truth, rel_floor=0.05)
        self.assertEqual(stats["n_rel"], 1)
        self.assertAlmostEqual(stats["mean_rel_error"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
