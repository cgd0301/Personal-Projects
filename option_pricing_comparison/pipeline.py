"""Run the full comparison: sample, train, price, score, and plot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .black_scholes import call_price, put_call_parity_gap
from .compare import regression_metrics
from .config import StudyConfig
from .dataset import features_from_frame, sample_contracts, split_frame
from .monte_carlo import mc_call_price, mc_call_prices
from .neural_net import predict_call_prices, train_option_mlp
from .visualize import write_all_figures


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)


def _strike_grid(config: StudyConfig) -> np.ndarray:
    return np.linspace(config.strike_min, config.strike_max, config.n_strikes)


def _mc_convergence(config: StudyConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Average |MC − BS| over a few independent runs at each path count."""
    strike = config.spot
    bs = float(
        call_price(config.spot, strike, config.maturity, config.rate, config.sigma)
    )
    errors = []
    for n_paths in config.mc_path_grid:
        abs_errs = []
        for _ in range(config.mc_convergence_reps):
            price, _ = mc_call_price(
                config.spot,
                strike,
                config.maturity,
                config.rate,
                config.sigma,
                n_paths,
                rng,
                antithetic=config.mc_antithetic,
            )
            abs_errs.append(abs(price - bs))
        errors.append(float(np.mean(abs_errs)))
    return np.asarray(config.mc_path_grid, dtype=np.float64), np.asarray(errors)


def run_study(config: StudyConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    parity = float(
        put_call_parity_gap(
            config.spot, config.spot, config.maturity, config.rate, config.sigma
        )
    )
    print(f"Put-call parity gap on the ATM benchmark: {parity:.2e}")

    print("Sampling training contracts and fitting the neural net...")
    contracts = sample_contracts(config.n_samples, rng, config)
    train, val, test = split_frame(contracts, rng, config.train_frac, config.val_frac)
    trained = train_option_mlp(
        features_from_frame(train),
        train["target"].to_numpy(),
        features_from_frame(val),
        val["target"].to_numpy(),
        hidden_size=config.hidden_size,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    print(
        f"Best validation MSE at epoch {trained.best_epoch}: {trained.best_val_mse:.4e}"
    )

    test = test.copy()
    test["nn_price"] = predict_call_prices(
        trained, features_from_frame(test), test["strike"].to_numpy()
    )

    extra = sample_contracts(
        config.n_extra,
        rng,
        config,
        sigma_min=config.extra_sigma_min,
        sigma_max=config.extra_sigma_max,
    )
    extra["nn_price"] = predict_call_prices(
        trained, features_from_frame(extra), extra["strike"].to_numpy()
    )

    print("Pricing the strike grid with Monte Carlo...")
    strikes = _strike_grid(config)
    spots = np.full_like(strikes, config.spot)
    maturities = np.full_like(strikes, config.maturity)
    rates = np.full_like(strikes, config.rate)
    sigmas = np.full_like(strikes, config.sigma)
    bs_grid = np.asarray(
        call_price(spots, strikes, maturities, rates, sigmas), dtype=np.float64
    )
    mc_grid, mc_se = mc_call_prices(
        spots,
        strikes,
        maturities,
        rates,
        sigmas,
        config.mc_paths,
        rng,
        antithetic=config.mc_antithetic,
    )
    nn_grid = predict_call_prices(
        trained,
        np.column_stack([spots / strikes, maturities, rates, sigmas]),
        strikes,
    )
    strike_table = pd.DataFrame(
        {
            "spot": spots,
            "strike": strikes,
            "maturity": maturities,
            "rate": rates,
            "sigma": sigmas,
            "bs_price": bs_grid,
            "mc_price": mc_grid,
            "mc_stderr": mc_se,
            "nn_price": nn_grid,
        }
    )

    print("Pricing the test set with Monte Carlo...")
    test["mc_price"], test["mc_stderr"] = mc_call_prices(
        test["spot"].to_numpy(),
        test["strike"].to_numpy(),
        test["maturity"].to_numpy(),
        test["rate"].to_numpy(),
        test["sigma"].to_numpy(),
        config.mc_test_paths,
        rng,
        antithetic=config.mc_antithetic,
    )

    print("Running the Monte Carlo path-count sweep...")
    conv_paths, conv_error = _mc_convergence(config, rng)

    nn_test = regression_metrics(test["nn_price"], test["bs_price"])
    mc_test = regression_metrics(test["mc_price"], test["bs_price"])
    nn_grid_metrics = regression_metrics(nn_grid, bs_grid)
    mc_grid_metrics = regression_metrics(mc_grid, bs_grid)
    nn_extra = regression_metrics(extra["nn_price"], extra["bs_price"])

    atm_idx = int(np.argmin(np.abs(strikes - config.spot)))
    metrics = {
        "seed": config.seed,
        "put_call_parity_gap": parity,
        "mc_paths_grid": config.mc_paths,
        "mc_paths_test": config.mc_test_paths,
        "mc_antithetic": config.mc_antithetic,
        "nn_best_epoch": trained.best_epoch,
        "nn_best_val_mse": trained.best_val_mse,
        "benchmark": {
            "spot": config.spot,
            "rate": config.rate,
            "sigma": config.sigma,
            "maturity": config.maturity,
            "atm_bs_price": float(bs_grid[atm_idx]),
            "atm_mc_price": float(mc_grid[atm_idx]),
            "atm_nn_price": float(nn_grid[atm_idx]),
        },
        "strike_grid": {
            "monte_carlo": mc_grid_metrics,
            "neural_net": nn_grid_metrics,
        },
        "test_set": {
            "monte_carlo": mc_test,
            "neural_net": nn_test,
        },
        "nn_extrapolation_high_sigma": nn_extra,
        "mc_convergence": {
            "n_paths": [int(n) for n in conv_paths],
            "mean_abs_error": [float(e) for e in conv_error],
        },
    }

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    strike_csv = config.output_dir / "strike_comparison.csv"
    test_csv = config.output_dir / "test_predictions.csv"
    strike_table.to_csv(strike_csv, index=False)
    test.to_csv(test_csv, index=False)

    figures = write_all_figures(
        config.output_dir,
        strike_table,
        conv_paths,
        conv_error,
        trained.train_loss,
        trained.val_loss,
        test["bs_price"].to_numpy(),
        test["mc_price"].to_numpy(),
        test["bs_price"].to_numpy(),
        test["nn_price"].to_numpy(),
    )
    print(f"Wrote metrics and figures under {config.output_dir}")
    return {
        "metrics": metrics_path,
        "strike_csv": strike_csv,
        "test_csv": test_csv,
        **figures,
    }
