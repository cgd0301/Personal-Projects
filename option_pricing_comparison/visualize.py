"""Five plain matplotlib figures for the pricing comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "-",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def plot_price_vs_strike(
    strikes: np.ndarray,
    bs_prices: np.ndarray,
    mc_prices: np.ndarray,
    nn_prices: np.ndarray,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(strikes, bs_prices, color="#1f4e79", lw=2.2, label="Black-Scholes")
    ax.plot(strikes, mc_prices, color="#c45911", lw=1.6, ls="--", label="Monte Carlo")
    ax.plot(strikes, nn_prices, color="#548235", lw=1.6, ls=":", label="Neural net")
    ax.set_xlabel("Strike")
    ax.set_ylabel("European call price")
    ax.set_title("Call price vs strike")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_mc_convergence(
    path_counts: np.ndarray,
    mean_abs_error: np.ndarray,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.loglog(path_counts, mean_abs_error, "o-", color="#c45911", lw=1.8, label="MC |error|")
    # Reference line with the theoretical 1/sqrt(N) slope, scaled to the first point.
    scale = mean_abs_error[0] * np.sqrt(path_counts[0])
    ref = scale / np.sqrt(path_counts)
    ax.loglog(path_counts, ref, color="#7f7f7f", ls="--", lw=1.4, label=r"$c / \sqrt{N}$")
    ax.set_xlabel("Number of Monte Carlo paths")
    ax.set_ylabel("Mean |MC − BS|")
    ax.set_title("Monte Carlo error shrinks like 1 / sqrt(N)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_nn_loss_curve(
    train_loss: list[float],
    val_loss: list[float],
    path: Path,
) -> Path:
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, train_loss, color="#1f4e79", lw=1.6, label="Train MSE")
    ax.plot(epochs, val_loss, color="#548235", lw=1.6, label="Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on C / K")
    ax.set_title("Neural-net training")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pred_vs_bs(
    bs_mc: np.ndarray,
    mc_prices: np.ndarray,
    bs_nn: np.ndarray,
    nn_prices: np.ndarray,
    path: Path,
) -> Path:
    lo = float(min(bs_mc.min(), bs_nn.min(), mc_prices.min(), nn_prices.min()))
    hi = float(max(bs_mc.max(), bs_nn.max(), mc_prices.max(), nn_prices.max()))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharex=True, sharey=True)
    panels = (
        (axes[0], bs_mc, mc_prices, "#c45911", "Monte Carlo"),
        (axes[1], bs_nn, nn_prices, "#548235", "Neural net"),
    )
    for ax, x, y, color, title in panels:
        ax.plot([lo, hi], [lo, hi], color="#7f7f7f", lw=1.2)
        ax.scatter(x, y, s=12, alpha=0.35, color=color, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("Black-Scholes price")
    axes[0].set_ylabel("Model price")
    fig.suptitle("Predicted price vs Black-Scholes", y=1.02)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_vs_moneyness(
    moneyness: np.ndarray,
    mc_abs_error: np.ndarray,
    nn_abs_error: np.ndarray,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(moneyness, mc_abs_error, color="#c45911", lw=1.8, label="Monte Carlo")
    ax.plot(moneyness, nn_abs_error, color="#548235", lw=1.8, label="Neural net")
    ax.axvline(1.0, color="#7f7f7f", ls=":", lw=1.2, label="ATM (K/S = 1)")
    ax.set_xlabel("Moneyness K / S")
    ax.set_ylabel("|model − Black-Scholes|")
    ax.set_title("Absolute error vs moneyness")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_all_figures(
    output_dir: Path,
    strike_table: pd.DataFrame,
    conv_paths: np.ndarray,
    conv_error: np.ndarray,
    train_loss: list[float],
    val_loss: list[float],
    scatter_mc_bs: np.ndarray,
    scatter_mc: np.ndarray,
    scatter_nn_bs: np.ndarray,
    scatter_nn: np.ndarray,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    moneyness = strike_table["strike"].to_numpy() / strike_table["spot"].to_numpy()
    return {
        "price_vs_strike": plot_price_vs_strike(
            strike_table["strike"].to_numpy(),
            strike_table["bs_price"].to_numpy(),
            strike_table["mc_price"].to_numpy(),
            strike_table["nn_price"].to_numpy(),
            output_dir / "price_vs_strike.png",
        ),
        "mc_convergence": plot_mc_convergence(
            conv_paths, conv_error, output_dir / "mc_convergence.png"
        ),
        "nn_loss_curve": plot_nn_loss_curve(
            train_loss, val_loss, output_dir / "nn_loss_curve.png"
        ),
        "pred_vs_bs": plot_pred_vs_bs(
            scatter_mc_bs,
            scatter_mc,
            scatter_nn_bs,
            scatter_nn,
            output_dir / "pred_vs_bs.png",
        ),
        "error_vs_moneyness": plot_error_vs_moneyness(
            moneyness,
            np.abs(strike_table["mc_price"] - strike_table["bs_price"]).to_numpy(),
            np.abs(strike_table["nn_price"] - strike_table["bs_price"]).to_numpy(),
            output_dir / "error_vs_moneyness.png",
        ),
    }
