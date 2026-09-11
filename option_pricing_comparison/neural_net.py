"""Small MLP that learns the Black-Scholes call map by backpropagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import FeatureScaler


class OptionMLP(nn.Module):
    """4 → 32 → 32 → 1 network. Softplus keeps the predicted C/K non-negative."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class TrainResult:
    model: OptionMLP
    scaler: FeatureScaler
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_mse: float


def _mse(model: OptionMLP, features: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(features)
        return float(nn.functional.mse_loss(pred, targets).item())


def train_option_mlp(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    *,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TrainResult:
    """Fit an MLP with Adam and MSE. Gradients come from loss.backward()."""
    torch.manual_seed(seed)
    scaler = FeatureScaler().fit(train_features)
    x_train = torch.tensor(scaler.transform(train_features), dtype=torch.float32)
    y_train = torch.tensor(train_targets.reshape(-1, 1), dtype=torch.float32)
    x_val = torch.tensor(scaler.transform(val_features), dtype=torch.float32)
    y_val = torch.tensor(val_targets.reshape(-1, 1), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = OptionMLP(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_loss: list[float] = []
    val_loss: list[float] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()  # backprop: d(MSE)/d(weights)
            optimizer.step()
            running += float(loss.item()) * batch_x.size(0)
            seen += batch_x.size(0)
        train_mse = running / max(seen, 1)
        val_mse = _mse(model, x_val, y_val)
        train_loss.append(train_mse)
        val_loss.append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model,
        scaler=scaler,
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        best_val_mse=best_val,
    )


def predict_call_prices(
    result: TrainResult,
    features: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Network output is C/K; multiply by K to recover a currency price."""
    x = torch.tensor(result.scaler.transform(features), dtype=torch.float32)
    result.model.eval()
    with torch.no_grad():
        ratio = result.model(x).cpu().numpy().reshape(-1)
    return ratio * np.asarray(strikes, dtype=np.float64)
