"""Config-driven PyTorch forecasting architectures.

This module exposes LSTM, GRU, and Transformer forecasters with validated model
configuration, deterministic weight initialization, and automatic device
selection. Configuration keys are normalized to remain compatible with the
repository's existing YAML schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class ModelConfigurationError(ValueError):
    """Raised when model configuration cannot produce a valid architecture."""


def get_device() -> torch.device:
    """Resolve the best available PyTorch device in CUDA → MPS → CPU order."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_weights(module: nn.Module) -> None:
    """Apply Xavier initialization to supported trainable layers."""

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    if isinstance(module, (nn.LSTM, nn.GRU)):
        for name, parameter in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)


class LSTMForecaster(nn.Module):
    """Sequence forecaster based on the final LSTM hidden state."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.apply(init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.fc(sequence[:, -1, :])


class GRUForecaster(nn.Module):
    """Sequence forecaster based on the final GRU hidden state."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.apply(init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(inputs)
        return self.fc(sequence[:, -1, :])


class TransformerForecaster(nn.Module):
    """Transformer encoder forecaster using the final encoded time step."""

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ModelConfigurationError("hidden_dim must be divisible by num_heads.")

        self.input_projection = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.apply(init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(inputs)
        encoded = self.transformer(encoded)
        return self.fc(encoded[:, -1, :])


def _positive_int(config: Mapping[str, Any], *keys: str, default: int | None = None) -> int:
    """Resolve a positive integer from the first available configuration key."""

    for key in keys:
        if key in config:
            value = int(config[key])
            break
    else:
        if default is None:
            raise ModelConfigurationError(f"Missing required model setting; expected one of {keys}.")
        value = default

    if value <= 0:
        raise ModelConfigurationError(f"{keys[0]} must be positive; received {value}.")
    return value


def _dropout(config: Mapping[str, Any]) -> float:
    value = float(config.get("dropout", 0.0))
    if not 0.0 <= value < 1.0:
        raise ModelConfigurationError("dropout must be in the range [0.0, 1.0).")
    return value


def build_model(
    config: Mapping[str, Any],
    input_size: int,
    output_size: int,
    *,
    device: torch.device | None = None,
) -> nn.Module:
    """Build a validated LSTM, GRU, or Transformer forecasting model.

    The repository YAML uses `hidden_units` and uppercase model names, while
    earlier code expected `hidden_dim` and lowercase names. Both forms are
    accepted to preserve compatibility and remove configuration drift.
    """

    if input_size <= 0 or output_size <= 0:
        raise ModelConfigurationError("input_size and output_size must be positive.")
    if "model" not in config or not isinstance(config["model"], Mapping):
        raise ModelConfigurationError("config must contain a 'model' mapping.")

    model_config = config["model"]
    model_type = str(model_config.get("type", "lstm")).strip().lower()
    hidden_dim = _positive_int(model_config, "hidden_dim", "hidden_units", default=64)
    num_layers = _positive_int(model_config, "num_layers", default=1)
    dropout = _dropout(model_config)

    if model_type == "lstm":
        model: nn.Module = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
        )
    elif model_type == "gru":
        model = GRUForecaster(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
        )
    elif model_type == "transformer":
        model = TransformerForecaster(
            input_size=input_size,
            num_heads=_positive_int(model_config, "num_heads", default=4),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
        )
    else:
        raise ModelConfigurationError(
            f"Unknown model type '{model_type}'. Expected LSTM, GRU, or Transformer."
        )

    return model.to(device or get_device())
