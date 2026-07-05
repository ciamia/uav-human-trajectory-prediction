from __future__ import annotations

from collections.abc import Mapping as RuntimeMapping
from collections.abc import Sequence as RuntimeSequence
from typing import Mapping, Sequence, Union

import torch


Condition = Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]


def _model_velocity(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    condition: Condition,
) -> torch.Tensor:
    if isinstance(condition, RuntimeMapping):
        return model(x, t, condition)
    if isinstance(condition, RuntimeSequence) and not isinstance(condition, torch.Tensor):
        return model(x, t, *condition)
    return model(x, t, condition)


def _check_inputs(noise: torch.Tensor, steps: int) -> None:
    if noise.ndim != 3:
        raise ValueError("noise must have shape [B, T, D]")
    if steps <= 0:
        raise ValueError("steps must be positive")


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    noise: torch.Tensor,
    condition: Condition,
    steps: int = 50,
) -> torch.Tensor:
    """Integrate dx/dt = v_theta(x, t, c) from Gaussian trajectory noise."""
    _check_inputs(noise, steps)
    x = noise.clone()
    dt = 1.0 / steps

    for step in range(steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
        velocity = _model_velocity(model, x, t, condition)
        x = x + dt * velocity
    return x


@torch.no_grad()
def heun_sample(
    model: torch.nn.Module,
    noise: torch.Tensor,
    condition: Condition,
    steps: int = 50,
) -> torch.Tensor:
    """Second-order predictor-corrector integration for dx/dt = v_theta(x, t, c)."""
    _check_inputs(noise, steps)
    x = noise.clone()
    dt = 1.0 / steps

    for step in range(steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
        t_next = torch.full((x.shape[0],), min((step + 1) * dt, 1.0), device=x.device, dtype=x.dtype)
        velocity = _model_velocity(model, x, t, condition)
        x_pred = x + dt * velocity
        velocity_next = _model_velocity(model, x_pred, t_next, condition)
        x = x + 0.5 * dt * (velocity + velocity_next)
    return x


def sample(
    model: torch.nn.Module,
    noise: torch.Tensor,
    condition: Condition,
    steps: int = 50,
    method: str = "heun",
) -> torch.Tensor:
    if method == "euler":
        return euler_sample(model, noise, condition, steps=steps)
    if method == "heun":
        return heun_sample(model, noise, condition, steps=steps)
    raise ValueError(f"Unknown sampler method: {method}")
